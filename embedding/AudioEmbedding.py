from transformers import Wav2Vec2Processor, Data2VecAudioForCTC
import os
import torch
import soundfile
import tqdm
from utils import resample
from pathlib import Path
import logging
from utils import types
from utils.profiling import get_memory_mb, get_gpu_memory_mb, get_available_memory_mb
import math
from pathlib import Path
from embedding.Transcription import Transcription
from embedding.Pipeline import Pipeline


log_folder = Path("log")
log_folder.mkdir(exist_ok=True)
log_file = log_folder / "data2vec.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class AudioEmbedding(Pipeline):
    def __init__(
        self, input_folder_path, output_folder_path, config: types.TransformerConfig
    ):
        super().__init__(input_folder_path, output_folder_path, config)

        self.processor = config.processor_class.from_pretrained(config.processor_name)
        self.model = config.model_class.from_pretrained(config.model_name)
        logger.info("Models loaded successfully")
        # Get audio files
        self.audio_files = [
            f
            for f in os.listdir(self.input_folder_path)
            if f.endswith(".wav") or f.endswith(".flac")
        ]

        # Process files with memory logging
        logger.info(f"Processing {len(self.audio_files)} audio files...")
        self.results = self.process_files()
        logger.info(f"Processed {len(self.results)} files successfully")

        self.transcriber = Transcription(
            input_folder_path, output_folder_path, config, self.results
        )

    def process_audio_chunk(self, audio_data, sample_rate):
        """
        Process a single audio chunk through the model.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            logits tensor from the model
        """
        # Resample if needed
        if sample_rate != 16000:
            audio_data = resample.resample(audio_data, sample_rate)
        # if (self.denoiser):
        #     audio_data = self.denoiser.denoise_deepfilternet(audio_data,sample_rate)

        # Preprocess and convert audio to input tensor
        input_values = self.processor(
            audio_data, return_tensors="pt", sampling_rate=16000, padding="longest"
        ).input_values

        # Forward pass
        with torch.no_grad():
            logits = self.model(input_values).logits

        return logits

    def needs_streaming(self, file_path, max_audio_sec=150):
        info = soundfile.info(file_path)
        duration = info.duration
        needs_stream = duration > max_audio_sec
        return needs_stream, duration, max_audio_sec

    def process_file_streaming(
        self, file_path, max_chunk_sec: float = 60.0, min_chunk_sec: float = 10.0
    ):
        """Process large files in manageable chunks"""
        logger.info(f"Streaming large file: {file_path}")
        results = []

        info = soundfile.info(file_path)
        total_duration = info.frames / info.samplerate
        chunk_duration = min(max_chunk_sec, max(min_chunk_sec, total_duration))
        chunk_samples = int(chunk_duration * info.samplerate)
        total_chunks = math.ceil(info.frames / chunk_samples)
        logger.info(
            f"Processing {os.path.basename(file_path)} in {total_chunks} chunks of {chunk_duration:.1f}s each"
        )

        with soundfile.SoundFile(file_path) as f:
            chunk_num = 0
            chunk_pbar = tqdm.tqdm(
                total=total_chunks,
                desc=f"Chunks for {os.path.basename(file_path)}",
                leave=False,
            )

            while True:
                chunk = f.read(chunk_samples, dtype="float32")
                if len(chunk) == 0:
                    break

                chunk_num += 1

                # Process through model
                logits = self.process_audio_chunk(chunk, info.samplerate)

                chunk_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_chunk_{chunk_num:03d}"
                results.append(
                    {
                        "filename": chunk_filename,
                        "logits": logits.cpu(),
                        "original_sample_rate": info.samplerate,
                        "is_chunk": True,
                        "chunk_number": chunk_num,
                    }
                )

                logger.info(
                    f"Chunk {chunk_num}/{total_chunks}: {len(chunk)/16000:.1f}s processed"
                )
                chunk_pbar.update(1)

                if chunk_num % 5 == 0:
                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            chunk_pbar.close()

        return results

    def process_files(self):
        """Process files and log memory usage for each file"""
        results = []

        for i, file_name in enumerate(
            tqdm.tqdm(self.audio_files, desc="Processing audio files")
        ):
            mem_before = get_memory_mb()
            gpu_mem_before = get_gpu_memory_mb()

            file_path = os.path.join(self.input_folder_path, file_name)
            use_streaming, _, _ = self.needs_streaming(file_path)

            if use_streaming:
                try:
                    chunk_results = self.process_file_streaming(file_path)
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Error streaming {file_name}: {e}")
                    continue
            else:
                try:
                    data, sample_rate = soundfile.read(file_path)

                    # Process through model
                    logits = self.process_audio_chunk(data, sample_rate)

                    results.append(
                        {
                            "filename": file_name,
                            "logits": logits,
                            "original_sample_rate": sample_rate,
                            "is_chunk": False,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")
                    continue

            mem_after = get_memory_mb()
            gpu_mem_after = get_gpu_memory_mb()
            log_msg = f"File {i+1}/{len(self.audio_files)} -> {file_name}: CPU: {mem_before:.1f} -> {mem_after:.1f} MB (+{mem_after-mem_before:.1f})"
            if torch.cuda.is_available():
                log_msg += f", GPU: {gpu_mem_before:.1f} -> {gpu_mem_after:.1f} MB (+{gpu_mem_after-gpu_mem_before:.1f})"
            logger.info(log_msg)

        return results

    def clear_logits(self):
        """Clear logits from memory to free up space"""
        mem_before = get_memory_mb()
        for result in self.results:
            if "logits" in result:
                del result["logits"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mem_after = get_memory_mb()
        logger.info(
            f"Cleared logits from memory: {mem_before:.1f} -> {mem_after:.1f} MB "
            f"({mem_before-mem_after:.1f} MB freed)"
        )

    def save_logits_to_file(self):
        """Save all logits tensors to .pt files"""

        # Create logits folder if it doesn't exist
        logits_folder = Path(self.output_folder_path) / "logits"
        logits_folder.mkdir(exist_ok=True, parents=True)

        logger.info(f"Saving {len(self.results)} logits to {logits_folder}")

        for result in tqdm.tqdm(self.results, desc="Saving logits"):
            filename = result["filename"]
            logits = result["logits"]

            # Remove extension and add _logits suffix
            base_name = os.path.splitext(filename)[0]
            output_path = logits_folder / f"{base_name}_logits.pt"

            try:
                torch.save(logits.cpu(), output_path)
            except Exception as e:
                logger.error(f"Error saving weights for {filename}: {e}")

        logger.info(f"Finished saving logits to {logits_folder}")

    def transcribe_pipeline(self):
        self.transcriber.create_asr_transcriptions()
        self.transcriber.cleanup_and_merge_transcripts(self.output_folder_path)
