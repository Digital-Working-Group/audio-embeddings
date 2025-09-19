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
import re
log_folder = Path("log")
log_folder.mkdir(exist_ok=True)
log_file = log_folder / "data2vec.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AudioEmbedding:
    def __init__(self, input_folder_path, output_folder_path, config: types.TransformerConfig):
        # Ensure the output folder exists, create if not
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
            
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.processor = config.processor_class.from_pretrained(config.processor_name)
        self.model = config.model_class.from_pretrained(config.model_name)
        
        # Get audio files
        self.audio_files = [f for f in os.listdir(input_folder_path) 
                           if f.endswith('.wav') or f.endswith('.flac')]
        
        # Process files with memory logging
        logger.info(f"Processing {len(self.audio_files)} audio files...")
        self.results = self.process_files()
        logger.info(f"Processed {len(self.results)} files successfully")

    def needs_streaming(self, file_path, max_audio_sec=150):
        info = soundfile.info(file_path)
        duration = info.duration
        needs_stream = duration > max_audio_sec
        return needs_stream, duration, max_audio_sec



    def process_file_streaming(self, file_path, max_chunk_sec: float = 60.0, min_chunk_sec: float = 10.0):
        """Process large files in manageable chunks"""
        logger.info(f"Streaming large file: {file_path}")
        results = []

        info = soundfile.info(file_path)

        # duration in seconds
        total_duration = info.frames / info.samplerate

        # determine chunk duration
        # if the file is long, chunk; otherwise, use the whole file
        chunk_duration = min(max_chunk_sec, max(min_chunk_sec, total_duration))
        chunk_samples = int(chunk_duration * info.samplerate)

        total_chunks = math.ceil(info.frames / chunk_samples)
        logger.info(f"Processing {os.path.basename(file_path)} in {total_chunks} chunks of {chunk_duration:.1f}s each")

        with soundfile.SoundFile(file_path) as f:
            chunk_num = 0
            chunk_pbar = tqdm.tqdm(total=total_chunks, desc=f"Chunks for {os.path.basename(file_path)}", leave=False)

            while True:
                chunk = f.read(chunk_samples, dtype="float32")
                if len(chunk) == 0:
                    break

                chunk_num += 1

                # resample if needed
                if info.samplerate != 16000:
                    chunk = resample.resample(chunk, info.samplerate)

                # convert to tensor
                input_values = self.processor(
                    chunk,
                    return_tensors="pt",
                    sampling_rate=16000,
                    padding="longest"
                ).input_values

                # forward pass
                with torch.no_grad():
                    logits = self.model(input_values).logits

                # store metadata + results
                chunk_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_chunk_{chunk_num:03d}"
                results.append({
                    "filename": chunk_filename,
                    "logits": logits.cpu(),  # move off GPU if CUDA available
                    "original_sample_rate": info.samplerate,
                    "is_chunk": True,
                    "chunk_number": chunk_num
                })

                logger.info(f"Chunk {chunk_num}/{total_chunks}: {len(chunk)/16000:.1f}s processed")

                chunk_pbar.update(1)

                # clean up periodically
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
        
        for i, file_name in enumerate(tqdm.tqdm(self.audio_files, desc="Processing audio files")):
            # Memory before processing this file
            mem_before = get_memory_mb()
            gpu_mem_before = get_gpu_memory_mb()
            
            file_path = os.path.join(self.input_folder_path, file_name)
            
            # Check if file will use too much memory (dynamic based on available memory)
            use_streaming = self.needs_streaming(file_path)
            
            if use_streaming:

                try:
                    chunk_results = self.process_file_streaming(file_path)
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Error streaming {file_name}: {e}")
                    continue
            else:
                # Read and process the audio file normally
                try:
                    data, sample_rate = soundfile.read(file_path)
                    
                    if sample_rate != 16000:
                        data = resample.resample(data, sample_rate)
                        
                except Exception as e:
                    logger.error(f"Error reading {file_name}: {e}")
                    continue
                
                # Process with model
                try:
                    input_values = self.processor(data, return_tensors="pt", padding="longest",
                                                sampling_rate=16000).input_values
                    
                    with torch.no_grad():
                        logits = self.model(input_values).logits
                    
                    # Store results
                    results.append({
                        'filename': file_name,
                        'logits': logits,
                        'original_sample_rate': sample_rate,
                        'is_chunk': False
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")
                    continue
            
            # Memory after processing this file
            mem_after = get_memory_mb()
            gpu_mem_after = get_gpu_memory_mb()
            
            # Log memory usage
            log_msg = f"File {i+1}/{len(self.audio_files)} -> {file_name}: CPU: {mem_before:.1f} -> {mem_after:.1f} MB (+{mem_after-mem_before:.1f})"
            if torch.cuda.is_available():
                log_msg += f", GPU: {gpu_mem_before:.1f} -> {gpu_mem_after:.1f} MB (+{gpu_mem_after-gpu_mem_before:.1f})"
            logger.info(log_msg)
        
        return results

    def _transcribe_logits(self, file_name, logits: torch.Tensor):
        """Transcribe logits to text and save to file"""
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        
        base_name = os.path.splitext(file_name)[0]
        output_file_path = os.path.join(
            self.output_folder_path, base_name + '_transcription.txt')
        
        with open(output_file_path, 'w') as f:
            f.write(transcription[0])

    def create_asr_transcriptions(self):
        """Create transcriptions from the in-memory logits"""
        logger.info("Creating transcriptions from processed logits...")
        
        for result in self.results:
            file_name = result['filename']
            logits = result['logits']
            
            try:
                self._transcribe_logits(file_name, logits)
            except Exception as e:
                logger.error(f"Failed to transcribe {file_name}: {e}")

    def clear_logits(self):
        """Clear logits from memory to free up space"""
        mem_before = get_memory_mb()
        
        for result in self.results:
            if 'logits' in result:
                del result['logits']
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        mem_after = get_memory_mb()
        logger.info(f"Cleared logits from memory: {mem_before:.1f} -> {mem_after:.1f} MB "
                   f"({mem_before-mem_after:.1f} MB freed)")

    def get_transcription(self, file_name):
        """Get transcription for a specific file without saving to disk"""
        for result in self.results:
            if result['filename'] == file_name:
                logits = result['logits']
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)
                return transcription[0]
        
        raise ValueError(f"File {file_name} not found in processed results")

    def get_all_transcriptions(self):
        """Get all transcriptions as a dictionary"""
        transcriptions = {}
        
        for result in self.results:
            file_name = result['filename']
            logits = result['logits']
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)
            transcriptions[file_name] = transcription[0]
        
        return transcriptions
    


    def cleanup_and_merge_transcripts(self,output_folder, delete_chunks=True):
        """Merge chunk transcription files and clean up"""
        folder = Path(output_folder)
        
        # Find chunk files: filename_chunk_001_transcription.txt
        chunk_files = {}
        for file in folder.glob("*_chunk_*_transcription.txt"):
            match = re.match(r'(.+)_chunk_(\d+)_transcription\.txt', file.name)
            if match:
                original_name, chunk_num = match.groups()
                if original_name not in chunk_files:
                    chunk_files[original_name] = []
                chunk_files[original_name].append((int(chunk_num), file))
        
        # Merge each group
        for original_name, chunks in chunk_files.items():
            chunks.sort()  # Sort by chunk number
            
            # Read and merge content
            merged_text = []
            for _, chunk_file in chunks:
                with open(chunk_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        merged_text.append(content)
            
            # Write merged file
            merged_file = folder / f"{original_name}_transcription.txt"
            with open(merged_file, 'w') as f:
                f.write('\n'.join(merged_text))
            
            # Clean up chunks
            for _, chunk_file in chunks:
                if delete_chunks:
                    chunk_file.unlink()
                else:
                    chunk_folder = folder / "chunks"
                    chunk_folder.mkdir(exist_ok=True)
                    chunk_file.rename(chunk_folder / chunk_file.name)
            
            logger.info(f"Merged {len(chunks)} chunks for {original_name}")