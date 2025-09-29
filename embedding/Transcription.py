import torch
from pathlib import Path
import logging
import os
import re
from embedding.Pipeline import Pipeline
logger = logging.getLogger(__name__)


class Transcription(Pipeline):
    """Handles the ML pipeline for transcription using Wav2Vec models"""

    def __init__(self, input_folder_path, output_folder_path, config):
        super().__init__(input_folder_path, output_folder_path, config)
        self.processor = config.processor_class.from_pretrained(config.processor_name)
        self.model = config.model_class.from_pretrained(config.model_name)
        self.results = []

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

    def cleanup_and_merge_transcripts(self, output_folder, delete_chunks=True):
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
