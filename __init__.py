import argparse
from embedding.AudioEmbedding import AudioEmbedding
from transformers import Wav2Vec2Processor, Data2VecAudioForCTC, Wav2Vec2ForCTC
from utils import types
import os
from dataclasses import dataclass
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Audio Embedding Pipeline")
    parser.add_argument('-t', '--transcribe', action='store_true', help='Run ASR transcription')
    parser.add_argument('-s', '--save-weights', action='store_true', help='Save final layer of model')
    parser.add_argument('-l', '--large', action='store_true', help='Use large Data2Vec model')
    parser.add_argument('--input-folder', required=True, help='Path to input folder (required)')
    parser.add_argument('--output-folder', required=True, help='Path to output folder (required)')

    if len(os.sys.argv) == 1:
        parser.print_usage()
        exit(1)

    args = parser.parse_args()
    INPUT_FOLDER_PATH = args.input_folder
    OUTPUT_FOLDER_PATH = args.output_folder

    transformer_config_wav2vec_base = types.TransformerConfig(
        processor_class=Wav2Vec2Processor,
        processor_name="facebook/wav2vec2-base-960h",
        model_class=Wav2Vec2ForCTC,
        model_name="facebook/wav2vec2-base-960h"
    )

    transformer_config_data2vec_large = types.TransformerConfig(
        processor_class=Wav2Vec2Processor,
        processor_name="facebook/wav2vec2-large-960h-lv60-self",
        model_class=Wav2Vec2ForCTC,
        model_name="facebook/wav2vec2-large-960h-lv60-self"
    )

        
    if args.large:
        selected_config = transformer_config_data2vec_large
    else:
        selected_config = transformer_config_wav2vec_base

    audio_embedding = AudioEmbedding(INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH, selected_config)
    if args.save_weights:
        audio_embedding.save_logits_to_file()
    if args.transcribe:
        audio_embedding.transcribe_pipeline()

if __name__ == "__main__":
    main()