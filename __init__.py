from embedding.AudioEmbedding import AudioEmbedding
from transformers import Wav2Vec2Processor, Data2VecAudioForCTC,Wav2Vec2ForCTC
from utils import types
import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()
INPUT_FOLDER_PATH = os.environ.get('INPUT_FOLDER_PATH')
OUTPUT_FOLDER_PATH = os.environ.get('OUTPUT_FOLDER_PATH')

def main():    
    

    transformer_config_wav2vec = types.TransformerConfig(
        processor_class=Wav2Vec2Processor,
        processor_name="facebook/wav2vec2-base-960h",
        model_class=Wav2Vec2ForCTC,
        model_name="facebook/wav2vec2-base-960h"
    )

    transformer_config_data2vec = types.TransformerConfig(
        processor_class=Wav2Vec2Processor,
        processor_name="facebook/wav2vec2-base-960h",
        model_class=Data2VecAudioForCTC,
        model_name="facebook/data2vec-audio-base-100h"
    )
    audio_embedding = AudioEmbedding(INPUT_FOLDER_PATH,OUTPUT_FOLDER_PATH, transformer_config_wav2vec)

    audio_embedding.create_asr_transcriptions()
    audio_embedding.cleanup_and_merge_transcripts(OUTPUT_FOLDER_PATH)

if __name__ == "__main__":
    main()