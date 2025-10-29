# Audio Embeddings
This repository contains example workflows, READMEs, sample data, and [Docker](https://www.docker.com/) files that facilitate the usage of various open-source voice feature extraction packages, tools, datasets, and models for generating audio embeddings.

It is a part of a larger [toolkit](https://github.com/FHS-BAP/Voice-Feature-Extraction-Toolkit/) that was developed to support scientific research surrounding investigations of relationships between brain aging and voice features, although the extraction of voice features does have wider applicability. We invite others to please offer their questions, ideas, feedback, and improvements on this repository.
## Available embedding models
| Name         | Description |
|--------------|-------------|
| **wav2vec2-base-960h** | The base model pretrained and fine-tuned on 960 hours of Librispeech on 16kHz sampled speech audio [wav2vec2](https://huggingface.co/facebook/wav2vec2-base-960h). |
| **wav2vec2-large-960h-lv60-self** | The large model pretrained and fine-tuned on 960 hours of Libri-Light and Librispeech on 16kHz sampled speech audio. Model was trained with Self-Training objective. [wav2vec2](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)|

These models are available publicly on HuggingFace and will be downloaded automatically.

## System requirements
Data2Vec in this configuration is loaded into system RAM. Running the model itself takes approximately 1GB of memory.

The pipeline effectively chunks long audio recordings depending on their length but 1GB plus 500MB of additional overhead memory is required.

This is tested on Python 3.11 running on debian systems and MacOS. Docker configurations are provided as well.

## Usage
### Local Setup

1. **Prepare your data**
   
   Place your audio recordings in a directory (e.g., `data2wav_in`).

2. **Set up Python environment**
   
   - Create a conda environment:
     ```sh
     conda create -n audio-embeddings python=3.11
     conda activate audio-embeddings
     ```
   
   - Install dependencies:
     ```sh
     pip install -r requirements.txt
     ```

3. **Run the pipeline**
   
   Basic usage:
   ```sh
   python __init__.py --input-folder data2wav_in --output-folder data2wav_out
   ```
   
   With optional features:
   ```sh
   python __init__.py --input-folder data2wav_in --output-folder data2wav_out --transcribe --save-weights
   ```
   
   **Available arguments:**
   - `--input-folder`: Path to input audio files (required)
   - `--output-folder`: Path for output files (required)
   - `--transcribe`: Generate transcription files from audio
   - `--save-weights`: Save model output logits as .pt files in `output-folder/logits/`

### Docker Setup

1. **Configure volumes in `compose.yml`**
   
   ```yaml
   volumes:
     - ./data2wav_in:/input
     - ./data2wav_out:/output
   ```

2. **Build and start containers**
   
   ```sh
   docker compose build
   docker compose up (or -d if you want to run in the background)
   ```

   If you want to edit the command with arguments etc. change the command line in the compose file. The first time you run the compose if you don't have a .cache directory you will have to download the model weights from HuggingFace which can take time depending on your internet speeds.

## Output Structure

After running the pipeline, your output folder will contain:

```
data2wav_out/
├── logits/                          # Model weights (if --save-weights used)
│   ├── audio1_logits.pt
│   └── audio2_logits.pt
└── audio1_transcription.txt         # Transcriptions (if --transcribe used)
```