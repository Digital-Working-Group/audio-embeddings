# Audio Embeddings
This repository contains example workflows, READMEs, sample data, and [Docker](https://www.docker.com/) files that facilitate the usage of various open-source voice feature extraction packages, tools, datasets, and models for generating audio embeddings.

It is a part of a larger [toolkit](https://github.com/FHS-BAP/Voice-Feature-Extraction-Toolkit/) that was developed to support scientific research surrounding investigations of relationships between brain aging and voice features, although the extraction of voice features does have wider applicability. We invite others to please offer their questions, ideas, feedback, and improvements on this repository.

## Overview
| Name | Description |
| - |-|
| **data2vec** | Create audio embeddings via self-supervised learning via [data2vec](https://huggingface.co/docs/transformers/en/model_doc/data2vec) and/or [wav2vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2).

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

1. **Configure volumes in `docker-compose.yml`**
   
   ```yaml
   volumes:
     - ./data2wav_in:/input
     - ./data2wav_out:/output
   ```

2. **Build and start containers**
   
   ```sh
   docker compose build
   docker compose up -d
   ```

## Output Structure

After running the pipeline, your output folder will contain:

```
data2wav_out/
├── logits/                          # Model weights (if --save-weights used)
│   ├── audio1_logits.pt
│   └── audio2_logits.pt
└── audio1_transcription.txt         # Transcriptions (if --transcribe used)
```