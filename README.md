# AI Audio Transcription, Translate API

This project delivers an API for high-accuracy audio transcription and multilingual translation, powered by state-of-the-art (SOTA) machine learning models. The transcription component processes audio input, returning detailed text and timestamp data. The translation module allows text input and provides translated output in the target language. Both models are fine-tuned to enhance performance in real-world applications.

## Features

- **Text Translation**: (Updating...)
- **Audio Transcription**: Converts audio files (e.g., `.wav`, `.mp3`) into text using the **Faster-Whisper large-v3 model**.
- **FastAPI Integration**: The API is built using **FastAPI** and provides endpoints for transcription.

## Requirements

- **Python 3.8+**
- **FastAPI**
- **Uvicorn**
- **WhisperModel** from faster-whisper
- **PyTorch** (optional for GPU support)
- **transformers**

## Installation

### Clone the repository:

```bash
git clone https://github.com/meta-node-blockchain/AI-translate.git
cd AI-translate
```

### Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
### Install dependencies:

Install the necessary libraries by running:

```bash
pip install -r requirements.txt
```
Current requirements.txt are using torch for `cpu`, if you use `gpu`, download torch suitable for your `gpu`. Visit: https://pytorch.org/
```bash
#Example dowload torch using CUDA 12.1
pip uninstall torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Download the facebook/nllb-200-3.3B model:
Install gdown 
```bash
pip install gdown
```
Download translate model
```bash
gdown https://drive.google.com/uc?id=15XXvrJRcaCgYgraaya3cm6rPVs8Tt65F
```
Unzip file 
```bash
tar -xf facebook.zip
```

### Download the Whisper model:
By default, the API uses the "large-v3" Whisper model. This model will be downloaded automatically the first time you run the application.

## Usage

### Run the server:

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app 
```

### Or you can run by using run.py:

Run `run.py`:

```bash
python run.py
```
The server will be running at http://127.0.0.1:3000.
