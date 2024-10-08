import os
from dotenv import load_dotenv
import torch

# Load biến môi trường từ file .env
load_dotenv()


class Settings:
    model_name: str = os.getenv("MODEL_NAME", "Helsinki-NLP/opus-mt-en-vi")
    max_length: int = int(os.getenv("MAX_LENGTH", 512))
    num_return_sequences: int = int(os.getenv("NUM_RETURN_SEQUENCES", 1))
    hf_token = os.getenv('HF_TOKEN')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transcript_model_name: str = os.getenv("TRANSCRIPT_MODEL_NAME", "medium") if torch.cuda.is_available() else 'small'
    transcript_dtype = os.getenv('TRANSCRIPT_DTYPE') if torch.cuda.is_available() else 'int8'
    beam_size = 5
    file_audio_size = 1024 * 1024 * int(os.getenv('MAX_FILE_SIZE'))
    min_silence_duration_ms = os.getenv('MIN_SILENCE_DURATION_MS')
    # Cấu hình server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))


print("Is cuda:", torch.cuda.is_available())
settings = Settings()
