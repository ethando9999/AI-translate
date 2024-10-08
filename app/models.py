import os

from torch._inductor.config import trace
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import whoami
from app import settings, iso_language_codes, reversed_iso_language_codes
from faster_whisper import WhisperModel

whoami(token=settings.hf_token)


def save_pretrain_locally(local_dir: str):
    global model, tokenizer

    # Create the local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    model_path = os.path.join(local_dir, "model")
    tokenizer_path = os.path.join(local_dir, "tokenizer")

    # Save model locally if it hasn't been saved already
    if not os.path.exists(model_path):
        print(f"Saving model locally to {model_path}...")
        model.save_pretrained(model_path)
    else:
        print(f"Model already exists locally at {model_path}")

    # Save tokenizer locally if it hasn't been saved already
    if not os.path.exists(tokenizer_path):
        print(f"Saving tokenizer locally to {tokenizer_path}...")
        tokenizer.save_pretrained(tokenizer_path)
    else:
        print(f"Tokenizer already exists locally at {tokenizer_path}")


# Function to translate text
def translate_text(text: str, src_lan: str, tgt_lang: str):
    translation_pipeline = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lan,
        tgt_lang=tgt_lang,
        max_length=settings.max_length,
        device=settings.device,

    )
    results = translation_pipeline(text, max_length=settings.max_length)
    return results[0]['translation_text']


# Load model and tokenizer for translation
model_name = settings.model_name
local_model_path = os.path.join(os.getcwd(), model_name)

try:
    # Try to load the tokenizer and model from the local path
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(local_model_path, 'tokenizer'),)
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(local_model_path, 'model'))
    print(f"Loaded translation model and tokenizer from {local_model_path}")
except Exception as e:
    print(f"Error loading translation model/tokenizer locally: {e}")
    # Save model and tokenizer locally if they don't exist
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    save_pretrain_locally(local_model_path)


# Function to save transcription model locally
def save_transcription_model_locally(local_dir: str):
    global model_transcript

    # Create the local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    model_transcript_path = os.path.join(local_dir, "model_transcript.pt")

    # Save transcription model locally using PyTorch
    if not os.path.exists(model_transcript_path):
        print(f"Saving transcription model locally to {model_transcript_path}...")
        torch.save(model_transcript.state_dict(), model_transcript_path)
    else:
        print(f"Transcription model already exists locally at {model_transcript_path}")


# Load transcription model
model_transcript = WhisperModel(settings.transcript_model_name, device=settings.device, compute_type=settings.transcript_dtype)

print("Load Transcription model: success")
def transcribe_audio(audio_path: str, language: str):
    global model_transcript
    try:
        # Thực hiện quá trình transcription
        segments, info = model_transcript.transcribe(audio_path,
                                                     beam_size=settings.beam_size,
                                                     language=language,
                                                     vad_filter=True,
                                                     vad_parameters=dict(min_silence_duration_ms=int(settings.min_silence_duration_ms)),
                                                     )

        print(f"Language: {info.language}, Duration: {info.duration}")

        # Thu thập toàn bộ văn bản từ các đoạn transcription
        # transcript = ""
        # for segment in segments:
        #     transcript += f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}\n"
        transcript = '<EOS>'.join([segment.text.strip() for segment in segments])

        # print("Transcript:", transcript)
        return transcript

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
