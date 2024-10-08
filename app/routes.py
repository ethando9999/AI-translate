from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from io import BytesIO
from app import translate_text, transcribe_audio, settings
import os

MAX_FILE_SIZE=settings.file_audio_size

# Tạo router
router = APIRouter()


#API route translate
@router.post("/translate/")
async def translate(input_text: str, src_lan:str, tgt_lang:str):
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is required")

    # Gọi hàm translate_text để dịch
    translated_text = translate_text(input_text, src_lan=src_lan, tgt_lang=tgt_lang)

    return {"translated_text": translated_text}


@router.post("/transcribe/")
async def transcribe(lan: str, file: UploadFile = File(...)):
    # Kiểm tra định dạng file
    # if file.content_type not in ["audio/wav", "audio/mpeg", "audio/mp3"]:
    #     raise HTTPException(status_code=400, detail="Invalid audio format. Only .wav, .mp3 allowed.")

    # Kiểm tra kích thước file
    content = await file.read()

    print("Size:", len(content))

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE / (1024 ** 2)} MB limit.")

    # Đọc nội dung file vào buffer
    audio_buffer = BytesIO(content)

    # Lưu file âm thanh tạm thời
    # audio_file_path = f"temp_{file.filename}"
    # with open(audio_file_path, "wb") as audio_file:
    #     content = await file.read()
    #     audio_file.write(content)

    # Gọi hàm để transcribe va xoa temp
    # transcription_result = transcribe_audio(audio_file_path, lan)
    # os.remove(audio_file_path)

    transcription_result = transcribe_audio(audio_buffer, lan)
    print(transcription_result)

    if transcription_result:
        return {"transcription": transcription_result}
    else:
        raise HTTPException(status_code=500, detail="Transcription failed.")