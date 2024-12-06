from fastapi import FastAPI, File, UploadFile, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import get_db, engine
from models import Base, AudioProcessLog
from celery_config import celery_app
import os
import json
import soundfile as sf
from vosk import Model, KaldiRecognizer
import openai
import pyttsx3
from dotenv import load_dotenv
import uuid
import logging
from scipy.io.wavfile import write
import librosa
import aiofiles
from prometheus_client import Counter, generate_latest
from celery.result import AsyncResult
import pandas as pd
import httpx

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 초기화
app = FastAPI()

# Prometheus 메트릭
REQUEST_COUNT = Counter("request_count", "Total number of requests received")

# Vosk 모델 로드
model_path = "vosk-model-small-ko-0.22"
model = Model(model_path)

# TTS 엔진 초기화
tts_engine = pyttsx3.init()

# 로깅 설정
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_error(message):
    logging.error(message)

@app.on_event("startup")
async def startup():
    """DB 및 모델 초기화"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def resample_audio(input_path, target_path, target_sr=16000):
    """오디오 샘플링 비율 변경"""
    y, sr = librosa.load(input_path, sr=None)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    write(target_path, target_sr, (y_resampled * 32767).astype('int16'))
    return target_path

def speak_text(text, output_file=None):
    """텍스트를 음성으로 변환하여 파일 저장"""
    if not output_file:
        output_file = f"temp/{uuid.uuid4()}.wav"
    tts_engine.save_to_file(text, output_file)
    tts_engine.runAndWait()
    return output_file

def stt_to_text(audio_path):
    """STT: 음성 파일을 텍스트로 변환"""
    with sf.SoundFile(audio_path) as audio:
        if audio.samplerate != 16000:
            raise ValueError("Audio sample rate must be 16kHz.")
        recognizer = KaldiRecognizer(model, audio.samplerate)
        while True:
            data = audio.read(4000, dtype="int16").tobytes()
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)
        result = json.loads(recognizer.FinalResult())
        return result.get("text", "")

@app.get("/metrics")
def metrics():
    """Prometheus 메트릭 엔드포인트"""
    return generate_latest()

@app.post("/stt-gpt-tts/")
async def process_audio(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """STT → GPT → TTS 프로세스"""
    REQUEST_COUNT.inc()
    audio_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    async with aiofiles.open(audio_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    resampled_audio_path = f"{audio_path}_resampled.wav"
    resample_audio(audio_path, resampled_audio_path)

    user_input = stt_to_text(resampled_audio_path)
    gpt_response = await ask_gpt(user_input)
    output_file = speak_text(gpt_response)

    new_log = AudioProcessLog(
        stt_result=user_input, gpt_response=gpt_response, tts_path=output_file
    )
    db.add(new_log)
    await db.commit()

    return {"user_input": user_input, "gpt_response": gpt_response, "audio_file": output_file}

@app.post("/stt-gpt-tts-async/")
async def process_audio_async(file: UploadFile = File(...)):
    """비동기 Celery 작업"""
    audio_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    async with aiofiles.open(audio_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    task = process_audio_task.delay(audio_path)
    return {"task_id": task.id, "status": "Processing"}

@celery_app.task
def process_audio_task(audio_path):
    """Celery 작업: STT → GPT → TTS"""
    resampled_audio_path = f"{audio_path}_resampled.wav"
    resample_audio(audio_path, resampled_audio_path)

    user_input = stt_to_text(resampled_audio_path)
    gpt_response = ask_gpt_sync(user_input)
    output_file = speak_text(gpt_response)

    return {"user_input": user_input, "gpt_response": gpt_response, "audio_file": output_file}

@app.get("/logs/")
async def get_logs(limit: int = 10, db: AsyncSession = Depends(get_db)):
    """로그 가져오기"""
    query = await db.execute(select(AudioProcessLog).order_by(AudioProcessLog.id.desc()).limit(limit))
    logs = query.scalars().all()
    return [{"id": log.id, "stt_result": log.stt_result, "gpt_response": log.gpt_response, "tts_path": log.tts_path} for log in logs]

async def ask_gpt(question):
    """GPT API 호출 (비동기)"""
    system_message = (
        "당신은 Dungeons & Dragons(D&D)의 던전 마스터이자 지식 전문가입니다. "
        "판타지 설정, D&D 게임 플레이, 전설 및 세계관 구축과 관련된 질문에 대해 상세하고 창의적인 한국어 답변을 제공합니다. "
        "판타지나 D&D 범위를 벗어나는 질문에 대해서는 '죄송하지만, 판타지나 D&D와 관련된 질문에만 답변할 수 있습니다.'라고 응답하세요."
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai.api_key}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ]
            }
        )
    return response.json()["choices"][0]["message"]["content"]

def ask_gpt_sync(question):
    """GPT API 호출 (동기)"""
    system_message = (
        "당신은 Dungeons & Dragons(D&D)의 던전 마스터이자 지식 전문가입니다. "
        "판타지 설정, D&D 게임 플레이, 전설 및 세계관 구축과 관련된 질문에 대해 상세하고 창의적인 한국어 답변을 제공합니다. "
        "판타지나 D&D 범위를 벗어나는 질문에 대해서는 '죄송하지만, 판타지나 D&D와 관련된 질문에만 답변할 수 있습니다.'라고 응답하세요."
    )
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {openai.api_key}"},
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ]
        }
    )
    return response.json()["choices"][0]["message"]["content"]
