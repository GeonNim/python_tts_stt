from fastapi import FastAPI, File, UploadFile
import uvicorn
import json
import soundfile as sf
from vosk import Model, KaldiRecognizer
import openai
import pyttsx3
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 초기화
app = FastAPI()

# Vosk 모델 로드
model_path = "vosk-model-small-ko-0.22"  # Hugging Face에서 다운로드한 모델 폴더 경로
model = Model(model_path)

# TTS 엔진 초기화
tts_engine = pyttsx3.init()

def speak_text(text, output_file="output.wav"):
    """텍스트를 음성으로 변환하고 파일로 저장"""
    tts_engine.save_to_file(text, output_file)
    tts_engine.runAndWait()
    return output_file

def stt_to_text(audio_path):
    """음성 파일을 텍스트로 변환"""
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

def ask_gpt(question):
    """GPT 모델에 질문하고 응답 반환"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "당신은 Dungeons & Dragons(D&D)의 던전 마스터이자 지식 전문가입니다. "
                    "판타지 설정, D&D 게임 플레이, 전설 및 세계관 구축과 관련된 질문에 대해 상세하고 창의적인 한국어 답변을 제공합니다. "
                    "판타지나 D&D 범위를 벗어나는 질문에 대해서는 정중하게 '죄송하지만, 저는 판타지나 D&D와 관련된 질문에만 답변할 수 있습니다.'라고 응답하세요."
                )},
                {"role": "user", "content": question}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"GPT 응답 실패: {e}")
        return "API 응답에 실패했습니다."

@app.post("/stt-gpt-tts/")
async def process_audio(file: UploadFile = File(...)):
    """STT -> GPT -> TTS 엔드포인트"""
    try:
        # 업로드된 음성 파일 저장
        audio_path = f"temp/{file.filename}"
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        # 음성 -> 텍스트 변환
        user_input = stt_to_text(audio_path)
        if not user_input:
            return {"error": "음성을 텍스트로 변환하지 못했습니다."}

        # GPT 응답 생성
        gpt_response = ask_gpt(user_input)

        # GPT 응답 -> 음성 변환
        output_file = speak_text(gpt_response)

        return {"user_input": user_input, "gpt_response": gpt_response, "audio_file": output_file}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
