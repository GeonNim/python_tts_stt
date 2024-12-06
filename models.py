from sqlalchemy import Column, Integer, String, Text
from .database import Base

class AudioProcessLog(Base):
    __tablename__ = "audio_process_logs"

    id = Column(Integer, primary_key=True, index=True)
    stt_result = Column(Text, nullable=False)  # STT 결과
    gpt_response = Column(Text, nullable=False)  # GPT 응답
    tts_path = Column(String, nullable=False)  # 생성된 TTS 파일 경로
