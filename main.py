import os
import uuid
import imageio_ffmpeg
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
from faster_whisper import WhisperModel

AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Whisper 모델
# small: 정확도/속도 균형
# medium: 더 정확하지만 많이 느림
model = WhisperModel("tiny", device="cpu", compute_type="int8")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    with open(os.path.join(BASE_DIR, "static", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())

    original_path = os.path.join(UPLOAD_DIR, file_id + "_" + file.filename)
    wav_path = os.path.join(UPLOAD_DIR, file_id + ".wav")
    txt_path = os.path.join(RESULT_DIR, file_id + ".txt")

    # 업로드 파일 저장
    with open(original_path, "wb") as f:
        f.write(await file.read())

    # 음성 파일을 wav로 변환
    audio = AudioSegment.from_file(original_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(wav_path, format="wav")

    # Whisper로 텍스트 변환
    # language=None: 한국어/영어 자동 감지
    segments, info = model.transcribe(
        wav_path,
        language=None,
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True
    )

    result_text = ""

    for segment in segments:
        result_text += f"[{segment.start:.1f}s ~ {segment.end:.1f}s] {segment.text.strip()}\n"

    # txt 파일 저장
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result_text.strip())

    return {
        "text": result_text.strip(),
        "txt_download_url": f"/download/{file_id}"
    }


@app.get("/download/{file_id}")
def download_txt(file_id: str):
    txt_path = os.path.join(RESULT_DIR, file_id + ".txt")

    return FileResponse(
        txt_path,
        media_type="text/plain",
        filename="transcript.txt"
    )
