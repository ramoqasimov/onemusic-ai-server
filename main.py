from fastapi import FastAPI, UploadFile, File
import uvicorn
import librosa
import numpy as np
import tempfile

app = FastAPI()

# ------------------------------------------------------
# 🎧 PROFESSIONAL GENRE LIST (AZ + TR + GLOBAL)
# ------------------------------------------------------
GENRES = [
    "Azeri Pop", "Turkish Pop", "Pop",
    "Rap", "Hip-Hop", "Trap",
    "R&B", "Soul",
    "EDM", "Deep House", "Dance",
    "Rock", "Alternative Rock", "Metal",
    "Arabesk", "Arabesk Rap",
    "Folk", "Ethno Pop", "Mugham Fusion",
    "Lo-Fi", "Acoustic", "Instrumental", "Classical"
]


# ------------------------------------------------------
# 🎛 Extract PRO audio features
# ------------------------------------------------------
def extract_features(path):
    y, sr = librosa.load(path)

    # Enerji
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # ZCR
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Spektral centroid
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Bandwidth
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    # Rolloff
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    # BPM / Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo)

    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "bpm": bpm
    }


# ------------------------------------------------------
# 🧠 PROFESSIONAL GENRE CLASSIFICATION LOGIC
# ------------------------------------------------------
def classify(f):
    rms = f["rms"]
    zcr = f["zcr"]
    centroid = f["centroid"]
    bandwidth = f["bandwidth"]
    rolloff = f["rolloff"]
    bpm = f["bpm"]

    # === REALISTIC AI RULES ===

    # Trap
    if bpm >= 130 and rms > 0.08 and bandwidth > 2600:
        return "Trap"

   
    if 80 <= bpm <= 110 and zcr > 0.08 and centroid < 2000:
        return "Rap"
    if 85 <= bpm <= 115 and zcr > 0.06 and centroid < 2500:
        return "Hip-Hop"

    # Pop (Azeri / Turkish)
    if 95 <= bpm <= 130 and rms > 0.05 and centroid > 1800:
        return "Pop"

    # EDM / Dance / Deep House
    if bpm >= 125 and centroid > 3000 and rolloff > 3500:
        return "EDM"
    if bpm >= 118 and centroid > 2800:
        return "Dance"
    if 110 <= bpm <= 124 and bandwidth > 2000:
        return "Deep House"

    # Rock / Metal
    if rms > 0.1 and bandwidth > 4000:
        return "Rock"
    if rms > 0.12 and bandwidth > 5000:
        return "Metal"

    # Soul / R&B
    if bpm <= 100 and centroid < 1800 and rms < 0.06:
        return "R&B"
    if rms < 0.05 and centroid < 1500:
        return "Soul"

    # Folk / Mugham
    if bpm <= 90 and centroid < 1200:
        return "Folk"
    if bpm <= 80 and centroid < 1000:
        return "Mugham Fusion"

    # Classical
    if rms < 0.03 and centroid < 800:
        return "Classical"

    # Acoustic
    if rms < 0.05 and zcr < 0.04:
        return "Acoustic"

    return "Instrumental"


# ------------------------------------------------------
# 🚀 API Endpoint
# ------------------------------------------------------
@app.post("/detect-genre")
async def detect_genre(file: UploadFile = File(...)):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp.write(await file.read())
    temp.close()

    # Extract features
    f = extract_features(temp.name)

    # Classify
    genre = classify(f)

    # SQL üçün format
    sql_format = "~" + genre.lower().replace(" ", "").replace("-", "")

    return {
        "genre": genre,
        "sql_genre": sql_format,
        "features": f
    }


# ------------------------------------------------------
# SERVER START
# ------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
