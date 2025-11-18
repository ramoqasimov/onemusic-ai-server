from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import soundfile as sf
import tempfile

app = FastAPI()

# ------------------------------------------------------
# 🎧 PROFESIONAL GENRE LIST (AZ + TR + GLOBAL)
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
# 🎛 LIBROSA OLMADAN PRO AUDIO FEATURES
# ------------------------------------------------------
def extract_features(path):
    y, sr = sf.read(path)

    # Stereo → mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # ======================
    # RMS (enerji)
    # ======================
    rms = float(np.sqrt(np.mean(y**2)))

    # ======================
    # Zero Crossing Rate
    # ======================
    zcr = float(((y[:-1] * y[1:]) < 0).mean())

    # ======================
    # Spectral Centroid
    # ======================
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))

    # ======================
    # Bandwidth
    # ======================
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid)**2) * spectrum) / np.sum(spectrum)))

    # ======================
    # Simple BPM estimate
    # ======================
    bpm = float((zcr * 200) + (centroid / 90))

    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "bpm": bpm
    }


# ------------------------------------------------------
# 🧠 PROFESSIONAL CLASSIFICATION (Sənin QAYDALARIN)
# ------------------------------------------------------
def classify(features):
    rms = features["rms"]
    zcr = features["zcr"]
    centroid = features["centroid"]
    bandwidth = features["bandwidth"]
    bpm = features["bpm"]

    # === TRAP ===
    if bpm >= 130 and rms > 0.08 and bandwidth > 2600:
        return "Trap"

    # === RAP ===
    if 80 <= bpm <= 110 and zcr > 0.08 and centroid < 2000:
        return "Rap"

    # === HIP-HOP ===
    if 85 <= bpm <= 115 and zcr > 0.06 and centroid < 2500:
        return "Hip-Hop"

    # === POP ===
    if 95 <= bpm <= 130 and rms > 0.05 and centroid > 1800:
        return "Pop"

    # === EDM ===
    if bpm >= 125 and centroid > 3000 and bandwidth > 3500:
        return "EDM"

    # === DANCE ===
    if bpm >= 118 and centroid > 2800:
        return "Dance"

    # === DEEP HOUSE ===
    if 110 <= bpm <= 124 and bandwidth > 2000:
        return "Deep House"

    # === ROCK ===
    if rms > 0.1 and bandwidth > 4000:
        return "Rock"

    # === METAL ===
    if rms > 0.12 and bandwidth > 5000:
        return "Metal"

    # === R&B ===
    if bpm <= 100 and centroid < 1800 and rms < 0.06:
        return "R&B"

    # === SOUL ===
    if rms < 0.05 and centroid < 1500:
        return "Soul"

    # === FOLK ===
    if bpm <= 90 and centroid < 1200:
        return "Folk"

    # === MUGHAM FUSION ===
    if bpm <= 80 and centroid < 1000:
        return "Mugham Fusion"

    # === CLASSICAL ===
    if rms < 0.03 and centroid < 800:
        return "Classical"

    # === ACOUSTIC ===
    if rms < 0.05 and zcr < 0.04:
        return "Acoustic"

    return "Instrumental"


# ------------------------------------------------------
# 🚀 API Endpoint
# ------------------------------------------------------
@app.post("/detect-genre")
async def detect_genre(file: UploadFile = File(...)):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp.write(await file.read())
    temp.close()

    features = extract_features(temp.name)
    genre = classify(features)

    # SQL üçün format: ~pop , ~rap , ~hiphop
    sql_genre = "~" + genre.lower().replace(" ", "").replace("-", "")

    return {
        "genre": genre,
        "sql_genre": sql_genre,
        "features": features
    }


# ------------------------------------------------------
# SERVER START (Render)
# ------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
