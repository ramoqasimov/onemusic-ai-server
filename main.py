from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import soundfile as sf
import tempfile

app = FastAPI()

# ================================
# 🎧 SQL-də olan orijinal kateqoriya adları
# ================================
GENRES = [
    "Azeri Pop", "Turkish Pop", "Pop",
    "Rap", "Hip-Hop", "Trap",
    "R&B", "Soul",
    "EDM", "Deep House", "Dance",
    "Rock", "Alternative Rock", "Metal",
    "Arabesk", "Arabesk Rap",
    "Folk", "Ethno Pop", "Mugham Fusion"
]

# ================================
# 🎛 Audio Feature Extraction
# ================================
def extract_features(path):
    y, sr = sf.read(path)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    rms = float(np.sqrt(np.mean(y ** 2)))
    zcr = float(((y[:-1] * y[1:]) < 0).mean())

    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))

    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum)))

    bpm = float((zcr * 200) + (centroid / 90))

    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "bpm": bpm
    }

# ================================
# 🧠 GENRE classifier (SQL adlarına uyğun)
# ================================
def classify(f):
    rms = f["rms"]
    zcr = f["zcr"]
    centroid = f["centroid"]
    bandwidth = f["bandwidth"]
    bpm = f["bpm"]

    if bpm >= 130 and rms > 0.08 and bandwidth > 2600:
        return "Trap"
    if 80 <= bpm <= 110 and zcr > 0.08 and centroid < 2000:
        return "Rap"
    if 85 <= bpm <= 115 and zcr > 0.06 and centroid < 2500:
        return "Hip-Hop"
    if 95 <= bpm <= 130 and rms > 0.05 and centroid > 1800:
        return "Pop"
    if bpm <= 100 and centroid < 1800 and rms < 0.06:
        return "R&B"
    if rms < 0.05 and centroid < 1500:
        return "Soul"
    if bpm >= 125 and centroid > 3000 and bandwidth > 3500:
        return "EDM"
    if bpm >= 118 and centroid > 2800:
        return "Dance"
    if 110 <= bpm <= 124 and bandwidth > 2000:
        return "Deep House"
    if rms > 0.1 and bandwidth > 4000:
        return "Rock"
    if rms > 0.09 and 3500 < bandwidth < 4000:
        return "Alternative Rock"
    if rms > 0.12 and bandwidth > 5000:
        return "Metal"
    if bpm <= 90 and centroid < 1200:
        return "Folk"
    if bpm <= 95 and 1200 <= centroid <= 2000:
        return "Ethno Pop"
    if centroid < 1500 and 60 <= bpm <= 100:
        return "Arabesk"
    if centroid < 1500 and bpm > 100 and zcr > 0.07:
        return "Arabesk Rap"
    if bpm <= 80 and centroid < 1000:
        return "Mugham Fusion"

    return "Instrumental"

# ================================
# 🚀 API Endpoint
# ================================
@app.post("/detect-genre")
async def detect_genre(file: UploadFile = File(...)):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp.write(await file.read())
    temp.close()

    f = extract_features(temp.name)
    genre = classify(f)

    return {
        "genre": genre,
        "features": f
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
