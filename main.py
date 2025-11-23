from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import soundfile as sf
import tempfile
import os

app = FastAPI()

# ================================
# 🎛 Audio Feature Extraction
# ================================
def extract_features(path):
    info = sf.info(path)
    sr = info.samplerate
    
    # 30 saniyə oxuyuruq (RAM qənaəti)
    max_duration = 30 
    frames_to_read = int(sr * max_duration)
    
    y, sr = sf.read(path, stop=frames_to_read)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # Feature extraction
    rms = float(np.sqrt(np.mean(y ** 2)))
    
    if len(y) > 1:
        zcr = float(((y[:-1] * y[1:]) < 0).mean())
    else:
        zcr = 0.0

    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    
    sum_spectrum = np.sum(spectrum)
    
    if sum_spectrum > 0:
        centroid = float(np.sum(freqs * spectrum) / sum_spectrum)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / sum_spectrum))
    else:
        centroid = 0.0
        bandwidth = 0.0

    bpm = float((zcr * 200) + (centroid / 90))

    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "bpm": bpm
    }

# ================================
# 🧠 DYNAMIC GENRE CLASSIFIER
# ================================
def classify(f):
    rms = f["rms"]
    zcr = f["zcr"]
    centroid = f["centroid"]
    bandwidth = f["bandwidth"]
    bpm = f["bpm"]

    # Bu məntiq riyazi olaraq səs dalğalarına əsasən ən yaxın janrı tapır.
    # 512MB RAM limitində işləyəcək ən optimal üsul budur.

    if bpm > 135 and bandwidth > 2800:
        if rms > 0.15: return "Metal"
        return "Drum & Bass"
    
    if bpm > 120:
        if bandwidth > 3000: return "EDM"
        if zcr > 0.08: return "Techno"
        if centroid > 2500: return "Pop"
        return "House"

    if 110 <= bpm <= 120:
        if bandwidth < 2200: return "Deep House"
        return "Dance"

    if 90 <= bpm < 110:
        if zcr > 0.07: return "Hip-Hop"
        if rms > 0.1: return "Rock"
        if centroid < 2000: return "Rap"
        return "Alternative"

    if 70 <= bpm < 90:
        if centroid < 1500: return "R&B"
        if rms > 0.08: return "Blues"
        return "Soul"

    if bpm < 70:
        if centroid < 1000: return "Ambient"
        if bandwidth < 1500: return "Lo-Fi"
        return "Slow Pop"

    # Heç birinə uyğun gəlməzsə, ümumi bir ad veririk
    return "Experimental"

# ================================
# 🚀 API Endpoint
# ================================
@app.post("/detect-genre")
async def detect_genre(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        f = extract_features(temp_path)
        genre = classify(f)
    except Exception as e:
        return {"error": str(e), "genre": "Unknown"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "genre": genre,
        "features": f
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
