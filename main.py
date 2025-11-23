from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import soundfile as sf
import tempfile
import os

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
# 🎛 Audio Feature Extraction (Optimized)
# ================================
def extract_features(path):
    # Fayl haqqında məlumat al (Sample Rate lazımdır)
    info = sf.info(path)
    sr = info.samplerate
    
    # ⚠️ RAM qənaəti üçün yalnız ilk 30 saniyəni oxuyuruq
    # 30 saniyə * sample_rate = oxunacaq freymlər
    max_duration = 30 
    frames_to_read = int(sr * max_duration)
    
    # Əgər mahnı qısadırsa, hamısını, uzundursa yalnız başlanğıcı oxu
    y, sr = sf.read(path, stop=frames_to_read)

    # Stereo (2 kanal) səsdirsə, Mono (1 kanal) edirik (RAM-ı yarıya endirir)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # Feature extraction (Numpy ilə)
    rms = float(np.sqrt(np.mean(y ** 2)))
    
    # ZCR hesablamasında sıfıra bölmə xətasına qarşı qoruma
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
# 🧠 GENRE classifier
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
    # Temp faylı yaradırıq
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        f = extract_features(temp_path)
        genre = classify(f)
    except Exception as e:
        return {"error": str(e)}
    finally:
        # İş bitdikdən sonra faylı mütləq silirik (Disk dolmasın deyə)
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "genre": genre,
        "features": f
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
