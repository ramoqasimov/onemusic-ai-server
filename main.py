from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import tempfile
import subprocess
import os
from scipy.io import wavfile

app = FastAPI()


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



def convert_to_wav(path):
    wav_path = path.replace(".mp3", ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", path, wav_path])
    return wav_path



def extract_features(path):
    sr, data = wavfile.read(path)

    # Stereo → mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    # RMS (enerji)
    rms = float(np.sqrt(np.mean(data ** 2)))

    # Zero Crossing Rate
    zcr = float(((data[:-1] * data[1:]) < 0).mean())

    # FFT → spectral centroid
    fft = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1 / sr)
    centroid = float(np.sum(freqs * fft) / np.sum(fft))

    # Bandwidth
    mean_freq = centroid
    bandwidth = float(np.sqrt(np.sum(((freqs - mean_freq) ** 2) * fft) / np.sum(fft)))

    # Rolloff
    cumulative = np.cumsum(fft)
    rolloff_idx = np.where(cumulative >= 0.85 * cumulative[-1])[0][0]
    rolloff = float(freqs[rolloff_idx])

    # Tempo (BPM)
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    peaks = np.diff(np.sign(np.diff(autocorr))) < 0
    peak_indices = np.where(peaks)[0]

    bpm = 0
    if len(peak_indices) > 1:
        peak_diff = peak_indices[1] - peak_indices[0]
        bpm = 60 / (peak_diff / sr)

    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "bpm": float(bpm)
    }



def classify(f):
    rms = f["rms"]
    zcr = f["zcr"]
    centroid = f["centroid"]
    bandwidth = f["bandwidth"]
    rolloff = f["rolloff"]
    bpm = f["bpm"]

    # Trap
    if bpm >= 130 and rms > 0.08 and bandwidth > 2600:
        return "Trap"

    # Rap / Hip-Hop
    if 80 <= bpm <= 110 and zcr > 0.08 and centroid < 2000:
        return "Rap"
    if 85 <= bpm <= 115 and zcr > 0.06 and centroid < 2500:
        return "Hip-Hop"

    # Pop
    if 95 <= bpm <= 130 and rms > 0.05 and centroid > 1800:
        return "Pop"

    # EDM / Dance
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



@app.post("/detect-genre")
async def detect_genre(file: UploadFile = File(...)):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp.write(await file.read())
    temp.close()

    wav_path = convert_to_wav(temp.name)

    f = extract_features(wav_path)
    genre = classify(f)

    sql_format = "~" + genre.lower().replace(" ", "").replace("-", "")

    return {
        "genre": genre,
        "sql_genre": sql_format,
        "features": f
    }



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
