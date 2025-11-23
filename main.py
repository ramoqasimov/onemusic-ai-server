from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import soundfile as sf
import tempfile
import os
from scipy.signal import find_peaks

app = FastAPI()

# ================================
# 🎛 AĞILLI AUDIO ANALİZ (Librosa-sız)
# ================================
def extract_features(path):
    # 1. Faylı oxu (Yalnız ilk 30 saniyə - RAM qənaəti)
    info = sf.info(path)
    sr = info.samplerate
    max_duration = 30 
    frames_to_read = int(sr * max_duration)
    
    y, sr = sf.read(path, stop=frames_to_read)

    # Stereo -> Mono çeviririk
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # Normalizasiya (Səsi standartlaşdırırıq)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # --- FEATURE 1: RMS (Enerji / Səs gücü) ---
    rms = float(np.sqrt(np.mean(y ** 2)))

    # --- FEATURE 2: ZCR (Sərtlik / Metalik səs) ---
    zcr = 0.0
    if len(y) > 1:
        zcr = float(((y[:-1] * y[1:]) < 0).mean())

    # --- FEATURE 3: FFT (Tezlik Analizi) ---
    # Səsi tezliklərə bölürük: Bass, Orta, Yüksək
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    
    # Enerjilərin cəmi
    total_energy = np.sum(spectrum)
    
    if total_energy > 0:
        # Bass (0 - 250 Hz)
        bass_idx = np.where((freqs >= 0) & (freqs <= 250))[0]
        bass_energy = np.sum(spectrum[bass_idx]) / total_energy

        # Mid (250 - 4000 Hz) - Vokal və alətlər
        mid_idx = np.where((freqs > 250) & (freqs <= 4000))[0]
        mid_energy = np.sum(spectrum[mid_idx]) / total_energy

        # High (4000+ Hz) - Zərb alətləri, cızıltı
        high_idx = np.where(freqs > 4000)[0]
        high_energy = np.sum(spectrum[high_idx]) / total_energy
    else:
        bass_energy = mid_energy = high_energy = 0

    # --- FEATURE 4: REAL BPM TƏXMİNİ (Signal Peaks) ---
    # Sadə amplituda piklərini tapırıq (Ritm)
    # Səsi biraz hamarlayırıq ki, pikləri tapaq
    window_size = int(sr * 0.2) # 0.2 saniyəlik pəncərə
    amplitude_envelope = np.convolve(np.abs(y), np.ones(window_size)/window_size, mode='same')
    
    # Piklər (Beat-lər)
    peaks, _ = find_peaks(amplitude_envelope, height=0.1, distance=sr/3) # Max 180 BPM (sr/3)
    
    # BPM hesabla
    if len(peaks) > 1:
        beat_times = peaks / sr
        intervals = np.diff(beat_times)
        avg_interval = np.mean(intervals)
        if avg_interval > 0:
            bpm = 60 / avg_interval
    else:
        bpm = 0

    # Çox aşağı və ya çox yuxarı BPM-i düzəldirik (Octave error correction)
    if bpm < 60 and bpm > 0: bpm *= 2
    if bpm > 180: bpm /= 2

    return {
        "rms": rms,
        "zcr": zcr,
        "bass": bass_energy,
        "mid": mid_energy,
        "high": high_energy,
        "bpm": bpm
    }

# ================================
# 🧠 JANR MƏNTİQİ (Daha Dəqiq)
# ================================
def classify(f):
    rms = f["rms"]   # Səs səviyyəsi (0.0 - 0.5)
    zcr = f["zcr"]   # Cızıltı/Sərtlik (Rock/Metal üçün yüksək olur)
    bass = f["bass"] # Bas səslər (Rap/Trap/EDM)
    high = f["high"] # İncə səslər
    bpm = f["bpm"]   # Sürət

    print(f"ANALYZED: BPM={bpm:.1f}, Bass={bass:.2f}, High={high:.2f}, ZCR={zcr:.3f}, RMS={rms:.3f}")

    # 1. METAL & ROCK (Yüksək enerji, çoxlu incə səs/cızıltı)
    if zcr > 0.08 and rms > 0.15:
        if bpm > 130: return "Metal"
        return "Rock"
    
    if zcr > 0.06 and mid_dominance(f):
        return "Alternative Rock"

    # 2. ELEKTRONİK & DANCE (Güclü Bass, Sabit BPM)
    if bass > 0.35 and rms > 0.1:
        if bpm > 135: return "Drum & Bass"
        if bpm > 126: return "Techno"
        if bpm > 115: return "EDM"
        if bpm > 105: return "House"
    
    # 3. RAP & HIP-HOP (Çox Güclü Bass, orta sürət)
    if bass > 0.40: # Bass çox güclüdürsə
        if bpm > 120: return "Trap"
        if bpm > 80: return "Rap"
        return "Hip-Hop"

    # 4. POP (Balanslı səs, orta bass, aydın səs)
    if rms > 0.08 and 0.2 < bass < 0.4:
        if bpm > 110: return "Pop"
        if bpm > 90: return "Modern Pop"

    # 5. SAKİT MAHNILAR
    if rms < 0.05: # Səs çox zəifdir
        if bpm < 80: return "Ambient"
        return "Classical"

    if bpm < 90:
        if bass > 0.3: return "R&B"
        if zcr < 0.03: return "Soul"
        return "Slow Pop"

    if 90 <= bpm <= 110:
        return "Indie Pop"

    return "General"

# Yardımçı funksiya: Orta səslər üstündürmü? (Gitara/Vokal)
def mid_dominance(f):
    return f["mid"] > f["bass"] and f["mid"] > f["high"]

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
        print("Error:", e)
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
