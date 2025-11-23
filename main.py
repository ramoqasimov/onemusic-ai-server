from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import tempfile
import os
import librosa

app = FastAPI()

# ================================
# 🎛 PROFESSIONAL AUDIO FEATURE EXTRACTION
# ================================
def extract_features(path):
    # ⚠️ RAM qənaəti: Yalnız ilk 30 saniyə, 22050Hz keyfiyyət, Mono
    try:
        y, sr = librosa.load(path, duration=30, sr=22050, mono=True)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

    # 1. BPM (Ritm sürəti)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    bpm = float(tempo[0])

    # 2. HPSS (Səsi Musiqi və Baraban hissələrinə ayırırıq)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Barabanların gücü (Rap/Trap/EDM üçün vacibdir)
    percussive_energy = np.mean(y_percussive ** 2)
    harmonic_energy = np.mean(y_harmonic ** 2)
    
    # Drum/Musiqi nisbəti
    percussive_ratio = percussive_energy / (harmonic_energy + 1e-6)

    # 3. SPECTRAL CONTRAST (Səsin "dolu" və ya "boş" olması)
    # Elektronik musiqilərdə yüksək, akustiklərdə aşağı olur
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    avg_contrast = np.mean(contrast)

    # 4. ZERO CROSSING RATE (Sərtlik - Rock/Metal üçün)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # 5. MFCC (Səsin Rəngi - Bas mı, incə mi?)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # MFCC[0] -> Səs səviyyəsi (Loudness)
    # MFCC[1] -> Bas və Orta səslər balansı (Mənfi olanda parlaq, müsbət olanda boğuq/baslı)
    bass_feature = mfcc_mean[1] 

    return {
        "bpm": bpm,
        "percussive_ratio": float(percussive_ratio),
        "contrast": float(avg_contrast),
        "zcr": float(zcr),
        "bass_feature": float(bass_feature) # Aşağı dəyər = İncə səs, Yuxarı dəyər = Bas
    }

# ================================
# 🧠 SCORING SYSTEM (Xal Sistemi)
# ================================
def classify_genre(f):
    if f is None:
        return "Unknown"

    bpm = f["bpm"]
    perc_ratio = f["percussive_ratio"] # Ritm gücü
    contrast = f["contrast"]           # Elektroniklik
    zcr = f["zcr"]                     # Aqressivlik (Metal/Rock)
    bass = f["bass_feature"]           # Səs rəngi (Yüksək = Baslı)

    scores = {}

    # --- 1. TRAP ---
    # Xüsusiyyətlər: Yüksək BPM (130+), Çox güclü ritm (percussive), Tünd səs
    scores["Trap"] = 0
    if bpm > 130: scores["Trap"] += 2
    if perc_ratio > 1.5: scores["Trap"] += 3
    if bass > 20: scores["Trap"] += 2

    # --- 2. HIP-HOP / RAP ---
    # Xüsusiyyətlər: Orta BPM (80-110), Güclü ritm
    scores["Hip-Hop"] = 0
    if 80 <= bpm <= 115: scores["Hip-Hop"] += 2
    if perc_ratio > 1.2: scores["Hip-Hop"] += 2
    if contrast > 20: scores["Hip-Hop"] += 1

    # --- 3. METAL ---
    # Xüsusiyyətlər: Çox yüksək ZCR (cızıltı), Aqressiv
    scores["Metal"] = 0
    if zcr > 0.08: scores["Metal"] += 5 # Ən vacib göstərici
    if bpm > 120: scores["Metal"] += 1
    if perc_ratio > 1.0: scores["Metal"] += 1

    # --- 4. ROCK ---
    # Xüsusiyyətlər: Yüksək ZCR (amma Metal qədər yox), Canlı alətlər
    scores["Rock"] = 0
    if 0.04 < zcr <= 0.08: scores["Rock"] += 3
    if bpm > 90: scores["Rock"] += 1
    if contrast < 22: scores["Rock"] += 1 # Daha akustik

    # --- 5. EDM / HOUSE ---
    # Xüsusiyyətlər: Sabit BPM (120-130), Yüksək kontrast (Elektronik)
    scores["EDM"] = 0
    if 118 <= bpm <= 132: scores["EDM"] += 3
    if contrast > 23: scores["EDM"] += 2
    if perc_ratio > 1.0: scores["EDM"] += 1

    # --- 6. POP ---
    # Xüsusiyyətlər: Balanslı, Orta kontrast, Çox sərt deyil
    scores["Pop"] = 0
    if 90 <= bpm <= 125: scores["Pop"] += 2
    if 0.02 < zcr < 0.06: scores["Pop"] += 1
    if perc_ratio < 1.5: scores["Pop"] += 1 # Ritm vokalın qarşısını kəsmir

    # --- 7. R&B / SOUL ---
    # Xüsusiyyətlər: Yavaş BPM, Yumşaq ritm, Aşağı ZCR
    scores["R&B"] = 0
    if bpm < 100: scores["R&B"] += 2
    if zcr < 0.03: scores["R&B"] += 2
    if 0.5 < perc_ratio < 1.2: scores["R&B"] += 1

    # --- 8. CLASSICAL / AMBIENT ---
    # Xüsusiyyətlər: Çox az ritm, Çox aşağı ZCR
    scores["Classical"] = 0
    if perc_ratio < 0.2: scores["Classical"] += 4
    if bpm < 80: scores["Classical"] += 1

    # --- GALİBİ SEÇİRİK ---
    # Xalları sıralayırıq
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_genre, best_score = sorted_scores[0]

    # Konsola yazırıq ki, nəticəni görəsən
    print(f"📊 ANALIZ: BPM={bpm:.0f} | DrumRatio={perc_ratio:.2f} | ZCR={zcr:.3f}")
    print(f"🏆 Qalib: {best_genre} (Xal: {best_score})")

    # Əgər heç bir xal toplaya bilməyibsə (çox qəribə səsdirsə)
    if best_score == 0:
        return "Alternative"

    return best_genre

# ================================
# 🚀 API Endpoint
# ================================
@app.post("/detect-genre")
async def detect_genre(file: UploadFile = File(...)):
    # Temp fayl yaradırıq
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        f = extract_features(temp_path)
        genre = classify_genre(f)
    except Exception as e:
        print(f"Server Error: {e}")
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
