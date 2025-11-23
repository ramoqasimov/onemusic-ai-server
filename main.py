from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import tempfile
import os
import librosa
from scipy.spatial import distance

app = FastAPI()

# ================================
# 🛠 YARDIMÇI: SANSÜR FUNKSİYASI (Fix Infinity/NaN)
# ================================
def safe_float(value):
    """
    Bu funksiya Infinity və ya NaN dəyərlərini 0.0-a çevirir.
    C# tərəfində JSON xətası olmaması üçün vacibdir.
    """
    if value is None:
        return 0.0
    if np.isinf(value) or np.isnan(value):
        return 0.0
    return float(value)

# ================================
# 📊 MEGA GENRE LISTE (Sənin PDF-dən)
# ================================
GENRE_PROFILES = {
    # --- 1. ROCK ---
    "Classic Rock":      {"bpm": 120, "zcr": 0.07, "bass": 0.30, "contrast": 22},
    "Hard Rock":         {"bpm": 130, "zcr": 0.10, "bass": 0.35, "contrast": 21},
    "Soft Rock":         {"bpm": 100, "zcr": 0.04, "bass": 0.25, "contrast": 20},
    "Pop Rock":          {"bpm": 115, "zcr": 0.06, "bass": 0.30, "contrast": 23},
    "Punk Rock":         {"bpm": 160, "zcr": 0.12, "bass": 0.25, "contrast": 20},
    "Psychedelic Rock":  {"bpm": 110, "zcr": 0.08, "bass": 0.30, "contrast": 19},
    "Indie Rock":        {"bpm": 118, "zcr": 0.06, "bass": 0.28, "contrast": 24},
    "Alternative Rock":  {"bpm": 125, "zcr": 0.09, "bass": 0.32, "contrast": 22},
    "Garage Rock":       {"bpm": 135, "zcr": 0.11, "bass": 0.30, "contrast": 18},
    "Glam Rock":         {"bpm": 128, "zcr": 0.07, "bass": 0.30, "contrast": 21},
    "Progressive Rock":  {"bpm": 130, "zcr": 0.08, "bass": 0.35, "contrast": 25},
    "Post-Rock":         {"bpm": 100, "zcr": 0.05, "bass": 0.40, "contrast": 26},
    "Math Rock":         {"bpm": 140, "zcr": 0.06, "bass": 0.25, "contrast": 27},
    "Noise Rock":        {"bpm": 150, "zcr": 0.15, "bass": 0.35, "contrast": 15},
    "Surf Rock":         {"bpm": 160, "zcr": 0.05, "bass": 0.25, "contrast": 22},
    "Southern Rock":     {"bpm": 110, "zcr": 0.06, "bass": 0.30, "contrast": 20},
    "Blues Rock":        {"bpm": 105, "zcr": 0.05, "bass": 0.30, "contrast": 20},
    "Folk Rock":         {"bpm": 95,  "zcr": 0.03, "bass": 0.20, "contrast": 19},
    "Krautrock":         {"bpm": 120, "zcr": 0.06, "bass": 0.35, "contrast": 23},
    "Stoner Rock":       {"bpm": 85,  "zcr": 0.10, "bass": 0.45, "contrast": 18},
    "Gothic Rock":       {"bpm": 115, "zcr": 0.07, "bass": 0.35, "contrast": 21},
    "Arena Rock":        {"bpm": 125, "zcr": 0.06, "bass": 0.35, "contrast": 22},
    "Space Rock":        {"bpm": 110, "zcr": 0.07, "bass": 0.40, "contrast": 24},

    # --- 2. METAL ---
    "Heavy Metal":       {"bpm": 140, "zcr": 0.12, "bass": 0.40, "contrast": 19},
    "Thrash Metal":      {"bpm": 180, "zcr": 0.16, "bass": 0.35, "contrast": 18},
    "Death Metal":       {"bpm": 160, "zcr": 0.18, "bass": 0.45, "contrast": 17},
    "Black Metal":       {"bpm": 170, "zcr": 0.20, "bass": 0.25, "contrast": 15},
    "Power Metal":       {"bpm": 160, "zcr": 0.10, "bass": 0.35, "contrast": 22},
    "Speed Metal":       {"bpm": 190, "zcr": 0.14, "bass": 0.30, "contrast": 20},
    "Symphonic Metal":   {"bpm": 135, "zcr": 0.09, "bass": 0.40, "contrast": 24},
    "Progressive Metal": {"bpm": 130, "zcr": 0.10, "bass": 0.35, "contrast": 26},
    "Doom Metal":        {"bpm": 60,  "zcr": 0.11, "bass": 0.50, "contrast": 18},
    "Gothic Metal":      {"bpm": 100, "zcr": 0.08, "bass": 0.40, "contrast": 21},
    "Industrial Metal":  {"bpm": 130, "zcr": 0.14, "bass": 0.50, "contrast": 25},
    "Melodic Death":     {"bpm": 150, "zcr": 0.13, "bass": 0.40, "contrast": 22},
    "Folk Metal":        {"bpm": 130, "zcr": 0.09, "bass": 0.30, "contrast": 20},
    "Viking Metal":      {"bpm": 140, "zcr": 0.11, "bass": 0.35, "contrast": 20},
    "Nu Metal":          {"bpm": 110, "zcr": 0.12, "bass": 0.55, "contrast": 20},
    "Metalcore":         {"bpm": 145, "zcr": 0.15, "bass": 0.45, "contrast": 21},
    "Deathcore":         {"bpm": 150, "zcr": 0.18, "bass": 0.60, "contrast": 18},
    "Grindcore":         {"bpm": 200, "zcr": 0.22, "bass": 0.30, "contrast": 14},
    "Groove Metal":      {"bpm": 120, "zcr": 0.11, "bass": 0.50, "contrast": 20},
    "Drone Metal":       {"bpm": 40,  "zcr": 0.10, "bass": 0.60, "contrast": 16},

    # --- 3. POP ---
    "Pop":               {"bpm": 118, "zcr": 0.04, "bass": 0.30, "contrast": 25},
    "Dance Pop":         {"bpm": 126, "zcr": 0.05, "bass": 0.45, "contrast": 26},
    "Electropop":        {"bpm": 128, "zcr": 0.06, "bass": 0.50, "contrast": 27},
    "Teens Pop":         {"bpm": 120, "zcr": 0.04, "bass": 0.30, "contrast": 25},
    "Indie Pop":         {"bpm": 105, "zcr": 0.04, "bass": 0.25, "contrast": 24},
    "K-Pop":             {"bpm": 128, "zcr": 0.05, "bass": 0.40, "contrast": 26},
    "J-Pop":             {"bpm": 135, "zcr": 0.06, "bass": 0.35, "contrast": 26},
    "Synth-Pop":         {"bpm": 115, "zcr": 0.04, "bass": 0.40, "contrast": 27},
    "Dream Pop":         {"bpm": 90,  "zcr": 0.02, "bass": 0.25, "contrast": 22},
    "Bubblegum Pop":     {"bpm": 130, "zcr": 0.03, "bass": 0.30, "contrast": 24},
    "Art Pop":           {"bpm": 100, "zcr": 0.05, "bass": 0.30, "contrast": 26},
    "Baroque Pop":       {"bpm": 110, "zcr": 0.03, "bass": 0.20, "contrast": 23},
    "Trip-Pop":          {"bpm": 85,  "zcr": 0.04, "bass": 0.40, "contrast": 24},

    # --- 4. EDM ---
    "House":             {"bpm": 124, "zcr": 0.05, "bass": 0.55, "contrast": 26},
    "Deep House":        {"bpm": 122, "zcr": 0.03, "bass": 0.60, "contrast": 25},
    "Future House":      {"bpm": 126, "zcr": 0.06, "bass": 0.65, "contrast": 27},
    "Progressive House": {"bpm": 128, "zcr": 0.05, "bass": 0.50, "contrast": 26},
    "Tropical House":    {"bpm": 110, "zcr": 0.03, "bass": 0.40, "contrast": 24},
    "Electro House":     {"bpm": 128, "zcr": 0.08, "bass": 0.60, "contrast": 28},
    "Tech House":        {"bpm": 125, "zcr": 0.05, "bass": 0.55, "contrast": 26},
    "Slap House":        {"bpm": 124, "zcr": 0.05, "bass": 0.70, "contrast": 27},
    "Bass House":        {"bpm": 128, "zcr": 0.09, "bass": 0.75, "contrast": 28},
    "Big Room":          {"bpm": 128, "zcr": 0.07, "bass": 0.65, "contrast": 26},
    "Trance":            {"bpm": 138, "zcr": 0.06, "bass": 0.50, "contrast": 27},
    "Psytrance":         {"bpm": 145, "zcr": 0.08, "bass": 0.55, "contrast": 28},
    "Goa Trance":        {"bpm": 142, "zcr": 0.07, "bass": 0.50, "contrast": 27},
    "Techno":            {"bpm": 130, "zcr": 0.06, "bass": 0.55, "contrast": 27},
    "Detroit Techno":    {"bpm": 135, "zcr": 0.05, "bass": 0.50, "contrast": 25},
    "Minimal Techno":    {"bpm": 126, "zcr": 0.04, "bass": 0.45, "contrast": 24},
    "Acid Techno":       {"bpm": 135, "zcr": 0.10, "bass": 0.50, "contrast": 28},
    "Dubstep":           {"bpm": 140, "zcr": 0.15, "bass": 0.75, "contrast": 29},
    "Brostep":           {"bpm": 145, "zcr": 0.18, "bass": 0.70, "contrast": 28},
    "Chillstep":         {"bpm": 140, "zcr": 0.06, "bass": 0.50, "contrast": 23},
    "Riddim":            {"bpm": 140, "zcr": 0.12, "bass": 0.70, "contrast": 27},
    "Drum & Bass":       {"bpm": 174, "zcr": 0.08, "bass": 0.55, "contrast": 25},
    "Jungle":            {"bpm": 170, "zcr": 0.10, "bass": 0.50, "contrast": 24},
    "Liquid DnB":        {"bpm": 174, "zcr": 0.06, "bass": 0.45, "contrast": 24},
    "Breakbeat":         {"bpm": 130, "zcr": 0.08, "bass": 0.55, "contrast": 25},
    "IDM":               {"bpm": 140, "zcr": 0.12, "bass": 0.35, "contrast": 26},
    "Ambient":           {"bpm": 60,  "zcr": 0.01, "bass": 0.20, "contrast": 15},
    "Chillout":          {"bpm": 80,  "zcr": 0.02, "bass": 0.30, "contrast": 20},
    "Lo-Fi":             {"bpm": 80,  "zcr": 0.03, "bass": 0.35, "contrast": 18},
    "Eurodance":         {"bpm": 140, "zcr": 0.05, "bass": 0.50, "contrast": 26},
    "Hardstyle":         {"bpm": 150, "zcr": 0.12, "bass": 0.80, "contrast": 28},
    "Hardcore":          {"bpm": 170, "zcr": 0.15, "bass": 0.75, "contrast": 27},
    "Gabber":            {"bpm": 180, "zcr": 0.16, "bass": 0.80, "contrast": 26},
    "Trap EDM":          {"bpm": 145, "zcr": 0.09, "bass": 0.70, "contrast": 27},
    "Future Bass":       {"bpm": 150, "zcr": 0.08, "bass": 0.65, "contrast": 26},

    # --- 5. CLASSICAL ---
    "Classical":         {"bpm": 70,  "zcr": 0.02, "bass": 0.15, "contrast": 18},
    "Baroque":           {"bpm": 90,  "zcr": 0.03, "bass": 0.10, "contrast": 18},
    "Romantic":          {"bpm": 60,  "zcr": 0.02, "bass": 0.20, "contrast": 19},
    "Opera":             {"bpm": 80,  "zcr": 0.04, "bass": 0.20, "contrast": 20},
    "Symphony":          {"bpm": 75,  "zcr": 0.03, "bass": 0.25, "contrast": 22},
    "Minimalism":        {"bpm": 110, "zcr": 0.02, "bass": 0.15, "contrast": 17},
    "Neoclassical":      {"bpm": 85,  "zcr": 0.02, "bass": 0.15, "contrast": 19},

    # --- 6. HIP-HOP ---
    "East Coast":        {"bpm": 92,  "zcr": 0.06, "bass": 0.50, "contrast": 22},
    "West Coast":        {"bpm": 95,  "zcr": 0.05, "bass": 0.55, "contrast": 23},
    "Trap":              {"bpm": 140, "zcr": 0.07, "bass": 0.65, "contrast": 24},
    "Drill":             {"bpm": 142, "zcr": 0.07, "bass": 0.70, "contrast": 23},
    "Boom Bap":          {"bpm": 90,  "zcr": 0.06, "bass": 0.50, "contrast": 21},
    "Gangsta Rap":       {"bpm": 94,  "zcr": 0.07, "bass": 0.60, "contrast": 22},
    "Cloud Rap":         {"bpm": 130, "zcr": 0.04, "bass": 0.45, "contrast": 19},
    "Mumble Rap":        {"bpm": 135, "zcr": 0.05, "bass": 0.55, "contrast": 21},
    "Emo Rap":           {"bpm": 120, "zcr": 0.05, "bass": 0.40, "contrast": 22},
    "Horrorcore":        {"bpm": 130, "zcr": 0.10, "bass": 0.55, "contrast": 24},
    "Crunk":             {"bpm": 150, "zcr": 0.08, "bass": 0.60, "contrast": 25},
    "Latin Trap":        {"bpm": 130, "zcr": 0.06, "bass": 0.60, "contrast": 25},
    "UK Rap":            {"bpm": 135, "zcr": 0.07, "bass": 0.55, "contrast": 23},
    "Grime":             {"bpm": 140, "zcr": 0.09, "bass": 0.55, "contrast": 24},

    # --- 7. R&B / SOUL ---
    "Soul":              {"bpm": 80,  "zcr": 0.02, "bass": 0.30, "contrast": 20},
    "Neo-Soul":          {"bpm": 85,  "zcr": 0.03, "bass": 0.35, "contrast": 21},
    "Contemp. R&B":      {"bpm": 95,  "zcr": 0.04, "bass": 0.45, "contrast": 23},
    "Funk":              {"bpm": 110, "zcr": 0.06, "bass": 0.50, "contrast": 24},
    "Disco":             {"bpm": 120, "zcr": 0.05, "bass": 0.45, "contrast": 25},

    # --- 8. JAZZ ---
    "Smooth Jazz":       {"bpm": 90,  "zcr": 0.03, "bass": 0.25, "contrast": 21},
    "Bebop":             {"bpm": 160, "zcr": 0.07, "bass": 0.30, "contrast": 24},
    "Swing":             {"bpm": 140, "zcr": 0.05, "bass": 0.35, "contrast": 23},
    "Cool Jazz":         {"bpm": 100, "zcr": 0.03, "bass": 0.25, "contrast": 20},
    "Latin Jazz":        {"bpm": 120, "zcr": 0.06, "bass": 0.40, "contrast": 24},
    "Free Jazz":         {"bpm": 130, "zcr": 0.10, "bass": 0.30, "contrast": 26},
    "Acid Jazz":         {"bpm": 110, "zcr": 0.06, "bass": 0.45, "contrast": 23},

    # --- 9. COUNTRY ---
    "Classic Country":   {"bpm": 90,  "zcr": 0.04, "bass": 0.25, "contrast": 19},
    "Country Pop":       {"bpm": 110, "zcr": 0.04, "bass": 0.30, "contrast": 22},
    "Bluegrass":         {"bpm": 140, "zcr": 0.06, "bass": 0.20, "contrast": 20},
    "Country Rock":      {"bpm": 120, "zcr": 0.06, "bass": 0.35, "contrast": 21},

    # --- 10. WORLD / LATIN ---
    "Latin Pop":         {"bpm": 105, "zcr": 0.04, "bass": 0.35, "contrast": 24},
    "Reggaeton":         {"bpm": 95,  "zcr": 0.05, "bass": 0.55, "contrast": 25},
    "Salsa":             {"bpm": 180, "zcr": 0.07, "bass": 0.40, "contrast": 25},
    "Bachata":           {"bpm": 130, "zcr": 0.04, "bass": 0.35, "contrast": 22},
    "Merengue":          {"bpm": 150, "zcr": 0.06, "bass": 0.40, "contrast": 24},
    "Flamenco":          {"bpm": 110, "zcr": 0.06, "bass": 0.25, "contrast": 23},
    "Turkish Pop":       {"bpm": 120, "zcr": 0.05, "bass": 0.40, "contrast": 24},
    "Arab Pop":          {"bpm": 110, "zcr": 0.05, "bass": 0.45, "contrast": 23},
    "Afrobeat":          {"bpm": 100, "zcr": 0.05, "bass": 0.50, "contrast": 22},
    "Bollywood":         {"bpm": 115, "zcr": 0.06, "bass": 0.40, "contrast": 25},

    # --- 11. BLUES & FOLK ---
    "Delta Blues":       {"bpm": 80,  "zcr": 0.05, "bass": 0.25, "contrast": 20},
    "Chicago Blues":     {"bpm": 100, "zcr": 0.06, "bass": 0.35, "contrast": 21},
    "Electric Blues":    {"bpm": 110, "zcr": 0.07, "bass": 0.35, "contrast": 22},
    "Indie Folk":        {"bpm": 100, "zcr": 0.03, "bass": 0.20, "contrast": 21},
    "Celtic Folk":       {"bpm": 110, "zcr": 0.05, "bass": 0.25, "contrast": 20},

    # --- 14. SOUNDTRACK ---
    "Film Score":        {"bpm": 80,  "zcr": 0.03, "bass": 0.30, "contrast": 22},
    "Game OST":          {"bpm": 120, "zcr": 0.05, "bass": 0.35, "contrast": 24},
    "Epic Music":        {"bpm": 130, "zcr": 0.08, "bass": 0.50, "contrast": 26},
    "Instrumental":      {"bpm": 90,  "zcr": 0.02, "bass": 0.25, "contrast": 18},

    # --- 15. EXPERIMENTAL ---
    "Avant-Garde":       {"bpm": 100, "zcr": 0.10, "bass": 0.30, "contrast": 18},
    "Noise":             {"bpm": 140, "zcr": 0.25, "bass": 0.40, "contrast": 12},
    "Drone":             {"bpm": 50,  "zcr": 0.08, "bass": 0.60, "contrast": 15},
    "Glitch":            {"bpm": 130, "zcr": 0.15, "bass": 0.40, "contrast": 20},

    # --- 16. CHILL / NEW AGE ---
    "Meditation":        {"bpm": 60,  "zcr": 0.01, "bass": 0.10, "contrast": 14},
    "Chillwave":         {"bpm": 90,  "zcr": 0.02, "bass": 0.35, "contrast": 20},
    "Vaporwave":         {"bpm": 85,  "zcr": 0.03, "bass": 0.35, "contrast": 18},
    "Synthwave":         {"bpm": 110, "zcr": 0.04, "bass": 0.45, "contrast": 24},
    "Retrowave":         {"bpm": 120, "zcr": 0.04, "bass": 0.50, "contrast": 25},

    # --- 17. OTHERS (PHONK ETC) ---
    "Phonk":             {"bpm": 160, "zcr": 0.12, "bass": 0.65, "contrast": 21},
    "Memphis Phonk":     {"bpm": 145, "zcr": 0.10, "bass": 0.60, "contrast": 20},
    "Drift Phonk":       {"bpm": 165, "zcr": 0.15, "bass": 0.75, "contrast": 22}, # Ən çox istənilən :)
    "Vaportrap":         {"bpm": 130, "zcr": 0.05, "bass": 0.50, "contrast": 20},
    "Chiptune":          {"bpm": 150, "zcr": 0.04, "bass": 0.20, "contrast": 18},
    "Mashup":            {"bpm": 128, "zcr": 0.06, "bass": 0.45, "contrast": 24}
}

# ================================
# 🎛 FEATURE EXTRACTION (Optimize + Sanitize)
# ================================
def extract_features(path):
    # RAM üçün: Yalnız 30 san, 22050Hz, Mono
    try:
        y, sr = librosa.load(path, duration=30, sr=22050, mono=True)
    except:
        return None

    # 1. BPM (Ritm)
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        bpm = safe_float(tempo[0])
    except:
        bpm = 100.0

    # 2. ZCR (Sərtlik)
    zcr = safe_float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # 3. Spectral Contrast
    try:
        S = np.abs(librosa.stft(y))
        contrast = safe_float(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)))
    except:
        contrast = 20.0 # Default

    # 4. Bass Energy
    try:
        spec = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1 / sr)
        total_energy = np.sum(spec)
        
        if total_energy > 0:
            bass_energy = safe_float(np.sum(spec[(freqs < 250)]) / total_energy)
        else:
            bass_energy = 0.0
    except:
        bass_energy = 0.0

    return {
        "bpm": bpm,
        "zcr": zcr,
        "contrast": contrast,
        "bass": bass_energy
    }

# ================================
# 🧠 EN YAXIN JANRI TAP (Math)
# ================================
def find_best_match(features):
    if features is None:
        return "Unknown"

    bpm = features['bpm']
    best_genre = "General"
    min_distance = float('inf')

    # Konsola nə tapdığını yazır (Logs-da görəcəksən)
    print(f"🔍 ANALİZ: BPM={bpm:.0f} | ZCR={features['zcr']:.3f} | Bass={features['bass']:.2f} | Contrast={features['contrast']:.1f}")

    for genre, profile in GENRE_PROFILES.items():
        # --- MƏSAFƏ HESABLAMA ---
        
        bpm_diff = abs(bpm - profile['bpm'])
        # Trap/Dubstep üçün 70/140 problemini həll edirik
        if bpm_diff > 40:
             bpm_diff = min(bpm_diff, abs(bpm/2 - profile['bpm']), abs(bpm*2 - profile['bpm']))

        zcr_diff = abs(features['zcr'] - profile['zcr']) * 60 
        bass_diff = abs(features['bass'] - profile['bass']) * 25
        contrast_diff = abs(features['contrast'] - profile['contrast']) * 1.5

        total_dist = (bpm_diff * 1.2) + zcr_diff + bass_diff + contrast_diff

        if total_dist < min_distance:
            min_distance = total_dist
            best_genre = genre

    return best_genre

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
        genre = find_best_match(f)
    except Exception as e:
        print(f"Xəta: {e}")
        return {"error": str(e), "genre": "Unknown"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Return edərkən də təhlükəsizlik üçün bir daha yoxlayırıq
    return {
        "genre": genre,
        "features": {
            "bpm": safe_float(f['bpm']),
            "zcr": safe_float(f['zcr']),
            "bass": safe_float(f['bass']),
            "contrast": safe_float(f['contrast'])
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
