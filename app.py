import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image

# Sembunyikan Error MediaPipe & Siapkan Status
MEDIAPIPE_READY = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_READY = True
except Exception:
    pass 

st.set_page_config(page_title="BODY SCAN AI", layout="centered")

st.title("BODY SCAN AI")
st.markdown("Sistem Skrining Komposisi Tubuh & Analisis Postur")
st.write("---")

# 1. INPUT DATA PENGGUNA
st.header("1. Input Data Pengguna")
col1, col2 = st.columns(2)
with col1:
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
    usia = st.number_input("Usia (tahun)", 10, 100, 22)
    tinggi = st.number_input("Tinggi Badan (cm)", 100, 250, 173)
with col2:
    berat = st.number_input("Berat Badan (kg)", 30, 200, 79)
    lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", 40, 150, 90)
    tidur = st.number_input("Durasi Tidur (jam)", 2, 12, 6)
aktivitas = st.selectbox("Aktivitas Fisik", ["Rendah", "Sedang", "Tinggi"])
st.write("---")

# 2. AMBIL FOTO & COMPUTER VISION
st.header("2. Ambil Foto Tubuh (Computer Vision)")
st.info("Sistem mendeteksi subjek manusia dengan Filter Super Ketat. Benda mati tertolak otomatis.")
foto_kamera = st.camera_input("Ambil Foto Postur Tubuh")

pose_detected = False
if foto_kamera is not None:
    image = Image.open(foto_kamera)
    image_np = np.array(image)
    validasi_sukses = False
    metode_deteksi = ""
    
    if MEDIAPIPE_READY:
        # Rencana A: Validasi Sendi MediaPipe (Sensitivitas Sangat Tinggi 80%)
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.8) as pose:
            results = pose.process(image_np)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image_np, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                validasi_sukses = True
                metode_deteksi = "MediaPipe Pose"
                
    # Rencana B: Jika MediaPipe gagal, panggil OpenCV
    if not validasi_sukses:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # FILTER KETAT ANTI-HALUSINASI (minNeighbors=8, minSize=60x60)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(60, 60))
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 3)
            validasi_sukses = True
            metode_deteksi = "OpenCV Face AI"

    if validasi_sukses:
        st.image(image_np, caption=f"Divalidasi oleh {metode_deteksi}", use_column_width=True)
        st.success("✅ Validasi Berhasil: Subjek manusia terdeteksi!")
        pose_detected = True
    else:
        st.image(image_np, caption="Objek Tidak Dikenali", use_column_width=True)
        st.error("❌ Gagal: Sistem mendeteksi benda mati/bukan manusia!")
        pose_detected = False
st.write("---")

# 3. PENGGABUNGAN DATA & MACHINE LEARNING
st.header("3. Analisis Machine Learning")
if st.button("⚙️ Gabungkan Data & Proses Analisis", type="primary", use_container_width=True):
    if foto_kamera is None:
        st.error("Gagal diproses: Anda belum mengambil foto tubuh!")
    elif not pose_detected:
        st.error("Gagal diproses: Foto tertolak! Sistem menolak memproses benda mati.")
    else:
        st.write("⏳ *Menggabungkan data foto (CV) dan data input numerik...*")
        
        tinggi_m = tinggi / 100
        bmi = berat / (tinggi_m ** 2)
        
        try:
            with open('bodyfat_model.pkl', 'rb') as f:
                model_ai = pickle.load(f)
            body_fat = model_ai.predict([[usia, berat, tinggi, lingkar_pinggang, bmi]])[0]
            metode = "Gradient Boosting Regressor"
        except Exception:
            jk_val = 1 if jenis_kelamin == "Pria" else 0
            body_fat = (1.20 * bmi) + (0.23 * usia) - (10.8 * jk_val) - 5.4
            metode = "Rumus Medis Standar"

        batas_pinggang = 90 if jenis_kelamin == "Pria" else 80
        visceral_risk = lingkar_pinggang > batas_pinggang
        
        if bmi < 18.5: status_imt = "Underweight"
        elif 18.5 <= bmi < 24.9: status_imt = "Ideal"
        elif 25 <= bmi < 29.9: status_imt = "Overweight"
        else: status_imt = "Obesitas"

        # 4. HASIL SKRINING
        st.write("---")
        st.header("4. Hasil Skrining Tubuh")
        st.caption(f"Diolah menggunakan integrasi CV & {metode}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("IMT (Indeks Massa Tubuh)", f"{bmi:.1f}", status_imt, delta_color="off" if status_imt=="Ideal" else "inverse")
        c2.metric("Estimasi Body Fat", f"{body_fat:.1f}%")
        c3.metric("Status Validasi CV", "Valid", "Terdeteksi", delta_color="normal")
        
        st.write("### 💖 Rekomendasi Gaya Hidup")
        st.markdown("🏃 **Olahraga (Aktivitas Fisik):**")
        if status_imt == "Obesitas" or visceral_risk:
            st.info("Fokus pada *Low-impact cardio* (berenang, sepeda, jalan cepat) durasi 45 menit/hari untuk membakar lemak visceral tanpa membebani sendi lutut.")
        else:
            st.info("Kombinasikan Latihan Kekuatan Otot (Hipertrofi) dan *HIIT Cardio* 3-4 kali seminggu untuk menjaga persentase lemak tetap ideal.")
            
        st.markdown("🍴 **Pola Makan (Saran Menu):**")
        if visceral_risk:
            st.warning("Lingkar pinggang melewati batas aman. Kurangi karbohidrat sederhana dan perbanyak serat. Batasi surplus kalori harian.")
        else:
            st.success("Terapkan pola makan seimbang (Piring Makanku). Pertahankan asupan protein untuk menjaga massa otot yang terdeteksi.")
            
        st.markdown("🌙 **Tidur (Kualitas):**")
        if tidur < 7:
            st.error(f"Durasi tidur Anda ({tidur} jam) kurang dari rekomendasi medis (7-8 jam). Hal ini dapat mengganggu regulasi hormon kortisol.")
        else:
            st.success(f"Durasi tidur optimal ({tidur} jam). Pertahankan untuk regenerasi sel tubuh yang maksimal.")
        st.balloons()