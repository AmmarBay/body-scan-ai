import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from PIL import Image

MEDIAPIPE_READY = False
MEDIAPIPE_IMPORT_ERROR = None
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_READY = True
    else:
        MEDIAPIPE_IMPORT_ERROR = "Paket MediaPipe tidak menyediakan modul solutions."
except ImportError as exc:
    MEDIAPIPE_IMPORT_ERROR = str(exc)

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
st.info(
    "Sistem memvalidasi pose tubuh. Pastikan kepala dan kedua bahu terlihat jelas; "
    "deteksi wajah saja tidak dianggap sebagai pose tubuh."
)
foto_kamera = st.camera_input("Ambil Foto Postur Tubuh")

pose_detected = False
if foto_kamera is not None:
    image = Image.open(foto_kamera).convert("RGB")
    image_np = np.array(image)

    if not MEDIAPIPE_READY:
        st.image(image_np, caption="Validasi pose tidak tersedia", use_column_width=True)
        st.error(
            "MediaPipe Pose tidak berhasil dimuat, sehingga foto tidak dapat "
            "divalidasi. Periksa instalasi dependensi MediaPipe."
        )
        if MEDIAPIPE_IMPORT_ERROR:
            st.caption(MEDIAPIPE_IMPORT_ERROR)
    else:
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.8) as pose:
            results = pose.process(image_np)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image_np, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark
                shoulder_indices = (
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                )
                pose_detected = all(
                    index < len(landmarks) and landmarks[index].visibility >= 0.5
                    for index in shoulder_indices
                )

        if pose_detected:
            st.image(image_np, caption="Pose tervalidasi oleh MediaPipe", use_column_width=True)
            st.success("✅ Pose tubuh terdeteksi; kedua bahu terlihat jelas.")
        else:
            st.image(image_np, caption="Validasi pose tertolak", use_column_width=True)
            st.error(
                "Pose tubuh belum terdeteksi dengan cukup jelas. Pastikan kepala "
                "dan kedua bahu masuk ke dalam foto, pencahayaan cukup, dan kamera "
                "tidak terlalu dekat."
            )
st.write("---")

# 3. PENGGABUNGAN DATA & MACHINE LEARNING
st.header("3. Analisis Machine Learning")
if st.button("⚙️ Gabungkan Data & Proses Analisis", type="primary", use_container_width=True):
    if foto_kamera is None:
        st.error("Gagal diproses: Anda belum mengambil foto tubuh!")
    elif not pose_detected:
        st.error("Gagal diproses: Foto tertolak oleh sistem Computer Vision.")
    else:
        st.write(
            "⏳ *Menghitung estimasi dari data numerik setelah foto divalidasi...*"
        )
        
        tinggi_m = tinggi / 100
        bmi = berat / (tinggi_m ** 2)
        
        model_path = Path(__file__).resolve().parent / "bodyfat_extended_model.pkl"
        try:
            with model_path.open("rb") as model_file:
                model_artifact = pickle.load(model_file)
        except FileNotFoundError:
            st.error(
                "Model BodyFat belum tersedia. Jalankan `python train_model.py` "
                "terlebih dahulu."
            )
            st.stop()
        except (
            OSError,
            pickle.UnpicklingError,
            EOFError,
            ImportError,
            AttributeError,
        ) as exc:
            st.error(f"Model BodyFat tidak dapat dibaca: {exc}")
            st.stop()

        expected_features = [
            "sex_encoded",
            "age_years",
            "weight_kg",
            "height_m",
            "abdomen_cm",
            "bmi",
        ]
        if (
            not isinstance(model_artifact, dict)
            or model_artifact.get("feature_names") != expected_features
            or "model" not in model_artifact
        ):
            st.error(
                "Format model tidak cocok dengan input aplikasi. "
                "Jalankan ulang `python train_model.py`."
            )
            st.stop()

        input_features = [
            1 if jenis_kelamin == "Pria" else 0,
            usia,
            berat,
            tinggi_m,
            lingkar_pinggang,
            bmi,
        ]
        try:
            predictions = np.asarray(
                model_artifact["model"].predict([input_features]), dtype=float
            )
        except (AttributeError, TypeError, ValueError) as exc:
            st.error(f"Gagal menghitung estimasi BodyFat: {exc}")
            st.stop()

        if predictions.size != 1 or not np.isfinite(predictions[0]):
            st.error("Model menghasilkan estimasi BodyFat yang tidak valid.")
            st.stop()

        body_fat = float(predictions[0])
        metode = model_artifact.get("model_name", "Model machine learning")

        batas_pinggang = 90 if jenis_kelamin == "Pria" else 80
        visceral_risk = lingkar_pinggang > batas_pinggang
        
        if bmi < 18.5: status_imt = "Underweight"
        elif 18.5 <= bmi < 24.9: status_imt = "Ideal"
        elif 25 <= bmi < 29.9: status_imt = "Overweight"
        else: status_imt = "Obesitas"

        # 4. HASIL SKRINING
        st.write("---")
        st.header("4. Hasil Skrining Tubuh")
        st.caption(
            f"Estimasi dihitung dari input numerik setelah foto divalidasi oleh CV, "
            f"menggunakan {metode}."
        )
        
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