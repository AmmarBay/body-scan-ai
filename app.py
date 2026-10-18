import streamlit as st
import cv2
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360.0 - angle
    return angle

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_READY = True
except Exception as e:
    MEDIAPIPE_READY = False

st.set_page_config(page_title="BODY SCAN AI", layout="wide", initial_sidebar_state="expanded")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/863/863620.png", width=100)
st.sidebar.title("BODY SCAN AI")
st.sidebar.markdown("**Ketua Proyek:** Ammar Baytin Nurzaman (15240123)")
st.sidebar.write("Anggota:\n- M Rofy Dwi Pamungkas\n- Ahmad Faqih Syarifuddin")

st.title("Sistem Skrining & Analisis Biomekanik AI")
st.markdown("*Minimum Viable Product (MVP) - v3.0 (Body Fat Edition)*")
st.write("---")

tab1, tab2 = st.tabs(["🎥 Pemindai Biomekanik & AI Trainer", "📊 Dasbor Lemak Tubuh & Rekomendasi"])

with tab1:
    st.header("1. Input Parameter Medis")
    col_in1, col_in2, col_in3, col_in4 = st.columns(4)
    with col_in1:
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
        usia = st.number_input("Usia (tahun)", 10, 100, 22)
    with col_in2:
        tinggi = st.number_input("Tinggi Badan (cm)", 100, 250, 173)
        berat = st.number_input("Berat Badan (kg)", 30, 200, 79)
    with col_in3:
        lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", 40, 150, 90)
        tidur = st.number_input("Durasi Tidur (jam)", 2, 12, 6)
    with col_in4:
        aktivitas = st.selectbox("Aktivitas Fisik", ["Rendah", "Sedang", "Tinggi"])

    st.write("---")
    st.header("2. AI Fitness Trainer (Repetition Counter)")
    
    kamera_aktif = st.checkbox("🟢 Nyalakan Kamera AI Trainer")
    frame_window = st.image([]) 
    
    if kamera_aktif:
        cap = cv2.VideoCapture(0)
        counter = 0
        stage = None
        if MEDIAPIPE_READY:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while kamera_aktif:
                    ret, frame = cap.read()
                    if not ret: break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)
                    
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        landmarks = results.pose_landmarks.landmark
                        bahu = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        siku = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        pergelangan = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        sudut_siku = calculate_angle(bahu, siku, pergelangan)
                        
                        if sudut_siku > 160: stage = "TURUN"
                        if sudut_siku < 50 and stage == 'TURUN':
                            stage = "NAIK"
                            counter += 1
                            
                        siku_px = tuple(np.multiply(siku, [640, 480]).astype(int))
                        cv2.putText(frame_rgb, str(int(sudut_siku)) + " derajat", siku_px, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(frame_rgb, (0,0), (640, 70), (0,0,0), -1)
                        cv2.putText(frame_rgb, f"STATUS: {stage} | TOTAL REPS: {counter}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    frame_window.image(frame_rgb)
        cap.release()
    else:
        st.info("Kamera dinonaktifkan.")

with tab2:
    st.header("Dasbor Analisis Komposisi Tubuh (Body Fat)")
    
    if st.button("🔄 Analisis Komposisi Tubuh", type="primary"):
        tinggi_m = tinggi / 100
        bmi = berat / (tinggi_m ** 2)
        
        # Algoritma Body Fat Percentage (Deurenberg Formula)
        jk_val = 1 if jenis_kelamin == "Pria" else 0
        body_fat = (1.20 * bmi) + (0.23 * usia) - (10.8 * jk_val) - 5.4
        
        # Kategori Lemak Tubuh
        if jenis_kelamin == "Pria":
            if body_fat < 14: bf_cat = "Atletis / Sangat Bugar"
            elif 14 <= body_fat < 25: bf_cat = "Kategori Normal"
            else: bf_cat = "Kelebihan Lemak (Risiko Obesitas)"
        else:
            if body_fat < 21: bf_cat = "Atletis / Sangat Bugar"
            elif 21 <= body_fat < 32: bf_cat = "Kategori Normal"
            else: bf_cat = "Kelebihan Lemak (Risiko Obesitas)"

        if bmi < 18.5:
            kategori, saran = "Underweight", "Fokus latihan hipertrofi beban, tingkatkan surplus kalori bernutrisi."
        elif 18.5 <= bmi < 24.9:
            kategori, saran = "Ideal", "Pertahankan rutinitas, seimbangkan latihan kardio dan resistensi otot."
        elif 25 <= bmi < 29.9:
            kategori, saran = "Overweight", "Terapkan defisit kalori, fokus kardio HIIT."
        else:
            kategori, saran = "Obesitas", "Kardio low-impact untuk menjaga sendi, perketat pengaturan diet."

        st.subheader("1. Metrik Komposisi Tubuh")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Skor IMT", f"{bmi:.1f}", kategori, delta_color="off" if kategori=="Ideal" else "inverse")
        col_m2.metric("Estimasi Lemak Tubuh", f"{body_fat:.1f}%", bf_cat, delta_color="off" if bf_cat=="Kategori Normal" or bf_cat=="Atletis / Sangat Bugar" else "inverse")
        
        st.progress(min(int((body_fat / 40) * 100), 100), text="Persentase Lemak Tubuh")

        st.subheader("2. Resep Gaya Hidup AI")
        st.info(f"🏋️ **Program Latihan:** {saran}")
        if tidur < 7: st.error("🛌 **Regenerasi:** Kurang dari 7 jam! Berisiko menghambat metabolisme pembakaran lemak.")
        else: st.success("🛌 **Regenerasi:** Optimal. Pertahankan untuk regulasi hormon.")
        # --- LOGIKA INDIKATOR METABOLIK CERDAS ---
        batas_pinggang = 90 if jenis_kelamin == "Pria" else 80
        
        if lingkar_pinggang > batas_pinggang:
            st.warning(f"⚠️ **Indikator Metabolik:** Lingkar pinggang ({lingkar_pinggang} cm) melebihi batas aman {jenis_kelamin} ({batas_pinggang} cm). Ini menandakan penumpukan lemak visceral perut. Kurangi asupan gula dan tingkatkan kardio.")
        else:
            st.success(f"✅ **Indikator Metabolik:** Lingkar pinggang ({lingkar_pinggang} cm) aman dari risiko sindrom metabolik (Batas {jenis_kelamin}: {batas_pinggang} cm).")
        # -----------------------------------------
        st.balloons()