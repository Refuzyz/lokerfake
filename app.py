# Import library yang dibutuhkan
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import easyocr
from pypdf import PdfReader
from PIL import Image

# --- Simulasi Data & Model agar Aplikasi Mandiri ---
data = {
    'teks_lowongan': [
        "Dibutuhkan segera IT Support. Syarat: lulusan SMK TKJ, paham routing Mikrotik, penempatan Jakarta Timur.",
        "KERJA MUDAH GAJI JUTAAN! Cukup balas chat dari rumah. Transfer biaya pendaftaran member 250rb ke rekening admin.",
        "Lowongan Network Engineer. Menguasai Cisco Packet Tracer, instalasi LAN, dan konfigurasi server.",
        "Dicari kandidat pegawai BUMN tanpa tes! Kuota terbatas. Bayar biaya akomodasi tiket pesawat 1 juta, nanti diganti.",
        "Dibutuhkan staf magang teknisi komputer, terbiasa merakit PC dan instalasi Windows. Kirim CV ke email resmi.",
        "Kerja sampingan ketik captcha dibayar dollar. Syarat bayar deposit awal $50."
    ],
    'label': [0, 1, 0, 1, 0, 1]
}
df_temp = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(df_temp['teks_lowongan'])
model = MultinomialNB()
model.fit(X_train_vec, df_temp['label'])
# --------------------------------------------------

# Inisialisasi EasyOCR Reader (bahasa Indonesia & Inggris)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['id', 'en'])

reader = load_ocr()

st.set_page_config(page_title="Pendeteksi Loker Palsu", page_icon="🔎", layout="centered")

st.title("Pendeteksi Loker Palsu (Anti-Scam) 🔎")
st.write("Sistem Machine Learning untuk mendeteksi indikasi penipuan pada lowongan pekerjaan beserta analisis teks & dokumen.")

# Input manual
teks_input = st.text_area("Teks Lowongan (Manual)", placeholder="Paste teks lowongan di sini...", height=150)

# Unggah foto atau dokumen
file_input = st.file_uploader("Unggah Screenshot atau PDF Lowongan (Akan Dibaca Otomatis)", type=["png", "jpg", "jpeg", "pdf"])

# Proses pembacaan file secara real-time
teks_dari_file = ""
if file_input is not None:
    file_type = file_input.name.split('.')[-1].lower()
    
    with st.spinner("Membaca isi file..."):
        if file_type in ["png", "jpg", "jpeg"]:
            try:
                image = Image.open(file_input)
                # Konversi gambar ke numpy array untuk EasyOCR
                image_np = np.array(image)
                # Lakukan OCR
                hasil_ocr = reader.readtext(image_np, detail=0)
                teks_dari_file = " ".join(hasil_ocr)
                st.info("Teks berhasil diekstrak dari gambar!")
            except Exception as e:
                st.error(f"Gagal membaca gambar: {e}")
                
        elif file_type == "pdf":
            try:
                reader_pdf = PdfReader(file_input)
                teks_list = []
                for page in reader_pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        teks_list.append(extracted)
                teks_dari_file = " ".join(teks_list)
                st.info("Teks berhasil diekstrak dari PDF!")
            except Exception as e:
                st.error(f"Gagal membaca PDF: {e}")

    # Tampilkan teks hasil ekstrak
    if teks_dari_file.strip():
        with st.expander("Lihat Hasil Teks Dokumen/Gambar"):
            st.write(teks_dari_file)

if st.button("Mulai Analisis AI"):
    # Gabungkan input manual dengan hasil pembacaan dokumen jika ada
    teks_gabungan = (teks_input.strip() + " " + teks_dari_file.strip()).strip()

    if not teks_gabungan:
        st.warning("Silakan masukkan teks atau unggah file terlebih dahulu untuk dianalisis.")
    else:
        with st.spinner("Sedang menganalisis teks..."):
            # Ekstraksi fitur teks menggunakan vectorizer
            teks_vec = vectorizer.transform([teks_gabungan])

            # AI memberikan prediksi dan probabilitas
            hasil_prediksi = model.predict(teks_vec)[0]
            probabilitas = model.predict_proba(teks_vec)[0]

            prob_asli = probabilitas[0] * 100
            prob_palsu = probabilitas[1] * 100

            st.subheader("Hasil Analisis AI & Alasan Keyakinan")

            if hasil_prediksi == 1:
                st.error(f"⚠️ HATI-HATI! Loker ini terindikasi PALSU / PENIPUAN.")
                st.write(f"**Probabilitas Palsu:** {prob_palsu:.2f}% (Tinggi)")
                st.write(f"**Probabilitas Asli:** {prob_asli:.2f}%")
                st.write("**Alasan Analisis:** Sistem mendeteksi kecenderungan pola kalimat yang sering digunakan oleh penipu, seperti meminta biaya pendaftaran, janji gaji tidak realistis, atau prosedur rekrutmen tidak resmi.")
            else:
                st.success(f"✅ AMAN. Loker ini terindikasi ASLI.")
                st.write(f"**Probabilitas Asli:** {prob_asli:.2f}% (Tinggi)")
                st.write(f"**Probabilitas Palsu:** {prob_palsu:.2f}%")
                st.write("**Alasan Analisis:** Sistem menilai struktur teks ini aman dan serupa dengan karakteristik penulisan lowongan kerja formal/profesional.")
