import streamlit as st
import joblib
import numpy as np


st.set_page_config(page_title="Prediksi Nilai siswa")

MODEL_PATH = "analisis_nilai_jam_belajar_model.sav"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Prediksi Nilai Ujian Siswa")

st.write("Masukkan jumlah jam belajar untuk memprediksi nilai ujian siswa.")

jam_belajar = st.number_input("Jam Belajar (dalam jam)", min_value=0, max_value=24, step=1)

if st.button("Prediksi"):
    try:
        data = np.array([[jam_belajar]])
        pred = model.predict(data)[0]
        st.success(f"Nilai ujian yang diprediksi: **{pred:.2f}**")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
