import streamlit as st
from pathlib import Path
import joblib

st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon=":mag:",
    layout="centered"
)

st.title(
    "Toxic Comment Classifier"
)

st.write("Mendeteksi Komentar Toxic di Media Sosial")

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent.parent / "model" / "pipeline_final.joblib"

    print("MODEL PATH:", model_path)
    print("EXISTS:", model_path.exists())

    return joblib.load(model_path)

pipeline = load_model()


def predict(text):
    # TODO: predict label
    label_num = pipeline.predict([text])[0]
    
    # TODO: ambil probabilities
    proba = pipeline.predict_proba([text])[0]

    # TODO: tentukan label string dan confidence yang sesuai
    if label_num == 1:
        label      = 'TOXIC'
        confidence = proba[1]
    else:
        label      = 'NON-TOXIC'
        confidence = proba[0]
    
    return label, confidence

# TODO: text area
user_input = st.text_area(
    label="Masukkan teks komentar:",
    height=100,
    placeholder="Masukkan Komentar..",
    label_visibility="collapsed"
)

# TODO: tombol
submitted = st.button(
    label="Cek Komentar",
    use_container_width=True
)

if submitted:
    # TODO: kondisi 1 — input kosong
    if not user_input.strip():
        st.warning("Komentar tidak boleh kosong")

    # TODO: kondisi 2 — ada input, lakukan prediksi
    else:
        label, confidence = predict(user_input)

        if label == "TOXIC":
            st.error(f"**{label}** — {confidence:.1%} confidence")
        else:
            st.success(f"**{label}** — {confidence:.1%} confidence")

        st.progress(float(confidence))

st.divider()

# TODO: expander dengan contoh kalimat
with st.expander("Contoh Kalimat"):
    st.caption("Toxic:")
    st.code("lu bisa baca ga sih tolol")
    st.code("dasar goblok banget sih")
    st.code("mending lu diem aja bego")

    st.caption("Non-Toxic:")
    st.code("makasih bro udah bantu gue")
    st.code("setuju sama pendapat lo")
    st.code("wah bagus banget ini")

st.caption(
    "Model dilatih pada data Twitter Indonesia. "
    "Hasil mungkin tidak akurat untuk semua konteks."
)
