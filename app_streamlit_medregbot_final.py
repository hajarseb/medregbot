
import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# ---------------------- إعداد الصفحة ----------------------
st.set_page_config(page_title="MedRegBot", layout="wide")

# ---------------------- تحميل الشعارات من ملفات محلية ----------------------
logo_left = Image.open("easy medical device.jpeg")
logo_right = Image.open("fmpm.jpeg")

# ---------------------- تصميم العناوين العلوية ----------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image(logo_left, width=120)
with col2:
    st.markdown("<h1 style='text-align: center; color: #003366;'>MedRegBot</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #005580;'>Votre assistant IA pour la réglementation des dispositifs médicaux</h4>", unsafe_allow_html=True)
with col3:
    st.image(logo_right, width=100)
st.markdown("---")

# ---------------------- تحميل Embeddings ----------------------
with open("embeddings_from_drive.pkl", "rb") as f:
    data = pickle.load(f)

texts = [d["text"] for d in data]
sources = [d.get("source", "Inconnu")] * len(data)
embeddings = np.array([d["embedding"] for d in data])

model = SentenceTransformer("sentence-transformers/LaBSE")

# ---------------------- تنظيم الأعمدة ----------------------
left, center, right = st.columns([2, 6, 2])

# ------ العمود الأيسر: التاريخ ------
with left:
    st.markdown("### 🕓 Historique")
    if "history" in st.session_state:
        for i, (role, msg) in enumerate(st.session_state.history):
            if role == "user":
                st.markdown(f"👤 {msg}")
            else:
                st.markdown(f"🤖 {msg}")

# ------ العمود الأوسط: المحادثة ------
with center:
    st.markdown("### 💬 Chat")
    if "history" not in st.session_state:
        st.session_state.history = []

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    user_input = st.chat_input("Posez votre question ici...")
    if user_input:
        st.session_state.history.append(("user", user_input))

        # استرجاع السياق الأقرب
        question_embedding = model.encode([user_input])
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-3:][::-1]

        top_chunks = [texts[i] for i in top_indices]
        top_sources = [sources[i] for i in top_indices]

        # تركيب الجواب
        response = "**Voici une réponse extraite des documents réglementaires :**\n\n"
        for i, chunk in enumerate(top_chunks):
            response += f"- 📄 *{top_sources[i]}*\n{chunk}\n\n"

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.history.append(("assistant", response))

# ------ العمود الأيمن: Upload PDF + langue ------
with right:
    st.markdown("### 🌐 Paramètres")
    lang = st.selectbox("Langue", ["Français", "English"])
    uploaded_file = st.file_uploader("📄 Télécharger un fichier PDF", type="pdf")
    if uploaded_file:
        st.success("✅ PDF importé avec succès ! (Traitement à venir...)")
