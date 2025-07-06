
import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MedRegBot", layout="wide")


col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("logo_left.jpeg", width=120)
with col2:
    st.markdown("<h1 style='text-align: center; color: #003366;'>MedRegBot</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #005580;'>Votre assistant IA pour la réglementation des dispositifs médicaux</h4>", unsafe_allow_html=True)
with col3:
    st.image("logo_right.jpeg", width=100)
st.markdown("---")

with open("embeddings_from_drive.pkl", "rb") as f:
    data = pickle.load(f)

model = SentenceTransformer("sentence-transformers/LaBSE")

left, center, right = st.columns([2, 6, 2])

with left:
    st.markdown("### 🕓 Historique")
    if "history" in st.session_state:
        for role, msg in st.session_state.history:
            if role == "user":
                st.markdown(f"👤 {msg}")
            else:
                st.markdown(f"🤖 {msg}")

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

        question_embedding = model.encode([user_input])
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-3:][::-1]

        top_chunks = [texts[i] for i in top_indices]
        top_sources = [sources[i] for i in top_indices]

        response = "**Voici une réponse extraite des documents réglementaires :**\n\n"
        for i, chunk in enumerate(top_chunks):
            response += f"- 📄 *{top_sources[i]}*\n{chunk}\n\n"

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.history.append(("assistant", response))

with right:
    st.markdown("### 🌐 Paramètres")
    lang = st.selectbox("Langue", ["Français", "English"])
    uploaded_file = st.file_uploader("📄 Télécharger un fichier PDF", type="pdf")
    if uploaded_file:
        st.success("✅ PDF importé avec succès ! (Traitement à venir...)")
