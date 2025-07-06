
import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(page_title="MedRegBot", layout="wide")

# ---------------------- الشعارات ----------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("logo_left.jpeg", width=120)
with col2:
    st.markdown("<h1 style='text-align: center; color: #003366;'>MedRegBot</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #005580;'>Votre assistant IA pour la réglementation des dispositifs médicaux</h4>", unsafe_allow_html=True)
with col3:
    st.image("logo_right.jpeg", width=100)
st.markdown("---")

# ---------------------- تحميل قاعدة البيانات ----------------------
with open("embeddings_from_drive.pkl", "rb") as f:
    data = pickle.load(f)

texts = [d["text"] for d in data]
sources = [d.get("source", "Inconnu") for d in data]
embeddings = np.array([d["embedding"] for d in data])

model = SentenceTransformer("sentence-transformers/LaBSE")

# ---------------------- إعداد الستايل ----------------------
whatsapp_style = '''
<style>
.user-msg {
    background-color: #ffffff;
    color: black;
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 70%;
    margin: 10px 0;
    margin-left: auto;
    text-align: right;
}
.bot-msg {
    background-color: #e6f0ff;
    color: black;
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 70%;
    margin: 10px 0;
    margin-right: auto;
    text-align: left;
}
</style>
'''

st.markdown(whatsapp_style, unsafe_allow_html=True)

# ---------------------- Session state ----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------- واجهة المحادثة ----------------------
st.markdown("### 💬 Chat avec MedRegBot")

for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg}</div>", unsafe_allow_html=True)

user_input = st.chat_input("Posez votre question ici...")
if user_input:
    st.session_state.history.append(("user", user_input))

    # --------- retrieval des chunks ---------
    question_embedding = model.encode([user_input])
    similarities = cosine_similarity(question_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]
    context = "\n".join([texts[i] for i in top_indices])

    # --------- Construction du prompt pour Gemma ---------
    prompt = f"Tu es un expert en réglementation des dispositifs médicaux. En te basant sur le texte suivant, réponds de façon claire et complète à la question utilisateur.\n\nTexte: {context}\n\nQuestion: {user_input}\n\nRéponse:"

    # --------- Appel à l'API Hugging Face Inference (Gemma) --------
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b-it"
   
HF_TOKEN = st.secrets["HF_TOKEN"]
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

response = requests.post(API_URL, headers=headers, json={"inputs": prompt})  # <-- Correction ici (enlevé 4 espaces)
if response.status_code == 200:
    answer = response.json()[0]["generated_text"].split("Réponse:")[-1].strip()
else:
    answer = "⚠️ Une erreur est survenue avec l'API Hugging Face."

    st.markdown(f"<div class='bot-msg'>{answer}</div>", unsafe_allow_html=True)
    st.session_state.history.append(("bot", answer))
