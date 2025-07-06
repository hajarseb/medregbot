import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from requests.exceptions import RequestException
import time
import torch  # Ajouté pour la gestion du device
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------- Configuration de base ----------------------
st.set_page_config(page_title="MedRegBot", layout="wide")

# ---------------------- Gestion des erreurs initiale ----------------------
@st.cache_resource
def load_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Utilisation d'un modèle plus léger pour compatibilité CPU
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device
        )
        return model
    except Exception as e:
        st.error(f"Échec du chargement du modèle: {str(e)}")
        return None

# ---------------------- Logos ----------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("logo_left.jpeg", width=120)
with col2:
    st.markdown("<h1 style='text-align: center; color: #003366;'>MedRegBot</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #005580;'>Votre assistant IA pour la réglementation des dispositifs médicaux</h4>", unsafe_allow_html=True)
with col3:
    st.image("logo_right.jpeg", width=100)
st.markdown("---")

# ---------------------- Chargement des données ----------------------
@st.cache_resource
def load_data():
    try:
        with open("embeddings_from_drive.pkl", "rb") as f:
            data = pickle.load(f)
        texts = [d["text"] for d in data]
        sources = [d.get("source", "Inconnu") for d in data]
        embeddings = np.array([d["embedding"] for d in data])
        return texts, sources, embeddings
    except Exception as e:
        st.error(f"Erreur de chargement des données: {str(e)}")
        return [], [], np.array([])

texts, sources, embeddings = load_data()
model = load_model()

if model is None:
    st.warning("""
    Mode dégradé activé (sans IA). 
    Fonctionnalités limitées disponibles.
    """)
    # Vous pourriez ajouter ici une logique de repli simple

# ---------------------- Style ----------------------
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
.loading-msg {
    color: #666;
    font-style: italic;
}
.error-msg {
    color: #ff3333;
    border-left: 3px solid #ff3333;
    padding-left: 10px;
}
</style>
'''
st.markdown(whatsapp_style, unsafe_allow_html=True)

# ---------------------- Configuration API ----------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_mistral(prompt, context):
    try:
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {
            "Authorization": f"Bearer {st.secrets['HF_TOKEN']}",
            "Content-Type": "application/json"
        }
        
        full_prompt = f"""<s>[INST] Tu es un expert en réglementation médicale EU. 
        Réponds en français en t'appuyant strictement sur ce contexte:
        
        Contexte: {context}
        
        Question: {prompt} [/INST]"""
        
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": 350,
                    "temperature": 0.5
                }
            },
            timeout=25
        )
        response.raise_for_status()
        return response
    except requests.HTTPError as http_err:
        st.error(f"Erreur API: {http_err}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue: {str(e)}")
        return None

# ---------------------- Session state ----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------- Interface de chat ----------------------
st.markdown("### 💬 Chat avec MedRegBot")

for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
    elif role == "bot":
        st.markdown(f"<div class='bot-msg'>{msg}</div>", unsafe_allow_html=True)
    elif role == "loading":
        st.markdown(f"<div class='loading-msg'>{msg}</div>", unsafe_allow_html=True)
    elif role == "error":
        st.markdown(f"<div class='error-msg'>{msg}</div>", unsafe_allow_html=True)

user_input = st.chat_input("Posez votre question ici...")
if user_input:
    st.session_state.history.append(("user", user_input))
    loading_msg = st.empty()
    
    try:
        # Recherche contextuelle
        loading_msg.markdown("<div class='loading-msg'>🔍 Recherche dans les textes réglementaires...</div>", unsafe_allow_html=True)
        
        if model is not None:
            question_embedding = model.encode([user_input])
            similarities = cosine_similarity(question_embedding, embeddings)[0]
            top_indices = similarities.argsort()[-3:][::-1]
            context = "\n".join([texts[i] for i in top_indices])
        else:
            # Fallback sans modèle
            context = "\n".join(texts[:3])  # Premiers textes comme contexte
        
        # Appel API
        loading_msg.markdown("<div class='loading-msg'>🧠 Analyse par l'expert IA...</div>", unsafe_allow_html=True)
        
        response = query_mistral(user_input, context) if st.secrets.get('HF_TOKEN') else None
        
        if response and response.status_code == 200:
            generated_text = response.json()[0]["generated_text"]
            answer = generated_text.split("[/INST]")[-1].strip()
        else:
            answer = """⚠️ Service IA temporairement indisponible. 
            Voici les extraits pertinents :
            \n\n""" + "\n\n- ".join([texts[i] for i in (top_indices if model else range(3))])
            
    except Exception as e:
        answer = f"<div class='error-msg'>⚠️ Erreur critique: {str(e)}</div>"
    finally:
        loading_msg.empty()
    
    st.session_state.history.append(("bot", answer))
    st.rerun()
