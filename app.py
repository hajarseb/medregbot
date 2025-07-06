import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from requests.exceptions import RequestException
import time
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------- Configuration de base ----------------------
st.set_page_config(page_title="MedRegBot", layout="wide")

# ---------------------- Gestion des erreurs améliorée ----------------------
@st.cache_resource
def load_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device
        )
        st.success("Modèle chargé avec succès")
        return model
    except Exception as e:
        st.error(f"Échec du chargement du modèle: {str(e)}")
        return None

@st.cache_resource
def load_data():
    try:
        with open("embeddings_from_drive.pkl", "rb") as f:
            data = pickle.load(f)
            if not data or len(data) == 0:
                raise ValueError("Fichier pickle vide ou invalide")
            
            # Validation des données
            validated_data = []
            for d in data:
                if "text" in d and "embedding" in d:
                    validated_data.append({
                        "text": d["text"],
                        "source": d.get("source", "Inconnu"),
                        "embedding": d["embedding"]
                    })
            
            if not validated_data:
                raise ValueError("Aucune donnée valide trouvée")
            
            return (
                [d["text"] for d in validated_data],
                [d["source"] for d in validated_data],
                np.array([d["embedding"] for d in validated_data])
            )
    except Exception as e:
        st.error(f"Erreur de chargement des données: {str(e)}")
        # Retourne des données minimales pour permettre le fonctionnement
        return ["Données non disponibles"], ["Système"], np.random.rand(1, 384)

# ---------------------- Initialisation ----------------------
model = load_model()
texts, sources, embeddings = load_data()

# ---------------------- Interface Utilisateur ----------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("logo_left.jpeg", width=120)
with col2:
    st.markdown("<h1 style='text-align: center; color: #003366;'>MedRegBot</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #005580;'>Votre assistant IA pour la réglementation des dispositifs médicaux</h4>", unsafe_allow_html=True)
with col3:
    st.image("logo_right.jpeg", width=100)
st.markdown("---")

# ---------------------- Style CSS ----------------------
st.markdown('''
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
.warning-msg {
    color: #cc9900;
    border-left: 3px solid #cc9900;
    padding-left: 10px;
}
</style>
''', unsafe_allow_html=True)

# ---------------------- Configuration API ----------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_mistral(prompt, context):
    try:
        if not st.secrets.get('HF_TOKEN'):
            st.error("Token API non configuré")
            return None

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
            timeout=30
        )
        return response
    
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue: {str(e)}")
        return None

# ---------------------- Gestion de l'état de session ----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------- Interface de chat ----------------------
st.markdown("### 💬 Chat avec MedRegBot")

# Affichage de l'historique
for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
    elif role == "bot":
        st.markdown(f"<div class='bot-msg'>{msg}</div>", unsafe_allow_html=True)
    elif role == "loading":
        st.markdown(f"<div class='loading-msg'>{msg}</div>", unsafe_allow_html=True)
    elif role == "error":
        st.markdown(f"<div class='error-msg'>{msg}</div>", unsafe_allow_html=True)
    elif role == "warning":
        st.markdown(f"<div class='warning-msg'>{msg}</div>", unsafe_allow_html=True)

# Gestion de l'entrée utilisateur
user_input = st.chat_input("Posez votre question ici...")
if user_input:
    st.session_state.history.append(("user", user_input))
    loading_msg = st.empty()
    
    try:
        # Étape 1: Recherche contextuelle
        loading_msg.markdown("<div class='loading-msg'>🔍 Recherche dans les textes réglementaires...</div>", unsafe_allow_html=True)
        
        if model is not None:
            question_embedding = model.encode([user_input])
            similarities = cosine_similarity(question_embedding, embeddings)[0]
            top_indices = similarities.argsort()[-3:][::-1]
            context = "\n\n".join([f"**Source:** {sources[i]}\n{texts[i]}" for i in top_indices])
        else:
            context = "\n\n".join([f"**Source:** {sources[i]}\n{texts[i]}" for i in range(min(3, len(texts)))])
            st.session_state.history.append(("warning", "Mode dégradé - utilisation des premières références"))
        
        # Étape 2: Appel API
        loading_msg.markdown("<div class='loading-msg'>🧠 Analyse par l'expert IA...</div>", unsafe_allow_html=True)
        
        response = None
        if st.secrets.get('HF_TOKEN'):
            response = query_mistral(user_input, context)
        
        # Étape 3: Traitement de la réponse
        if response is not None and response.status_code == 200:
            try:
                generated_text = response.json()[0]["generated_text"]
                answer = generated_text.split("[/INST]")[-1].strip()
            except (KeyError, IndexError) as e:
                answer = f"""⚠️ Format de réponse inattendu\n\n**Extraits pertinents:**\n{context}"""
        else:
            answer = f"""ℹ️ Service IA indisponible\n\n**Références réglementaires pertinentes:**\n{context}"""
            
    except Exception as e:
        answer = f"""⚠️ Erreur temporaire\n\n**Nous vous proposons ces extraits:**\n{
        context if 'context' in locals() else 'Aucune référence disponible'
        }"""
        st.error(f"Erreur: {str(e)}")
    
    finally:
        loading_msg.empty()
    
    st.session_state.history.append(("bot", answer))
    st.rerun()
