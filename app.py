import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from requests.exceptions import RequestException
import time  # Ajoutez ceci avec les autres imports
from tenacity import retry, stop_after_attempt, wait_exponential  # Pour les réessais automatiques

st.set_page_config(page_title="MedRegBot", layout="wide")

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

# ---------------------- Chargement de la base de données ----------------------
@st.cache_resource
def load_data():
    with open("embeddings_from_drive.pkl", "rb") as f:
        data = pickle.load(f)
    texts = [d["text"] for d in data]
    sources = [d.get("source", "Inconnu") for d in data]
    embeddings = np.array([d["embedding"] for d in data])
    return texts, sources, embeddings

texts, sources, embeddings = load_data()
model = SentenceTransformer("sentence-transformers/LaBSE")

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
</style>
'''
st.markdown(whatsapp_style, unsafe_allow_html=True)

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

user_input = st.chat_input("Posez votre question ici...")
if user_input:
    # Étape 1: Ajout du message utilisateur
    st.session_state.history.append(("user", user_input))
    loading_msg = st.empty()  # Crée un placeholder pour le message de chargement
    
    try:
        # Étape 2: Recherche contextuelle (rapide)
        loading_msg.markdown("<div class='loading-msg'>🔍 Recherche dans les textes réglementaires...</div>", unsafe_allow_html=True)
        question_embedding = model.encode([user_input])
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-3:][::-1]
        context = "\n".join([texts[i] for i in top_indices])
        
        # Étape 3: Appel API avec gestion du temps
        loading_msg.markdown("<div class='loading-msg'>🧠 Analyse par l'expert IA...</div>", unsafe_allow_html=True)
        start_time = time.time()
        
        response = query_mistral(user_input, context)
        
        # Feedback si délai dépassé
        if time.time() - start_time > 15:
            loading_msg.markdown("<div class='loading-msg'>⏳ L'analyse prend plus de temps que prévu...</div>", unsafe_allow_html=True)
        
        # Étape 4: Traitement réponse
        if response.status_code == 200:
            generated_text = response.json()[0]["generated_text"]
            answer = generated_text.split("[/INST]")[-1].strip()
        elif response.status_code == 503:
            answer = "⚠️ Le système est occupé. Merci de reformuler votre question."
        else:
            answer = f"⚠️ Réponse partielle (code {response.status_code})"
            
    except Exception as e:
        answer = f"⚠️ Service temporairement indisponible: {str(e)}"
    finally:
        loading_msg.empty()  # Supprime le message de chargement
    
    # Étape 5: Affichage réponse
    st.session_state.history.append(("bot", answer))

        # --------- Appel à l'API Hugging Face (Mistral gratuit) ---------
        # ---------------------- Configuration API ----------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_mistral(prompt, context):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    headers = {
        "Authorization": f"Bearer {st.secrets.get('HF_TOKEN', '')}",
        "Content-Type": "application/json"
    }
    
    full_prompt = f"""<s>[INST] Tu es un expert en réglementation médicale. 
    Réponds en français en 3-5 phrases maximum en t'appuyant sur ce contexte:
    
    Contexte: {context}
    
    Question: {prompt} [/INST]"""
    
    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 350,  # Réponse plus courte
                "temperature": 0.7  # Moins de créativité, plus factuel
            }
        },
        timeout=20  # Timeout plus court
    )
    return response

        # Mise à jour du message de chargement
        st.session_state.history = [h for h in st.session_state.history if h[0] != "loading"]
        st.session_state.history.append(("loading", "Génération de la réponse..."))
        st.rerun()

        with st.spinner(""):
            response = requests.post(
                API_URL,
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": 500}
                },
                timeout=45
            )
        
        if response.status_code == 200:
            generated_text = response.json()[0]["generated_text"]
            answer = generated_text.split("[/INST]")[-1].strip()
        elif response.status_code == 503:
            answer = "⚠️ Le modèle est en cours de chargement. Merci de réessayer dans 30 secondes."
        else:
            answer = f"⚠️ Erreur API (code {response.status_code}): {response.text[:200]}..."

    except RequestException as e:
        answer = f"⚠️ Erreur de connexion: {str(e)}"
    except Exception as e:
        answer = f"⚠️ Erreur inattendue: {str(e)}"

    # Mise à jour finale de l'historique
    st.session_state.history = [h for h in st.session_state.history if h[0] != "loading"]
    st.session_state.history.append(("bot", answer))
