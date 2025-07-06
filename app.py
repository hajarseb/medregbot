import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from requests.exceptions import RequestException
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import logging

# ---------------------- Configuration initiale ----------------------
logger = logging.getLogger(__name__)
st.set_page_config(
    page_title="MedRegBot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------- Style CSS ----------------------
st.markdown('''
<style>
.user-msg {
    background-color: #f0f2f6;
    color: #000;
    padding: 12px 18px;
    border-radius: 18px 18px 0 18px;
    max-width: 80%;
    margin: 8px 0 8px auto;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.bot-msg {
    background-color: #e6f0ff;
    color: #000;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 0;
    max-width: 80%;
    margin: 8px auto 8px 0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.loading-msg {
    color: #666;
    font-style: italic;
    padding: 10px;
    background-color: #f9f9f9;
    border-radius: 8px;
    margin: 10px 0;
}
.error-msg {
    color: #d32f2f;
    padding: 12px;
    background-color: #ffebee;
    border-radius: 8px;
    border-left: 4px solid #d32f2f;
    margin: 10px 0;
}
.warning-msg {
    color: #ffa000;
    padding: 12px;
    background-color: #fff8e1;
    border-radius: 8px;
    border-left: 4px solid #ffa000;
    margin: 10px 0;
}
.stChatInput {
    position: fixed;
    bottom: 20px;
    width: 70%;
    left: 15%;
}
</style>
''', unsafe_allow_html=True)

# ---------------------- Initialisation ----------------------
@st.cache_resource(show_spinner="Chargement du modèle d'embedding...")
def load_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        logger.info("Modèle chargé avec succès sur %s", device)
        return model
    except Exception as e:
        logger.error("Échec du chargement du modèle: %s", str(e))
        st.error(f"Échec du chargement du modèle: {str(e)}")
        return None

@st.cache_resource(show_spinner="Chargement des données de référence...")
def load_data():
    try:
        with open("embeddings_from_drive.pkl", "rb") as f:
            data = pickle.load(f)
            
            if not data or len(data) == 0:
                logger.warning("Fichier pickle vide ou invalide")
                return [], [], np.array([])
            
            validated_data = []
            for d in data:
                if isinstance(d, dict) and "text" in d and "embedding" in d:
                    validated_data.append({
                        "text": d["text"],
                        "source": d.get("source", "Source inconnue"),
                        "embedding": d["embedding"]
                    })
                else:
                    logger.warning("Entrée de données invalide ignorée: %s", str(d)[:100])
            
            if not validated_data:
                raise ValueError("Aucune donnée valide après validation")
            
            texts = [d["text"] for d in validated_data]
            sources = [d["source"] for d in validated_data]
            embeddings = np.array([d["embedding"] for d in validated_data])
            
            logger.info("Données chargées avec succès: %d textes", len(texts))
            return texts, sources, embeddings
            
    except Exception as e:
        logger.error("Erreur critique de chargement des données: %s", str(e))
        st.error(f"Erreur de chargement des données: {str(e)}")
        return ["Erreur de chargement des données"], ["Système"], np.random.rand(1, 384)

model = load_model()
texts, sources, embeddings = load_data()

# ---------------------- Fonctions principales ----------------------
def validate_message(msg):
    """Valide la structure d'un message de chat"""
    if not isinstance(msg, dict):
        return False
    if "role" not in msg or "content" not in msg:
        return False
    if not isinstance(msg["content"], str):
        return False
    return True

def display_chat_history():
    """Affiche l'historique de chat avec validation robuste"""
    if "history" not in st.session_state:
        st.session_state.history = []
    
    valid_messages = []
    for msg in st.session_state.history:
        try:
            if not validate_message(msg):
                logger.warning("Message invalide ignoré: %s", str(msg)[:100])
                continue
                
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                st.markdown(f"<div class='user-msg'><strong>Vous:</strong><br>{content}</div>", unsafe_allow_html=True)
            elif role == "assistant":
                st.markdown(f"<div class='bot-msg'><strong>MedRegBot:</strong><br>{content}</div>", unsafe_allow_html=True)
            elif role == "loading":
                st.markdown(f"<div class='loading-msg'>{content}</div>", unsafe_allow_html=True)
            elif role == "error":
                st.markdown(f"<div class='error-msg'>{content}</div>", unsafe_allow_html=True)
            elif role == "warning":
                st.markdown(f"<div class='warning-msg'>{content}</div>", unsafe_allow_html=True)
                
            valid_messages.append(msg)
            
        except Exception as e:
            logger.error("Erreur d'affichage du message: %s", str(e), exc_info=True)
    
    # Mettre à jour l'historique avec seulement les messages valides
    st.session_state.history = valid_messages

def find_relevant_context(question, top_k=3):
    """Trouve le contexte pertinent pour une question donnée"""
    try:
        if model is None or embeddings.size == 0:
            return [], "Modèle ou embeddings non disponibles"
        
        question_embedding = model.encode([question])
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        context_list = []
        for i in top_indices:
            context_list.append({
                "text": texts[i],
                "source": sources[i],
                "score": float(similarities[i])
            })
        
        return context_list, "Success"
    
    except Exception as e:
        logger.error("Erreur dans find_relevant_context: %s", str(e), exc_info=True)
        return [], f"Erreur lors de la recherche de contexte: {str(e)}"

def format_context(context_list):
    """Formate le contexte pour l'affichage"""
    if not context_list:
        return "Aucun contexte pertinent trouvé"
    
    formatted = []
    for i, ctx in enumerate(context_list, 1):
        formatted.append(
            f"**Référence {i}** (Score: {ctx['score']:.2f}, Source: {ctx['source']})\n"
            f"{ctx['text']}\n"
        )
    return "\n\n".join(formatted)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
def query_mistral(prompt, context):
    """Interroge l'API Mistral avec gestion des erreurs"""
    try:
        API_TOKEN = st.secrets.get('HF_TOKEN', '')
        if not API_TOKEN:
            raise ValueError("Token API non configuré")
        
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
        
        system_prompt = """Tu es un expert en réglementation médicale européenne. 
        Réponds en français de manière claire et précise en t'appuyant strictement sur le contexte fourni.
        Si la réponse n'est pas dans le contexte, dis simplement que tu ne sais pas."""
        
        full_prompt = f"""<s>[INST] <<SYS>>{system_prompt}<</SYS>>
        
        Contexte: {context}
        
        Question: {prompt} [/INST]"""
        
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.3,
                    "do_sample": True,
                    "top_p": 0.9
                }
            },
            timeout=45
        )
        
        response.raise_for_status()
        return response
    
    except requests.exceptions.RequestException as e:
        logger.error("Erreur de connexion à l'API: %s", str(e))
        raise
    except Exception as e:
        logger.error("Erreur inattendue lors de l'appel API: %s", str(e))
        raise

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

# Initialisation de l'historique
if "history" not in st.session_state:
    st.session_state.history = [{
        "role": "assistant",
        "content": "Bonjour ! Je suis MedRegBot, votre assistant pour la réglementation des dispositifs médicaux. Comment puis-je vous aider ?",
        "timestamp": time.time()
    }]
    st.session_state.context = ""

# Affichage de l'historique
display_chat_history()

# Gestion de l'entrée utilisateur
user_input = st.chat_input("Posez votre question sur la réglementation médicale...")
if user_input:
    # Ajout du message utilisateur
    st.session_state.history.append({
        "role": "user",
        "content": user_input,
        "timestamp": time.time()
    })
    
    # Recherche de contexte
    loading_placeholder = st.empty()
    loading_placeholder.markdown("<div class='loading-msg'>🔍 Recherche dans les textes réglementaires...</div>", unsafe_allow_html=True)
    
    context_list, context_status = find_relevant_context(user_input)
    formatted_context = format_context(context_list)
    st.session_state.context = formatted_context
    
    if context_status != "Success":
        st.session_state.history.append({
            "role": "error",
            "content": f"Erreur lors de la recherche: {context_status}",
            "timestamp": time.time()
        })
    
    # Génération de la réponse
    loading_placeholder.markdown("<div class='loading-msg'>🧠 Analyse par l'expert IA...</div>", unsafe_allow_html=True)
    
    try:
        if st.secrets.get('HF_TOKEN'):
            response = query_mistral(user_input, formatted_context)
            
            if response.status_code == 200:
                try:
                    generated_text = response.json()[0]["generated_text"]
                    answer = generated_text.split("[/INST]")[-1].strip()
                    answer = answer.replace("<s>", "").replace("</s>", "")
                    
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": time.time(),
                        "context": context_list
                    })
                except (KeyError, IndexError) as e:
                    error_msg = f"Erreur de format de réponse de l'API: {str(e)}"
                    logger.error(error_msg)
                    st.session_state.history.append({
                        "role": "error",
                        "content": error_msg,
                        "timestamp": time.time()
                    })
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": f"Voici les références pertinentes:\n\n{formatted_context}",
                        "timestamp": time.time()
                    })
            else:
                error_msg = f"Erreur API (code {response.status_code})"
                logger.error("%s: %s", error_msg, response.text[:200])
                st.session_state.history.append({
                    "role": "error",
                    "content": error_msg,
                    "timestamp": time.time()
                })
                st.session_state.history.append({
                    "role": "assistant",
                    "content": f"Service IA temporairement indisponible. Voici les références:\n\n{formatted_context}",
                    "timestamp": time.time()
                })
        else:
            st.session_state.history.append({
                "role": "warning",
                "content": "Configuration API manquante - mode référence seulement",
                "timestamp": time.time()
            })
            st.session_state.history.append({
                "role": "assistant",
                "content": f"Voici les références pertinentes:\n\n{formatted_context}",
                "timestamp": time.time()
            })
    
    except Exception as e:
        error_msg = f"Erreur lors de la génération de réponse: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.session_state.history.append({
            "role": "error",
            "content": error_msg,
            "timestamp": time.time()
        })
        st.session_state.history.append({
            "role": "assistant",
            "content": f"Erreur temporaire. Voici les références:\n\n{formatted_context}",
            "timestamp": time.time()
        })
    
    finally:
        loading_placeholder.empty()
        st.rerun()

# Panel de débogage dans la sidebar
if st.sidebar.checkbox("Afficher les détails techniques"):
    st.sidebar.markdown("### Contexte technique")
    st.sidebar.write(f"Modèle chargé: {'Oui' if model else 'Non'}")
    st.sidebar.write(f"Nombre de textes chargés: {len(texts)}")
    st.sidebar.write(f"Taille des embeddings: {embeddings.shape if embeddings.size > 0 else 'Aucun'}")
    
    if st.session_state.get("context"):
        st.sidebar.markdown("### Dernier contexte utilisé")
        st.sidebar.text_area("Contexte", st.session_state.context, height=300)
