import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from requests.exceptions import RequestException
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import logging
from huggingface_hub import login

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
/* [Vos styles CSS existants] */
.user-msg { background-color: #f0f2f6; [...] }
.bot-msg { background-color: #e6f0ff; [...] }
/* [Conservez tous vos styles existants] */
</style>
''', unsafe_allow_html=True)

# ---------------------- Initialisation ----------------------
@st.cache_resource(show_spinner="Chargement du modèle d'embedding...")
def load_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Modèle 768D
            device=device
        )
        logger.info(f"Modèle chargé sur {device}")
        return model
    except Exception as e:
        logger.error(f"Erreur de chargement du modèle : {str(e)}")
        st.error("Erreur de chargement du modèle AI")
        return None

@st.cache_resource(show_spinner="Chargement des données réglementaires...")
def load_data():
    """Charge les données depuis Hugging Face ou en local"""
    try:
        # Option 1 : Depuis Hugging Face
        from datasets import load_dataset
        dataset = load_dataset(
            "HAD-JER/medregbot-data",
            use_auth_token=st.secrets.get('HF_TOKEN', ''),
            download_mode="force_redownload"
        )
        texts = dataset['train']['text']
        sources = dataset['train']['source']
        embeddings = np.array(dataset['train']['embedding'])
        
        # Vérification des dimensions
        if embeddings.shape[1] != 768:
            raise ValueError(f"Dimension incorrecte des embeddings : {embeddings.shape[1]} au lieu de 768")
            
        return texts, sources, embeddings
        
    except Exception as e:
        logger.error(f"Erreur HF : {str(e)} - Tentative de chargement local...")
        
        # Option 2 : Fallback local
        try:
            df = pd.read_csv("data/medreg_texts.csv")
            model = load_model()
            df['embedding'] = df['text'].apply(lambda x: model.encode(x).tolist())
            return (
                df['text'].tolist(),
                df['source'].tolist(),
                np.array(df['embedding'].tolist())
            )
        except Exception as e:
            logger.critical(f"Erreur de chargement local : {str(e)}")
            return ["Données non disponibles"], ["Système"], np.random.rand(1, 768)

# ---------------------- Fonctions principales ----------------------
def find_relevant_context(question, top_k=3):
    """Trouve les textes pertinents avec scores de similarité"""
    try:
        model = load_model()
        question_embed = model.encode([question])
        similarities = cosine_similarity(question_embed, embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                "text": texts[i],
                "source": sources[i],
                "score": float(similarities[i])
            } for i in top_indices
        ]
    except Exception as e:
        logger.error(f"Erreur de recherche : {str(e)}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_llm(prompt, context):
    """Interroge le modèle Mistral avec gestion d'erreurs"""
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers={"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"},
            json={
                "inputs": f"""<s>[INST] <<SYS>>
                Tu es un expert en réglementation médicale UE. 
                Réponds en français en citant strictement ce contexte :
                {context}
                <</SYS>> {prompt} [/INST]""",
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.3
                }
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()[0]["generated_text"].split("[/INST]")[-1].strip()
    except Exception as e:
        logger.error(f"Erreur LLM : {str(e)}")
        raise

# ---------------------- Interface ----------------------
model = load_model()
texts, sources, embeddings = load_data()

# Initialisation de la session
if "history" not in st.session_state:
    st.session_state.history = [{
        "role": "assistant", 
        "content": "Bonjour ! Je suis MedRegBot, votre assistant pour la réglementation médicale. Posez-moi une question."
    }]

# Affichage de l'historique
for msg in st.session_state.history:
    st.markdown(
        f"""<div class='{"user" if msg["role"] == "user" else "bot"}-msg'>
            <strong>{'Vous' if msg['role'] == 'user' else 'MedRegBot'}:</strong><br>
            {msg['content']}
        </div>""",
        unsafe_allow_html=True
    )

# Gestion des requêtes
if user_input := st.chat_input("Votre question..."):
    st.session_state.history.append({"role": "user", "content": user_input})
    
    with st.spinner("🔍 Recherche en cours..."):
        try:
            # 1. Recherche contextuelle
            context = find_relevant_context(user_input)
            formatted_ctx = "\n\n".join(
                f"**Source {i+1}** ({c['score']:.2f}): {c['source']}\n{c['text']}"
                for i, c in enumerate(context)
            )
            
            # 2. Génération de réponse
            with st.spinner("🧠 Analyse..."):
                answer = query_llm(user_input, formatted_ctx)
                st.session_state.history.append({
                    "role": "assistant",
                    "content": answer,
                    "context": context
                })
                
        except Exception as e:
            st.error(f"Erreur : {str(e)}")
            st.session_state.history.append({
                "role": "assistant",
                "content": f"""Je ne peux pas répondre actuellement. 
                Voici des références pertinentes :\n\n{formatted_ctx}"""
            })
    
    st.rerun()

# Panel de debug
if st.sidebar.checkbox("Debug"):
    st.sidebar.write("### Statistiques")
    st.sidebar.write(f"- Textes chargés : {len(texts)}")
    st.sidebar.write(f"- Dernière question : {user_input[:50]}...")
    if st.session_state.history[-1].get("context"):
        st.sidebar.download_button(
            "Exporter les données",
            pd.DataFrame(st.session_state.history[-1]["context"]).to_csv().encode('utf-8'),
            "medregbot_context.csv"
        )
