import streamlit as st
import requests

st.title("🎯 Recommandations d'articles (ALS)")

# API endpoint local
API_URL = "http://127.0.0.1:8000/recommend"

# Input utilisateur
user_id = st.number_input("Entrez un identifiant utilisateur :", min_value=0, value=1, step=1)
top_n = st.slider("Nombre de recommandations :", 1, 10, 5)

if st.button("Obtenir les recommandations"):
    try:
        response = requests.post(API_URL, json={"user_id": user_id, "top_n": top_n})
        if response.status_code == 200:
            result = response.json()
            st.success(f"Recommandations pour l'utilisateur {user_id} :")
            for i, article_id in enumerate(result["recommendations"], 1):
                st.write(f"{i}. Article ID : {article_id}")
        else:
            st.error(f"Erreur {response.status_code} : {response.json()['detail']}")
    except Exception as e:
        st.error(f"Erreur de connexion à l’API : {e}")