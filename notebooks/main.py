from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Charger les ressources
als_model = joblib.load("model/als_model.pkl")
user_map, item_map, reverse_item_map = joblib.load("model/mappings.pkl")
csr_train = joblib.load("model/csr_train.pkl")

# Initialiser FastAPI
app = FastAPI()

# Modèle d'entrée JSON
class InputData(BaseModel):
    user_id: int
    top_n: int = 5

# Endpoint de test
@app.get("/")
def root():
    return {"message": "API Recommandation ALS prête."}

# Endpoint principal
@app.post("/recommend")
def recommend(data: InputData):
    user_id = data.user_id
    top_n = data.top_n

    # Vérifier si l'utilisateur est connu
    if user_id not in user_map:
        raise HTTPException(status_code=404, detail="Utilisateur inconnu")

    # Récupérer l'index utilisateur
    user_idx = user_map[user_id]
    user_items = csr_train[user_idx]

    try:
        # Recommandation principale
        recommended = als_model.recommend(
            user_idx, user_items, N=top_n, filter_already_liked_items=True
        )
        recommended_articles = [int(reverse_item_map[int(row[0])]) for row in recommended]

        # Fallback si pas assez de recommandations
        if len(recommended_articles) < top_n:
            nb_missing = top_n - len(recommended_articles)

            # Articles déjà vus
            seen_items = set(csr_train[user_idx].nonzero()[1])
            all_items_sorted = np.argsort(np.array(csr_train.sum(axis=0)).ravel())[::-1]  # articles populaires

            fallback = []
            for item_idx in all_items_sorted:
                article_id = int(reverse_item_map[item_idx])
                if item_idx not in seen_items and article_id not in recommended_articles:
                    fallback.append(article_id)
                if len(fallback) >= nb_missing:
                    break

            article_ids = recommended_articles + fallback
        else:
            article_ids = recommended_articles

        return {
            "user_id": user_id,
            "top_n": top_n,
            "recommendations": article_ids
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur recommandation : {str(e)}")
