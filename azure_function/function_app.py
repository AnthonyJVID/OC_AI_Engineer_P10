from azure.functions import HttpRequest, HttpResponse
import azure.functions as func

import json
import joblib
import scipy.sparse as sp
from pathlib import Path

from fonctions import recommander_implicit   # ton module utils

# ────────────────────────────────────────────────────────────────
# Chargement des artefacts (fait une seule fois au cold‑start)
# ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent / "model"

als_model               = joblib.load(ROOT / "als_model.pkl")
user_map, item_map, _   = joblib.load(ROOT / "mappings.pkl")
csr_train               = sp.load_npz(ROOT / "csr_train.npz")
popular_items           = joblib.load(ROOT / "popular_items.pkl")

# ────────────────────────────────────────────────────────────────
# Définition de la Function App et de la route HTTP
# ────────────────────────────────────────────────────────────────
app = func.FunctionApp()

@app.function_name(name="reco_api")
@app.route(route="reco_api", auth_level=func.AuthLevel.ANONYMOUS)
def reco_api(req: HttpRequest) -> HttpResponse:
    """
    URL d’appel :
        GET /api/reco_api?user_id=<entier>
    Réponse : {"user_id": 42, "recommendations": [id1, id2, … id5]}
    """
    # 1) Lecture et validation du paramètre
    try:
        user_id = int(req.params.get("user_id"))
    except (TypeError, ValueError):
        return HttpResponse(
            "Paramètre manquant ou invalide : passer ?user_id=<entier>",
            status_code=400
        )

    # 2) Recommandations ALS (peut renvoyer < 5 articles si l’historique est pauvre)
    recs = recommander_implicit(
        user_id,                       # utilisateur
        None,                          # df_interactions pas nécessaire ici
        als_model, user_map, item_map,
        csr_train,
        top_n=5
    )

    # 3) Conversion numpy.int64 → int pour la sérialisation JSON
    recs = [int(a) for a in recs]

    # 4) Complément avec les articles populaires si besoin
    if len(recs) < 5:
        recs += [int(a) for a in popular_items if a not in recs][:5 - len(recs)]

    # 5) Réponse JSON
    body = json.dumps({"user_id": int(user_id), "recommendations": recs})
    return HttpResponse(body, mimetype="application/json")