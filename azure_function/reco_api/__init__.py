import json, joblib, scipy.sparse as sp
from pathlib import Path
from azure.functions import HttpRequest, HttpResponse

def recommander_implicit(user_id, interaction_df, model, user_map, item_map, interaction_matrix, top_n=5):
    """
    Recommande les top_n articles à un utilisateur avec le modèle ALS.
    """
    if user_id not in user_map:
        return []

    user_idx = user_map[user_id]
    reverse_item_map = {v: k for k, v in item_map.items()}

    user_items = interaction_matrix[user_idx]

    # Recommandation (structure renvoyée : tableau NumPy, pas forcément une liste de tuples)
    recommended = model.recommend(user_idx, user_items, N=top_n, filter_already_liked_items=True)

    # Extraction manuelle
    results = []
    for row in recommended:
        item_id = int(row[0])
        results.append(reverse_item_map[item_id])

    return results

# ─── Chargement des artefacts (fait une seule fois) ────────────────
ROOT = Path(__file__).resolve().parent.parent / "model"
als_model     = joblib.load(ROOT / "als_model.pkl")
user_map, item_map, _ = joblib.load(ROOT / "mappings.pkl")
csr_train     = sp.load_npz(ROOT / "csr_train.npz")
popular_items = joblib.load(ROOT / "popular_items.pkl")

def main(req: HttpRequest) -> HttpResponse:
    try:
        user_id = int(req.params.get("user_id"))
    except (TypeError, ValueError):
        return HttpResponse(
            "Paramètre manquant ou invalide : ?user_id=<entier>",
            status_code=400
        )

    # 1) Génère vos recommandations (remplacez par votre fonction)
    recs = recommander_implicit(
        user_id, None,
        als_model, user_map, item_map,
        csr_train,
        top_n=5
    )
    # 2) numpy.int64 → int
    recs = [int(r) for r in recs]
    # 3) Complément si <5
    if len(recs) < 5:
        recs += [int(a) for a in popular_items if a not in recs][:5 - len(recs)]

    body = json.dumps({"user_id": user_id, "recommendations": recs})
    return HttpResponse(body, mimetype="application/json")
