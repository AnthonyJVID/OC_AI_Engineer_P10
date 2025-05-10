from azure.functions import HttpRequest, HttpResponse
import azure.functions as func
import json, joblib, scipy.sparse as sp
from pathlib import Path
from fonctions import recommander_implicit

# ─── Chargement des artefacts une seule fois ────────────────────────────
ROOT = Path(__file__).resolve().parent / "model"

als_model = joblib.load(ROOT / "als_model.pkl")
user_map, item_map, _ = joblib.load(ROOT / "mappings.pkl")
csr_train = sp.load_npz(ROOT / "csr_train.npz")
popular_items = joblib.load(ROOT / "popular_items.pkl")

# ─── Définition de l'HTTP trigger (5 recommandations garanties) ─────────
app = func.FunctionApp()

@app.function_name(name="reco_api")
@app.route(route="reco_api", auth_level=func.AuthLevel.ANONYMOUS)
def reco_api(req: HttpRequest) -> HttpResponse:
    try:
        user_id = int(req.params.get("user_id"))
    except (TypeError, ValueError):
        return HttpResponse("Passer ?user_id=<entier>", status_code=400)

    recs = recommander_implicit(user_id, None,
                                als_model, user_map, item_map, csr_train, 5)

    # Compléter si < 5
    if len(recs) < 5:
        recs += [a for a in popular_items if a not in recs][:5 - len(recs)]

    body = json.dumps({"user_id": user_id, "recommendations": recs})
    return HttpResponse(body, mimetype="application/json")