# fonctions.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, ndcg_at_k

def construire_matrice_similarite(df):
    """
    Construit une matrice de similarité entre articles à partir de leurs métadonnées.
    """
    df["article_features"] = (
        "cat_" + df["category_id"].astype(str) + " " +
        "pub_" + df["publisher_id"].astype(str)
    )

    df_articles = df.drop_duplicates(subset="article_id")[["article_id", "article_features"]].copy()

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_articles["article_features"])
    similarity_matrix = cosine_similarity(X)

    return df_articles.reset_index(drop=True), similarity_matrix

def recommender_content_based(user_id, df_clicks, df_articles, similarity_matrix, top_n=5):
    """
    Recommande des articles similaires à ceux déjà lus par un utilisateur.
    """
    clicked = df_clicks[df_clicks["user_id"] == user_id]["click_article_id"].unique()
    if len(clicked) == 0:
        return []

    indices_clicked = df_articles[df_articles["article_id"].isin(clicked)].index
    mean_similarity = similarity_matrix[indices_clicked].mean(axis=0)

    articles_all = df_articles["article_id"].values
    mask = np.isin(articles_all, clicked, invert=True)

    recommended_indices = mean_similarity[mask].argsort()[::-1][:top_n]
    recommended_ids = articles_all[mask][recommended_indices]

    return recommended_ids.tolist()

def preparer_dataset_surprise(df):
    """
    Prépare un dataset Surprise à partir du DataFrame des clics.
    """
    df_ratings = df[["user_id", "click_article_id"]].copy()
    df_ratings["rating"] = 1  # Clic = feedback implicite

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df_ratings, reader)

    return data

def entrainer_modele_surprise(data):
    """
    Entraîne un modèle SVD de Surprise sur les données.
    """
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD()
    model.fit(trainset)
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    return model, trainset

def recommender_surprise(user_id, model, df, top_n=5):
    """
    Recommande les top N articles à un utilisateur avec le modèle SVD.
    """
    all_articles = df["click_article_id"].unique()
    seen = df[df["user_id"] == user_id]["click_article_id"].unique()
    unseen = [aid for aid in all_articles if aid not in seen]

    predictions = [(aid, model.predict(user_id, aid).est) for aid in unseen]
    predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)

    top_articles = [int(aid) for aid, _ in predictions_sorted[:top_n]]
    return top_articles

def preparer_matrices_implicit(df, user_col="user_id", item_col="click_article_id", value_col="popularity_indicator"):
    """
    Prépare les matrices CSR pour entraînement avec implicit.
    """
    users = df[user_col].unique()
    items = df[item_col].unique()

    user_map = {u: i for i, u in enumerate(users)}
    item_map = {i: j for j, i in enumerate(items)}

    df["user_idx"] = df[user_col].map(user_map)
    df["item_idx"] = df[item_col].map(item_map)

    shape = (len(users), len(items))
    matrix = csr_matrix((df[value_col], (df["user_idx"], df["item_idx"])), shape=shape)

    return matrix, user_map, item_map, df

def entrainer_modele_als(csr_interaction_matrix):
    """
    Entraîne un modèle ALS de la librairie implicit.
    """
    model = AlternatingLeastSquares(factors=32, regularization=0.05, iterations=20)
    model.fit(csr_interaction_matrix)
    return model

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

def evaluer_modele_implicit(model, csr_train, csr_test, k=5):
    """
    Évalue un modèle `implicit` avec les métriques standard pour les données implicites.
    """
    precision = precision_at_k(model, csr_train, csr_test, K=k, num_threads=1)
    map_score = mean_average_precision_at_k(model, csr_train, csr_test, K=k, num_threads=1)
    ndcg = ndcg_at_k(model, csr_train, csr_test, K=k, num_threads=1)

    print(f"📊 Évaluation ALS - Top {k}")
    print(f"Precision@{k} : {precision:.4f}")
    print(f"MAP@{k}       : {map_score:.4f}")
    print(f"NDCG@{k}      : {ndcg:.4f}")

def evaluer_modele_als(df, k=5, test_size=0.2, random_state=77):
    """
    Évalue un modèle ALS avec des mappings cohérents entre train et test,
    compatible avec implicit.evaluation.*
    """

    from sklearn.model_selection import train_test_split
    from scipy.sparse import csr_matrix

    # Créer mapping global
    user_ids = df["user_id"].unique()
    item_ids = df["click_article_id"].unique()
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {i: j for j, i in enumerate(item_ids)}
    shape = (len(user_map), len(item_map))

    # Split
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train = df_train.copy()
    df_test = df_test[df_test["user_id"].isin(user_map)].copy()

    # Ajouter index
    df_train["user_idx"] = df_train["user_id"].map(user_map)
    df_train["item_idx"] = df_train["click_article_id"].map(item_map)
    df_test["user_idx"] = df_test["user_id"].map(user_map)
    df_test["item_idx"] = df_test["click_article_id"].map(item_map)

    # CSR train
    csr_train = csr_matrix(
        (df_train["popularity_indicator"].astype(np.float32),
        (df_train["user_idx"], df_train["item_idx"])),
        shape=shape
    )

    # CSR test : on remplit une matrice vide, ligne par ligne
    test_rows = df_test["user_idx"].astype(int).values
    test_cols = df_test["item_idx"].astype(int).values
    test_data = df_test["popularity_indicator"].astype(np.float32).values

    csr_test = csr_matrix((test_data, (test_rows, test_cols)), shape=shape)

    # Entraînement
    model = entrainer_modele_als(csr_train)

    # Liste des user_idx valides (présents dans le test ET non vides dans le train)
    valid_users = np.intersect1d(
        np.unique(csr_train.nonzero()[0]),
        np.unique(csr_test.nonzero()[0])
    )

    # Ne garde que ces utilisateurs pour évaluation (index 2D + reconversion en CSR)
    csr_train_eval = csr_train[valid_users, :].tocsr()
    csr_test_eval  = csr_test[valid_users, :].tocsr()


    # Évaluation avec implicit (sur les users valides uniquement)
    precision = precision_at_k(model, csr_train_eval, csr_test_eval, K=k, num_threads=1)
    map_score = mean_average_precision_at_k(model, csr_train_eval, csr_test_eval, K=k, num_threads=1)
    ndcg = ndcg_at_k(model, csr_train_eval, csr_test_eval, K=k, num_threads=1)

    print(f"📊 Évaluation ALS - Top {k} (sur {len(valid_users)} utilisateurs)")
    print(f"Precision@{k} : {precision:.4f}")
    print(f"MAP@{k}       : {map_score:.4f}")
    print(f"NDCG@{k}      : {ndcg:.4f}")

import numpy as np

def evaluer_modele_als_manuel(model, df_train, df_test, user_map, item_map, csr_train, k=5):
    """
    Évalue un modèle ALS en calculant manuellement Precision@k, MAP@k et NDCG@k.
    """
    # mapping inverse pour retrouver les article_id
    reverse_item_map = {v: k for k, v in item_map.items()}

    precisions = []
    APs = []
    NDCGs = []

    # itérer sur chaque user présent dans le test
    for user_id, user_idx in user_map.items():
        test_items = df_test.loc[df_test.user_id == user_id, "click_article_id"].unique()
        if len(test_items) == 0:
            continue

        # recommander k articles
        rec_idxs, _ = model.recommend(user_idx, csr_train[user_idx], N=k, filter_already_liked_items=True)
        recs = [reverse_item_map[i] for i in rec_idxs]

        # Precision@k
        hits = [1 if a in test_items else 0 for a in recs]
        precision = sum(hits) / k
        precisions.append(precision)

        # Average Precision
        cum_hits = 0
        AP = 0.0
        for i, h in enumerate(hits, start=1):
            if h:
                cum_hits += 1
                AP += cum_hits / i
        APs.append(AP / cum_hits if cum_hits > 0 else 0.0)

        # NDCG@k
        # DCG
        dcg = hits[0]
        for i, h in enumerate(hits[1:], start=2):
            dcg += h / np.log2(i)
        # IDCG = sum_{i=1..min(len(test_items),k)} 1/log2(i+1)
        ideal_gains = [1] * min(len(test_items), k)
        idcg = ideal_gains[0] if ideal_gains else 0.0
        for i, _ in enumerate(ideal_gains[1:], start=2):
            idcg += 1 / np.log2(i)
        NDCGs.append(dcg / idcg if idcg > 0 else 0.0)

    # Moyennes
    print(f"📊 Évaluation manuel ALS - Top {k}")
    print(f"Precision@{k} : {np.mean(precisions):.4f}")
    print(f"MAP@{k}       : {np.mean(APs):.4f}")
    print(f"NDCG@{k}      : {np.mean(NDCGs):.4f}")

def recommander_similaire_article(article_id, df_articles, similarity_matrix, top_n=5):
    """
    Renvoie les top_n articles les plus similaires à l'article donné.
    - article_id : ID de l'article de référence
    - df_articles : DataFrame avec colonnes ['article_id', ...]
    - similarity_matrix : matrice (n_articles × n_articles) de similarité cosinus
    """
    # Vérifier que l'article existe
    if article_id not in df_articles["article_id"].values:
        return []

    # Index de l'article dans df_articles
    idx = df_articles.index[df_articles["article_id"] == article_id][0]

    # Liste (index, score)
    sim_scores = list(enumerate(similarity_matrix[idx]))
    # Trier par score décroissant et exclure l'article lui-même
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [ (i,score) for i,score in sim_scores if i != idx ]

    # Récupérer les top_n indices
    top_idxs = [ i for i,_ in sim_scores[:top_n] ]

    # Retourner les article_id correspondants
    return df_articles.loc[top_idxs, "article_id"].tolist()

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_item_similarity_svd(model_svd, trainset):
    """
    Construit la liste des article_ids et la matrice de similarité cosine
    sur les embeddings latents produits par le modèle SVD (Surprise).
    """
    # embeddings items (shape = n_items × n_factors)
    emb = model_svd.qi
    # raw IDs : trainset.all_items() renvoie les inner_iid
    inner_ids = trainset.all_items()
    raw_ids = [int(trainset.to_raw_iid(iid)) for iid in inner_ids]
    # similarité cosine
    sim_mat = cosine_similarity(emb)
    return raw_ids, sim_mat

def recommender_latent_item_svd(article_id, raw_ids, sim_mat, top_n=5):
    """
    Recommande les top_n articles similaires à article_id
    en utilisant la matrice de similarité issue du SVD.
    """
    if article_id not in raw_ids:
        return []
    idx = raw_ids.index(article_id)
    scores = [(i, score) for i, score in enumerate(sim_mat[idx]) if i != idx]
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [ raw_ids[i] for i,_ in top ]

def build_item_similarity_als(model_als, item_map):
    """
    Construit la liste des article_ids et la matrice de similarité cosine
    sur les embeddings latents produits par le modèle ALS (implicit).
    """
    # mapping inverse idx → article_id
    inv_map = {v: k for k, v in item_map.items()}
    # embeddings items (shape = n_items × factors)
    emb = model_als.item_factors
    # reconstituer la liste d'IDs dans l'ordre des facteurs
    ids = [inv_map[i] for i in range(emb.shape[0])]
    # calcul de la similarité cosine
    sim_mat = cosine_similarity(emb)
    return ids, sim_mat

def recommender_latent_item_als(article_id, ids, sim_mat, top_n=5):
    """
    Recommande les top_n articles similaires à article_id
    en utilisant la matrice de similarité issue de l'ALS.
    """
    if article_id not in ids:
        return []
    idx = ids.index(article_id)
    scores = [(i, score) for i, score in enumerate(sim_mat[idx]) if i != idx]
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [ ids[i] for i,_ in top ]

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_item_similarity_hybrid(df_articles, content_sim, cf_ids, cf_sim, alpha=0.5):
    """
    Prépare les structures pour un recommender hybride :
    - df_articles   : DataFrame avec 'article_id' dans l’ordre de content_sim
    - content_sim   : matrice (n×n) de similarité content-based
    - cf_ids        : liste d'article_id dans l’ordre de cf_sim
    - cf_sim        : matrice (n×n) de similarité collaborative
    - alpha         : poids donné au content-based (0 ≤ alpha ≤ 1)
    Renvoie :
    - ids_common    : liste d'article_id considérés (intersection des deux)
    - hybrid_sim    : matrice (m×m) de similarité hybride
    """
    # liste content et collaborative
    content_ids = df_articles["article_id"].tolist()
    # on ne garde que les IDs communs et leurs indices
    ids_common = [i for i in content_ids if i in cf_ids]
    idx_cont = {aid: content_ids.index(aid) for aid in ids_common}
    idx_cf   = {aid: cf_ids.index(aid)       for aid in ids_common}

    m = len(ids_common)
    hybrid_sim = np.zeros((m, m), dtype=float)

    # calcul score hybride
    for i, aid_i in enumerate(ids_common):
        for j, aid_j in enumerate(ids_common):
            c = content_sim[idx_cont[aid_i], idx_cont[aid_j]]
            f = cf_sim   [idx_cf[aid_i],    idx_cf[aid_j]   ]
            hybrid_sim[i, j] = alpha * c + (1 - alpha) * f

    return ids_common, hybrid_sim

def recommender_hybrid_article(article_id, ids_common, hybrid_sim, top_n=5):
    """
    Pour un article_id de référence, renvoie les top_n les plus proches
    selon la matrice híbride hybrid_sim indexée par ids_common.
    """
    if article_id not in ids_common:
        return []
    i0 = ids_common.index(article_id)
    scores = [(j, hybrid_sim[i0, j]) for j in range(len(ids_common)) if j != i0]
    top   = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [ids_common[j] for j,_ in top]

import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy

def split_temporal(df, ts_col="created_at_ts", cutoff=None, quantile=0.8):
    """
    Sépare df en df_train / df_test selon created_at_ts (ms).
    Si cutoff=None, utilise le quantile (par défaut 0.8).
    """
    df = df.copy()
    df["created_at"] = pd.to_datetime(df[ts_col], unit="ms")
    if cutoff is None:
        cutoff = df["created_at"].quantile(quantile)
    df_train = df[df["created_at"] <= cutoff].copy()
    df_test  = df[df["created_at"]  > cutoff].copy()
    return df_train, df_test

def train_test_svd_temporal(df, ts_col="created_at_ts", cutoff=None):
    """
    Entraîne un SVD sur les interactions ≤ cutoff et évalue sur > cutoff.
    Retourne (model, trainset, testset, {"rmse":…, "mae":…}).
    """
    # 1) split
    df_train, df_test = split_temporal(df, ts_col=ts_col, cutoff=cutoff)
    # 2) préparer le trainset Surprise
    df_tr = df_train[["user_id", "click_article_id"]].assign(rating=1)
    reader = Reader(rating_scale=(0,1))
    data_train = Dataset.load_from_df(df_tr, reader)
    trainset = data_train.build_full_trainset()
    # 3) testset manuel
    testset = [(u,i,1) for u,i in zip(df_test.user_id, df_test.click_article_id)]
    # 4) entraînement & éval
    model = SVD()
    model.fit(trainset)
    preds = model.test(testset)
    return model, trainset, testset, {
    "rmse": accuracy.rmse(preds, verbose=False),
    "mae":  accuracy.mae (preds, verbose=False)
    }

def split_leave_one_out(df, ts_col="created_at_ts"):
    """
    Pour chaque pair (user, item), on garde le dernier click en test,
    le reste en train.
    """
    df = df.copy()
    df["created_at"] = pd.to_datetime(df[ts_col], unit="ms")
    # on trie puis on marque le dernier de chaque utilisateur
    df = df.sort_values(["user_id", "created_at"])
    df["rank"] = df.groupby("user_id").cumcount(ascending=False)
    df_test  = df[df["rank"] == 0].drop(columns="rank")
    df_train = df[df["rank"] >  0].drop(columns="rank")
    return df_train, df_test

def top_k_with_fallback(user_id, model, user_map, item_map, csr_train,
                        popular_items, k=5):
    """Renvoie toujours k articles, en complétant par les plus populaires."""
    recs = recommander_implicit(user_id, None, model,
                                user_map, item_map, csr_train, top_n=k)
    if len(recs) < k:
        recs += [aid for aid in popular_items if aid not in recs][:k-len(recs)]
    return recs