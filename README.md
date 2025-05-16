# 📚 My Content – Système de Recommandation d’Articles

Ce projet est un MVP (Minimum Viable Product) de système de recommandation pour l’application **My Content**, une start-up visant à encourager la lecture grâce à des recommandations personnalisées d’articles.

---

## 🚀 Objectif du projet

> En tant qu’utilisateur de l’application, je veux recevoir une sélection de cinq articles adaptés à mes intérêts.

L’objectif est de concevoir une solution complète de recommandation incluant :

- Un moteur IA (filtrage collaboratif et par contenu)
- Une API déployée via **Azure Function** (architecture serverless)
- Une interface de démonstration via **Streamlit**
- Une structuration **industrialisable** via Git et un dépôt GitHub

---

## 📁 Arborescence du projet

```
├── notebooks/                 → Notebook d'exploration et modélisation
├── fonctions.py              → Fonctions de traitement, entraînement et recommandation
├── azure_function/
│   ├── model/                → Artefacts du modèle (ALS, mappings, matrices)
│   └── reco_api/             → Azure Function (code API de recommandation)
├── app/
│   └── app.py                → Interface utilisateur Streamlit
├── data/                     → Données sources (articles, clics)
├── README.md                 → Présentation du projet
└── requirements.txt          → Dépendances Python
```

---

## 🧠 Modèles testés

Trois approches de recommandation ont été explorées :

| Modèle               | Type                     | Description                                                             |
|----------------------|--------------------------|-------------------------------------------------------------------------|
| Content-Based        | Filtrage par contenu     | Basé sur les métadonnées des articles (`category_id`, `publisher_id`)  |
| SVD (Surprise)       | Collaborative filtering  | Recommandation via factorisation matricielle implicite                 |
| ALS (Implicit)       | Collaborative filtering  | Modèle de popularité implicite (librairie `implicit`)                  |

---

## 📊 Résultats des évaluations

- **SVD (Surprise)** : RMSE ≈ 0.0465, MAE ≈ 0.0234
- **ALS (Implicit)** : Precision@5 ≈ 0.0086, MAP@5 ≈ 0.0260, NDCG@5 ≈ 0.0284
- **Cold-start** : Gestion par fallback sur les articles populaires

---

## 🏗️ Architecture logicielle retenue

```text
[User ID] ──▶ [App Streamlit]
                 │
                 ▼
         [Azure Function API]
                 │
                 ▼
       [Modèle ALS + Mappings + Matrice CSR]
```

---

## 🧪 Lancer l’application en local

### 1. Prérequis

- Python 3.10+
- `pip install -r requirements.txt`

### 2. Lancer l’interface utilisateur

```bash
cd app
streamlit run app.py
```

### 3. Lancer l’API Azure en local (optionnel)

```bash
cd azure_function
func start
```

---

## ☁️ Déploiement Azure Function

```bash
func azure functionapp publish reco-p10 --python
```

Le modèle, les mappings et la matrice CSR sont stockés dans `azure_function/model/`.

---

## 🔁 Mise à jour des utilisateurs / articles

- Le système gère les **utilisateurs inconnus** en complétant avec les articles les plus populaires.
- Les **nouvelles interactions** peuvent être intégrées par **réentraînement périodique** ou **mise à jour des artefacts**.

---

## 🧩 Dépendances principales

- `pandas`, `numpy`, `scikit-learn`
- `surprise`, `implicit`
- `scipy`, `joblib`, `azure-functions`
- `streamlit`, `matplotlib`, `seaborn`

---

## 🧠 Auteur

Projet réalisé par **AnthonyJVID**, dans le cadre du parcours *Engineer IA* chez OpenClassrooms.

---

## 📄 Licence

Ce projet est librement réutilisable à des fins pédagogiques ;)
