# COMPTE-RENDU DÉTAILLÉ — ATELIER PRATIQUE 01
**Machine Learning Appliqué · Licence 3 · DSI**
**Sujet : Appeler un modèle et l'exposer derrière une API (Guichet ML)**

---

## 1. Présentation générale du TP

### 1.1 Qu'est-ce qu'on essaie de faire globalement ?
Dans cet atelier, nous construisons la première brique logicielle d'une application d'orientation appelée **Guichet ML**. 
Le service des étudiants reçoit quotidiennement des messages rédigés en français, en anglais ou en arabe. Avant d'orienter l'étudiant vers le bon guichet, notre système doit détecter automatiquement la langue du texte.

**Le point essentiel à comprendre :**
Nous ne créons ni n'entraînons de modèle d'intelligence artificielle. Des modèles performants existent déjà sur le marché. Notre métier d'ingénieur/développeur consiste à **consommer** ce modèle à travers une **API REST** propre, sécurisée et fiable.

Notre mission est de faire en sorte que notre backend soit capable de :
1. Définir un **contrat strict** de données (entrée et sortie) qui ne dépend d'aucun fournisseur externe.
2. Isoler le fournisseur d'IA derrière une interface interchangeable (design pattern Adapter / Provider).
3. Utiliser un **fournisseur simulé (Mock)** pour développer, tester et valider l'application 100% hors ligne sans dépendre d'Internet, de clé API ou de quotas.
4. Gérer proprement les erreurs (requête invalide du client -> HTTP 422, panne du service distant -> HTTP 503) sans faire planter l'application et sans divulguer d'informations sensibles.

### 1.2 Avec quoi avons-nous travaillé ?
- **Python 3** : Le langage de programmation standard en Data & ML.
- **Environnement virtuel (`venv`)** : Pour isoler nos bibliothèques du Python système.
- **FastAPI** : Le framework web moderne en Python pour créer des API REST rapides et autodocumentées.
- **Pydantic (v2)** : La bibliothèque de validation des données et des schémas (définition des types et contraintes).
- **Uvicorn** : Le serveur ASGI qui fait tourner notre API FastAPI.
- **python-dotenv** : Pour lire les variables de configuration depuis un fichier `.env`.
- **Pytest & HTTPX** : Pour automatiser les tests unitaires et d'intégration hors ligne.
- **Git** : Pour le versionnement du code avec exclusion stricte des secrets (`.env`) et dossiers lourds (`.venv`).

---

## 2. Déroulement étape par étape

---

### Diagnostic préalable de l'environnement

#### Ce qu'on fait et pourquoi :
Avant de commencer, on vérifie que Python est bien installé sur notre machine et qu'il est accessible dans le terminal PowerShell.

#### Commande exécutée :
```powershell
py --version
```

#### Résultat obtenu :
```text
Python 3.14.2
```
> **Point de contrôle validé :** La version est bien 3.11 ou supérieure.

---

### Étape 1 — Créer le projet `guichet-ml`

#### Ce qu'on fait et pourquoi :
Nous créons une arborescence de dossiers propre pour organiser notre code (`app` pour le code source de l'API, `tests` pour les tests automatisés).
Nous créons ensuite un **environnement virtuel** (`.venv`) : cela garantit que nos bibliothèques sont installées dans le projet et ne polluent pas la machine.

#### Commandes exécutées dans PowerShell :
```powershell
mkdir guichet-ml
cd guichet-ml
mkdir app, tests
py -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Fichier `requirements.txt` :
Nous fixons les versions exactes des dépendances pour garantir que le projet est 100% reproductible par n'importe quel collègue ou enseignant :
```text
fastapi==0.141.1
uvicorn[standard]==0.46.0
pydantic==2.13.3
python-dotenv==1.2.2
python-multipart==0.0.26
httpx==0.28.1
pytest==9.1.1
```

#### Installation des dépendances :
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import fastapi, pydantic; print('environnement OK')"
```
*Sortie :* `environnement OK`

#### Protection des secrets (`.gitignore`, `.env.example`, `.env`) :
On ne met **jamais** de secrets (mots de passe, jetons API) dans Git.
1. **Fichier `.gitignore`** :
```text
.venv/
__pycache__/
.env
.pytest_cache/
```

2. **Fichier `.env.example`** (public, versionné dans Git, montre le modèle de variables sans valeurs secrètes) :
```ini
INFERENCE_MODE=mock
HF_TOKEN=
MODEL_ID=papluca/xlm-roberta-base-language-detection
REVIEW_THRESHOLD=0.60
```

3. **Fichier `.env`** (privé, local, non versionné dans Git) :
```ini
INFERENCE_MODE=mock
```

#### Contrôle Git :
```powershell
git init
git status
```
*Résultat :* Seuls `.env.example`, `.gitignore`, `README.md`, `app/`, `requirements.txt` et `tests/` apparaissent. **Ni `.env` ni `.venv` n'apparaissent dans Git.**

---

### Étape 2 — Écrire le contrat avant le code

#### Ce qu'on fait et pourquoi :
Dans une architecture professionnelle, on ne commence jamais par brancher le modèle. On commence par **définir le contrat d'interface (schéma)** que notre API s'engage à respecter.
Ainsi, si le fournisseur de modèle change ou modifie son format de réponse un jour, le reste de notre système ne casse pas car notre contrat reste identique.

#### Fichier `app/schemas.py` :
```python
from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


class LanguageRequest(BaseModel):
    text: str = Field(min_length=3, max_length=1000)


class LanguageResponse(BaseModel):
    provider: str
    model: str
    latency_ms: float
    language: str
    requires_review: bool
    predictions: list[Prediction]
```

#### Justification détaillée de chaque champ de `LanguageResponse` :
| Champ | Pourquoi il est là ? |
| :--- | :--- |
| `provider` | Savoir qui a réellement répondu : le fournisseur réel ou le composant simulé (`mock`). |
| `model` | Tracer quelle version du modèle a produit le résultat (auditabilité). |
| `latency_ms` | Mesurer le temps d'exécution en millisecondes pour analyser la performance. |
| `language` | La décision finale retenue (le code de langue ayant obtenu le meilleur score). |
| `requires_review` | **Le champ le plus important** : indique par `true` si la confiance du modèle est inférieure au seuil (`REVIEW_THRESHOLD`), signalant qu'un humain doit vérifier. |
| `predictions` | Liste détaillée de toutes les langues testées et de leurs scores de probabilité, utile pour le diagnostic. |

#### Fichier `app/config.py` :
Centralise la lecture de l'environnement :
```python
import os
from dotenv import load_dotenv

load_dotenv()

MODE = os.getenv("INFERENCE_MODE", "mock").lower()
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = os.getenv("MODEL_ID", "papluca/xlm-roberta-base-language-detection")
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.60"))
```

---

### Étape 3 — Le fournisseur simulé (`MockProvider`)

#### Ce qu'on fait et pourquoi :
Pour développer et tester sans connexion Internet, sans carte bancaire et sans jeton API payant, on crée un **fournisseur simulé (Mock)**. Ce composant applique des règles déterministes mais renvoie exactement les mêmes types d'objets (`Prediction`) qu'un vrai modèle.
Les deux classes (`MockProvider` et `HfProvider`) exposent exactement la même méthode : `detect_language(self, text: str) -> list[Prediction]`.

#### Fichier `app/providers.py` :
```python
from app.config import HF_TOKEN, MODEL_ID
from app.schemas import Prediction


class MockProvider:
    name = "mock"

    def detect_language(self, text: str) -> list[Prediction]:
        # Ceci n'est PAS un modèle : ce sont des règles simples,
        # qui produisent le même type de réponse qu'un vrai modèle.
        if any("\u0600" <= c <= "\u06ff" for c in text):
            return [Prediction(label="ar", score=0.97)]
        mots_fr = {"je", "mon", "ma", "compte", "bonjour", "pas", "plus"}
        if set(text.lower().replace(".", "").split()) & mots_fr:
            return [Prediction(label="fr", score=0.95),
                    Prediction(label="it", score=0.03)]
        return [Prediction(label="en", score=0.93),
                Prediction(label="de", score=0.04)]


class HfProvider:
    name = "huggingface"

    def __init__(self):
        from huggingface_hub import InferenceClient
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN absent")
        self.client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

    def detect_language(self, text: str) -> list[Prediction]:
        brut = self.client.text_classification(text, model=MODEL_ID, top_k=3)
        return [Prediction(label=i.label, score=i.score) for i in brut]


def get_provider(mode: str):
    if mode == "real":
        try:
            return HfProvider()
        except Exception:
            return MockProvider()   # repli automatique
    return MockProvider()
```

---

### Étape 4 — L'API FastAPI

#### Ce qu'on fait et pourquoi :
Nous assemblons le tout dans `app/main.py`. Nous exposons deux routes HTTP :
1. `GET /health` : Vérifier que l'API est en vie et voir quel mode (`mock` ou `real`) est activé.
2. `POST /detect-language` : Valide la requête avec Pydantic, mesure le temps d'exécution (`time.perf_counter()`), appelle le provider sélectionné, calcule la meilleure prédiction, compare avec le seuil `REVIEW_THRESHOLD`, et renvoie la réponse normalisée `LanguageResponse`.

#### Fichier `app/main.py` :
```python
import time

from fastapi import FastAPI, HTTPException

from app.config import MODE, MODEL_ID, REVIEW_THRESHOLD
from app.providers import get_provider
from app.schemas import LanguageRequest, LanguageResponse

app = FastAPI(title="Guichet ML", version="1.0.0")
provider = get_provider(MODE)


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE, "provider": provider.name}


@app.post("/detect-language", response_model=LanguageResponse)
def detect_language(payload: LanguageRequest):
    debut = time.perf_counter()
    try:
        predictions = provider.detect_language(payload.text)
    except Exception as exc:
        raise HTTPException(503, "Service d'inference indisponible") from exc
    meilleur = max(predictions, key=lambda p: p.score)
    return LanguageResponse(
        provider=provider.name,
        model=MODEL_ID,
        latency_ms=round((time.perf_counter() - debut) * 1000, 1),
        language=meilleur.label,
        requires_review=meilleur.score < REVIEW_THRESHOLD,
        predictions=predictions,
    )
```
*(Ne pas oublier le fichier vide `app/__init__.py` pour que Python reconnaisse le package).*

#### Démarrage du serveur :
```powershell
uvicorn app.main:app --reload
```
- Documentation Swagger interactive accessible à : `http://127.0.0.1:8000/docs`
- Route de santé : `http://127.0.0.1:8000/health`

#### Test manuel depuis PowerShell :
```powershell
$body = @{ text = "Je souhaite reinitialiser mon mot de passe." } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/detect-language" -ContentType "application/json" -Body $body
```

#### Réponse reçue :
```json
{
  "provider": "mock",
  "model": "papluca/xlm-roberta-base-language-detection",
  "latency_ms": 0.1,
  "language": "fr",
  "requires_review": false,
  "predictions": [
    {
      "label": "fr",
      "score": 0.95
    },
    {
      "label": "it",
      "score": 0.03
    }
  ]
}
```

---

### Étape 5 — Les tests automatisés et cas d'erreur

#### Ce qu'on fait et pourquoi :
Une application sans tests ne garantit rien. Nous écrivons des tests avec `pytest` et `TestClient` pour valider le statut de santé, la détection en français et en arabe, et le rejet immédiat d'une entrée invalide.

#### Fichier `tests/test_api.py` :
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_mode_mock():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["mode"] == "mock"


def test_detection_francais():
    r = client.post("/detect-language",
                    json={"text": "Je ne peux plus acceder a mon compte."})
    assert r.status_code == 200
    assert r.json()["language"] == "fr"


def test_detection_arabe():
    r = client.post("/detect-language",
                    json={"text": "لا أستطيع الدخول إلى حسابي"})
    assert r.status_code == 200
    assert r.json()["language"] == "ar"


def test_texte_trop_court_rejete():
    assert client.post("/detect-language", json={"text": "ab"}).status_code == 422
```

#### Commande d'exécution :
```powershell
pytest -q
```

#### Résultat obtenu :
```text
....                                                                     [100%]
4 passed in 0.43s
```
> **Point de contrôle validé :** Les 4 tests passent au vert en moins d'une demi-seconde hors ligne !

---

### Tableau d'analyse des cas d'erreur rencontrés :

| Cas testé | Requête envoyée | Code HTTP attendu | Explication détaillée |
| :--- | :--- | :---: | :--- |
| **Texte valide** | `{"text": "Je souhaite réinitialiser..."}` | **200 OK** | Entrée valide : le provider traite le texte et renvoie la réponse normalisée avec la latence calculée. |
| **Texte trop court** | `{"text": "ab"}` | **422 Unprocessable Entity** | Erreur client : Pydantic applique la règle `min_length=3` et refuse la requête avant d'appeler le modèle. |
| **Champ `text` manquant** | `{}` | **422 Unprocessable Entity** | Erreur client : le champ obligatoire `text` est absent du corps JSON. |
| **Fournisseur en panne** | Provider levant une exception (coupure réseau, timeout) | **503 Service Unavailable** | Erreur serveur/tiers : le backend intercepte l'erreur technique et renvoie un message générique `"Service d'inference indisponible"` sans faire crasher l'API ni divulguer d'informations sensibles. |

---

## 3. Réponses au Quiz de sortie

### Question 1 : Quelle est la différence entre entraîner un modèle et faire de l'inférence ?
> **Réponse :**
> - **L'entraînement** consiste à faire apprendre au modèle ses paramètres (poids) à partir d'un grand jeu de données. C'est une opération très longue, coûteuse et qui nécessite des GPU puissants.
> - **L'inférence** consiste simplement à utiliser un modèle déjà entraîné pour faire une prédiction sur une nouvelle donnée fournie en entrée. L'opération est rapide et peu gourmande par requête.

---

### Question 2 : Un score de 0,98 garantit-il que la prédiction est correcte ?
> **Réponse :**
> **Non.** Un score de 0,98 représente uniquement la certitude (confiance mathématique) du modèle statistique par rapport à ce qu'il a appris, mais ne garantit pas la vérité absolue. Le modèle peut se tromper avec un score très élevé s'il fait face à un cas hors distribution, un texte ambigu ou du bruit.

---

### Question 3 : Pourquoi l'API ne renvoie-t-elle pas directement la réponse du fournisseur ?
> **Réponse :**
> Pour respecter le principe de découplage et protéger le système : si le fournisseur change son format JSON, s'il renomme un champ, ou si nous décidons de changer de fournisseur (passer de Hugging Face à OpenAI ou à un modèle local), nos applications clientes ne casseront pas car notre contrat d'API (`LanguageResponse`) reste stable et sous notre contrôle.

---

### Question 4 : Pourquoi une clé d'API ne doit-elle jamais se trouver dans le code du navigateur ?
> **Réponse :**
> Tout code exécuté dans le navigateur (JavaScript) est public et inspectable par n'importe quel utilisateur via les outils de développement (F12). Une clé d'API placée dans le front-end peut être volée immédiatement, entraînant des dépenses financières imprévues ou des abus de quotas. Seul le serveur backend doit détenir la clé.

---

### Question 5 : Que doit faire l'application quand le meilleur score est inférieur au seuil ?
> **Réponse :**
> L'application doit positionner le champ booléen `requires_review = true` dans la réponse `LanguageResponse`. Cela indique aux couches métier et aux agents du service étudiant qu'une vérification humaine est nécessaire car l'IA n'est pas suffisamment confiante.

---

### Question 6 : À quoi sert exactement le mode MOCK, et que ne prouve-t-il pas ?
> **Réponse :**
> - **Ce à quoi il sert :** Il permet de tester et de valider toute l'intégration logicielle (validation Pydantic, routes HTTP, sérialisation JSON, gestion des erreurs) de manière rapide, reproductible et totalement hors ligne sans dépendre d'Internet ni d'un compte externe.
> - **Ce qu'il ne prouve pas :** Il ne prouve absolument rien sur la précision linguistique, la performance réelle ni la qualité des prédictions d'un vrai modèle de Machine Learning.

---

## 4. Fiche mémo des commandes et notions clés

| Objectif | Commande ou notion |
| :--- | :--- |
| **Créer l'environnement virtuel** | `py -m venv .venv` puis `.venv\Scripts\Activate.ps1` |
| **Installer les dépendances figées** | `python -m pip install -r requirements.txt` |
| **Lancer le serveur en direct** | `uvicorn app.main:app --reload` |
| **Lancer les tests unitaires** | `pytest -q` |
| **Interface Swagger automatique** | `http://127.0.0.1:8000/docs` |
| **Valider une chaîne (longueur)** | `Field(min_length=3, max_length=1000)` |
| **Valider une probabilité (score)** | `Field(ge=0, le=1)` |
| **Mesurer la latence** | `time.perf_counter()` |
| **Refuser une entrée invalide** | Code HTTP **422** (géré nativement par Pydantic) |
| **Signaler un tiers en panne** | `raise HTTPException(503, "Service d'inference indisponible")` |
| **Bascule de mode** | `INFERENCE_MODE=mock` ou `real` dans `.env` |
