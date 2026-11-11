# Guichet ML

API de détection de langue pour le service des étudiants.

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Copier `.env.example` en `.env` et remplir les valeurs :

```bash
cp .env.example .env
```

## Lancer l'application

```bash
uvicorn app.main:app --reload
```

Puis ouvrir http://127.0.0.1:8000/docs

## Tests

```bash
pytest -q
```
