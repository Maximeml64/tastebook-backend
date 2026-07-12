# Tastebook Backend — Stack figée

> Standard global : `../CLAUDE.md`. Ce fichier ne liste que les contraintes propres à ce backend.

## Nature

Backend Python **FastAPI** déployé sur Railway, qui sert l'app mobile Tastebook (`../Tastebook`). **Pas une app React Native** — le standard mobile dans `../STANDARD_NEW_APPS.md` ne s'applique pas ici.

## Stack en prod (NE PAS migrer sans demande explicite)

- **Framework** : FastAPI
- **Runtime** : Python (cf. `requirements.txt`)
- **Déploiement** : Railway (config dans `Procfile`)
- **Point d'entrée** : `main.py`

## Règles de modification

- **Pas de migration de framework** (Flask, Django, etc.) sans demande explicite.
- **Pas d'introduction d'ORM** (SQLAlchemy, Tortoise, etc.) sans demande explicite.
- Toute modif touchant les routes appelées par l'app Tastebook demande de vérifier la compat côté mobile.
- **Secrets et clés API jamais commités** : `.env` local, variables Railway en prod.
