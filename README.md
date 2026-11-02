# DataTalk — interroger SQLite et MongoDB en langage naturel

## Objectif

DataTalk est une plateforme de *Business Intelligence conversationnelle* : l'utilisateur pose une question en français,
sans connaître SQL ni MongoDB, et obtient **une réponse rédigée en langage naturel**. Les données peuvent se trouver dans
une base SQLite, dans une base MongoDB, ou dans les deux à la fois (question « hybride »). La requête utilisée est
toujours affichée pour que le résultat reste vérifiable.

## Méthode

Un pipeline d'agents (LangGraph) traite chaque question :

```
question ─► classifieur ─┬─► agent SQL ────────────────┐
                         ├─► agent MongoDB ────────────┤─► rédaction de la réponse
                         └─► planificateur hybride ─┐  │   (en français)
                              étape 1 ─► étape 2 ───┘──┘
```

- **Classifieur** : décide si la question concerne SQL, MongoDB ou les deux.
- **Agents SQL et MongoDB** : un LLM écrit la requête à partir du schéma (types et valeurs exactes des colonnes) et des
  règles métier retrouvées par un **RAG** (embeddings). La requête est **validée avant d'être exécutée**, puis corrigée
  automatiquement en cas de rejet (3 tentatives) :
  - lecture seule uniquement (aucun `INSERT`, `DELETE`, `CREATE`, `$out`, `$merge`…) ;
  - SQL compilé par SQLite (`EXPLAIN`) : tables et colonnes inventées, syntaxe ; champs MongoDB vérifiés ;
  - vérification sémantique par un LLM : la requête répond-elle à la question ?
- **Question hybride** : le planificateur la **découpe en deux sous-questions dépendantes**. La première source retourne
  les identifiants des entités concernées ; le code les injecte dans la requête de la seconde source (le LLM ne les recopie
  jamais). Le lien entre les deux bases est une **correspondance** (ex. `customers.customer_id ↔ support_tickets.customer_id`),
  **détectée automatiquement** par la ressemblance des noms et le recoupement des valeurs, et utilisée seulement si sa
  confiance est suffisante (0,8). Elle peut être confirmée ou rejetée dans l'interface.
- **Modèles** : Claude par défaut. En option, des **modèles open source** écrivent les requêtes (SQLCoder pour le SQL,
  Qwen2.5-Coder pour MongoDB), hébergés sur Google Colab (étape 7) ; Claude garde alors la classification, le plan hybride, les
  validations et la réponse finale.

## Utilisation

### 1. Prérequis
Python 3.10 ou plus récent, Node.js, une clé API Claude. MongoDB n'est nécessaire que pour se connecter à une base
MongoDB existante (un export JSON peut aussi être importé sans serveur).

### 2. Installation
```bash
git clone https://github.com/Emmanuella-05/memoire_2026.git
cd memoire_2026/datatalk
git checkout second

python -m venv venv
source venv/Scripts/activate        # Windows (Git Bash) ; Linux/macOS : source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
```bash
cp .env.example .env
```
Ouvrir `.env` et renseigner au minimum `CLAUDE_API_KEY`. Ne jamais commiter ce fichier.

### 4. Lancer le projet
Deux terminaux, ouverts dans `datatalk/` :
```bash
# Terminal 1 — backend (environnement virtuel activé)
python -m uvicorn backend.main:app --reload       # http://localhost:8000/docs

# Terminal 2 — interface
cd frontend
npm install
npm run dev                                       # http://localhost:5173
```
Au premier lancement, un petit modèle d'embeddings (≈ 90 Mo) est téléchargé : attendre `Application startup complete`.

### 5. Charger les données
Dans l'interface, section **« Configurer les données »** :
1. **Base SQLite** : importer un fichier `.db`.
2. **MongoDB** : soit *Se connecter à MongoDB* (URI `mongodb://localhost:27017` et nom de la base, accès en lecture
   seule), soit importer un export JSON `{ "collection": [ {...} ] }`.
3. **Règles métier** (facultatif) : un fichier de règles, par exemple la définition d'un « client actif ».
4. Cliquer sur **Réanalyser** : le système documente les bases et détecte les correspondances.
5. Vérifier les correspondances proposées : confirmer ou rejeter celles qui sont douteuses.

### 6. Poser une question
Saisir une question en français, par exemple :
- SQL : *« Combien de clients par ville ? »*
- MongoDB : *« Quels tickets concernent un retard de livraison ? »*
- Hybride : *« Quelles villes comptent le plus de clients ayant signalé un produit endommagé ? »*

La réponse s'affiche en langage naturel. Le bloc **Traçabilité** montre la source choisie, les sous-questions du plan
hybride, les requêtes exécutées et le nombre de tentatives.

### 7. (Facultatif) Modèles open source sur Google Colab

Par défaut, Claude écrit les requêtes. Pour que **SQLCoder-7B** (SQL) et **Qwen2.5-Coder-7B** (MongoDB) les écrivent
à sa place, il faut les faire tourner sur un GPU : un PC de 8 Go de RAM sans GPU ne les supporte pas. Le projet reste
sur le PC et appelle les modèles à distance. Claude garde la classification, le plan hybride, les validations et la
réponse finale.

**Les trois clés à ne pas confondre**

| Nom | D'où elle vient | Où la mettre |
|---|---|---|
| `NGROK_AUTHTOKEN` | ton compte gratuit sur ngrok.com (*Your Authtoken*) | Colab uniquement |
| `LLM_API_KEY` | **tu l'inventes** : une suite longue et aléatoire, qui protège l'URL publique | Colab **et** `.env` du PC (même valeur) |
| `CLAUDE_API_KEY` | console.anthropic.com | `.env` du PC uniquement |

Pour générer `LLM_API_KEY` : `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

**Sur Colab (à refaire à chaque session)**
1. Ouvrir un nouveau notebook sur colab.research.google.com, puis *Exécution → Modifier le type d'exécution → GPU T4*.
2. Dans le panneau **Fichiers** (icône dossier, à gauche), importer `datatalk/colab/llm_server.py` depuis ce dépôt.
   Vérifier avec une cellule `!ls /content` que le fichier est bien là.
3. Exécuter ces trois cellules, l'une après l'autre :
```python
# Cellule 1 : dépendances
!pip install -q fastapi uvicorn pyngrok accelerate bitsandbytes
```
```python
# Cellule 2 : clés
import os
os.environ["NGROK_AUTHTOKEN"] = "TON_JETON_NGROK"
os.environ["LLM_API_KEY"] = "TA_CLE_INVENTEE"
```
```python
# Cellule 3 : démarrage (reste « en cours » : c'est normal, ne pas l'arrêter)
!python /content/llm_server.py
```
4. Attendre : le premier démarrage télécharge les modèles (plusieurs minutes). Le serveur affiche
   `URL publique : https://xxxx.ngrok-free.app`, puis `sqlcoder-7b-2 prêt` et `qwen2.5-coder-7b prêt`. Ne pas poser de
   question avant ces deux lignes.

**Sur le PC**
1. Dans `datatalk/.env`, renseigner (sans guillemets) :
```
LLM_BASE_URL=https://xxxx.ngrok-free.app
LLM_API_KEY=TA_CLE_INVENTEE
```
2. Redémarrer le backend (Ctrl+C, puis relancer la commande de l'étape 4).
3. Vérifier la connexion : ouvrir `http://localhost:8000/docs`, lancer `POST /llm/test`, on attend `"ok": true`.
4. Poser une question : la ligne d'état de l'interface indique « requêtes générées par des modèles ouverts », et
   « Traçabilité » montre la question traduite en anglais pour les modèles.

**À savoir**
- L'URL ngrok gratuite **change à chaque session** : mettre à jour `LLM_BASE_URL` dans `.env` et redémarrer le backend.
- Colab gratuit coupe la session après un temps d'inactivité : relancer les cellules 2 et 3.
- Chaque question fait plusieurs appels de modèles : compter quelques secondes à quelques dizaines de secondes.
- Revenir à Claude sans rien supprimer : `DATATALK_GENERATOR=claude` dans `.env` (ou vider `LLM_BASE_URL`), puis
  redémarrer le backend. `DATATALK_FIX_WITH=claude` fait corriger les requêtes ratées par Claude plutôt que par
  le modèle ouvert.

**En cas de problème**
- *« No such file or directory : llm_server.py »* : le fichier n'est pas importé dans Colab (étape 2).
- *Erreur ngrok « invalid endpoint configuration »* : exécuter `!pip install -q -U pyngrok`, puis relancer la cellule 3.
- *« Le serveur de modèles ouverts refuse la clé »* : `LLM_API_KEY` diffère entre Colab et `.env`.
- *« HTTP 500 »* : la cause est écrite dans le message ; la trace complète s'affiche dans la cellule 3 de Colab
  (souvent une mémoire GPU saturée : le serveur réessaie seul avec moins de faisceaux).
- *« Prompt trop long »* : le schéma ou les règles métier sont trop volumineux pour le GPU.

## Remarques

- Les bases ne sont **jamais modifiées** : toute requête d'écriture est refusée.
- Un extrait des résultats est envoyé à l'API Claude pour rédiger la réponse : ne pas utiliser de données confidentielles
  sans l'accord de leur propriétaire.
- Les fichiers `data/` générés par l'application (documentation, correspondances, import JSON) et le fichier `.env`
  sont locaux et ne doivent pas être publiés.