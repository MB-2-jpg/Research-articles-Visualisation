# -*- coding: utf-8 -*-
"""
AFC Topics–Auteurs avec nomination automatique des libellés (domaines & sous-thèmes),
et fallback unique en mode « mot-clé principal » si SciBERT ne peut pas se charger
(évite que deux topics aient le même nom).

Cette version détecte deux formats d’auteurs après le titre :
 1) plusieurs auteurs sur une ou deux lignes, séparés par des virgules
 2) un auteur par ligne (avec parfois un indice d’affiliation à la fin)
"""

import os
import re
import sys
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Tenter d’importer les packages nécessaires à SciBERT
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

from prince import CA
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import unicodedata



import re
import unicodedata

# --- Fonctions utilitaires ---


def normalize_str(s: str) -> str:
    """
    Retire les accents et met en minuscules.
    Exemple : "Mathieu Labeáu" → "mathieu labeau"
    """
    nfkd = unicodedata.normalize("NFKD", s)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_accents.lower().strip()


def clean_author_name(line: str) -> str:
    """
    Retire tout caractère de fin qui n'est pas lettre (y compris chiffres, symboles, astérisques).
    Conserve accents, apostrophes, tirets, etc.
    Exemples :
      - "Mathieu Labeau1"    → "Mathieu Labeau"
      - "David Perera∗ 1"    → "David Perera"
      - "Maria Boritchev *"  → "Maria Boritchev"
      - "Phoebe O’Connor2†"  → "Phoebe O’Connor"
    """
    # Supprime tout caractère final non lettre, apostrophe ou tiret
    cleaned = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'\- ]+$", "", line).strip()
    return cleaned


def supprimer_stopwords(texte: str, langue='english') -> str:
    """
    Enlève les stopwords d'un texte. Retourne la chaîne sans stopwords.
    """
    mots = word_tokenize(texte)
    stop_words = set(stopwords.words(langue))
    mots_filtrés = [mot for mot in mots if mot.lower() not in stop_words]
    return ' '.join(mots_filtrés)


def to_2d(x):
    """
    Assure la forme 2D pour cosine_similarity, 
    qu'il s'agisse d'un torch.Tensor ou d'un numpy.ndarray.
    """
    if TRANSFORMERS_AVAILABLE and isinstance(x, torch.Tensor):
        return x if x.ndim == 2 else x.unsqueeze(0)
    import numpy as np
    if isinstance(x, np.ndarray):
        return x if x.ndim == 2 else x.reshape(1, -1)
    return x


def build_embedding_text_for_domain(domain_name: str, subthemes: dict) -> str:
    """
    Construit un texte pour un domaine en concaténant :
    - Nom du domaine
    - Noms des sous-thèmes
    - Mots-clés des sous-thèmes
    """
    parts = [domain_name]
    for subtheme, infos in subthemes.items():
        parts.append(subtheme)
        parts.extend(infos.get("keywords", []))
    return " ".join(parts)


def build_embedding_text_for_subtheme(subtheme_name: str, infos: dict) -> str:
    """
    Construit un texte pour un sous-thème : nom du sous-thème + mots-clés
    """
    return " ".join([subtheme_name] + infos.get("keywords", []))


def get_topic_embedding(text: str, tokenizer, model):
    """
    Calcule l'embedding (CLS token) pour un texte via SciBERT.
    Retourne un torch.Tensor de taille (hidden_size,).
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)
    return cls_emb.squeeze(0)                     # (hidden_size,)


def extract_embeddings_from_dic(dic: dict, tokenizer, model):
    """
    Pour chaque domaine et chaque sous-thème de `dic`, calcule un embedding SciBERT.
    Retourne :
      - domain_labels    (list[str])
      - domain_embs      (torch.Tensor, shape = (n_domains, hidden_size))
      - subtheme_labels  (list[str])
      - subtheme_embs    (torch.Tensor, shape = (n_subthemes, hidden_size))
    """
    domain_labels    = []
    domain_embs      = []
    subtheme_labels  = []
    subtheme_embs    = []

    for domain_name, subthemes in dic.items():
        text_domain = build_embedding_text_for_domain(domain_name, subthemes)
        emb_domain  = get_topic_embedding(text_domain, tokenizer, model)
        domain_labels.append(domain_name)
        domain_embs.append(emb_domain)

        for subtheme_name, infos in subthemes.items():
            text_sub = build_embedding_text_for_subtheme(subtheme_name, infos)
            emb_sub  = get_topic_embedding(text_sub, tokenizer, model)
            subtheme_labels.append(subtheme_name)
            subtheme_embs.append(emb_sub)

    domain_embs_tensor   = torch.stack(domain_embs)    # (n_domains, hidden_size)
    subtheme_embs_tensor = torch.stack(subtheme_embs)  # (n_subthemes, hidden_size)
    return domain_labels, domain_embs_tensor, subtheme_labels, subtheme_embs_tensor


# --- Paramètres ---

THESIS_FILE = "../pdf_scraping/articles_permanents.txt"
TARGET_AUTHORS = {
    "Roland Badeau", "Pascal Bianchi", "Philippe Ciblat",
    "Stephan Clémençon", "Florence d'Alché-Buc", "Slim Essid",
    "Olivier Fercoq", "Pavlo Mozharovskyi", "Geoffroy Peeters",
    "Gaël Richard", "François Roueff", "Maria Boritchev",
    "Radu Dragomir", "Mathieu Fontaine", "Ekhiñe Irurozki",
    "Yann Issartel", "Hicham Janati", "Ons Jelassi",
    "Matthieu Labeau", "Charlotte Laclau",
    "Laurence Likforman-Sulem", "Yves Grenier"
}

# On construira cet ensemble pour les comparaisons sans tenir compte des accents
TARGET_NORMALIZED = { normalize_str(n) for n in TARGET_AUTHORS }


# --- Lecture du fichier et découpage en “Article #N” ---

with open(THESIS_FILE, 'r', encoding='utf-8') as f:
    contenu = f.read()

# La regex repère “========== Article #1: … ==========” au début de ligne
article_pattern = re.compile(r"^=+ Article #(\d+):.*$", re.MULTILINE)
splits = article_pattern.split(contenu)
# splits = ["…avant…", "1", blocTexte1, "2", blocTexte2, "3", blocTexte3, …]

# On va stocker chaque article sous cette forme :
#   {
#     "numero": 1,
#     "titre": "Titre de l’article...",
#     "raw_authors": [liste de noms brut],
#     "membres":    [sous‐liste de raw_authors filtrée par TARGET_AUTHORS]
#   }
articles = []

for idx in range(1, len(splits), 2):
    num = int(splits[idx])
    bloc = splits[idx + 1].split("\n")  # liste des lignes du bloc

    # 1) EXTRACTION DU TITRE (plusieurs lignes possibles)
    title_lines = []
    i = 0

    # On saute d'abord toute ligne vide éventuelle
    while i < len(bloc) and not bloc[i].strip():
        i += 1

    # Tant que la ligne ne contient pas une virgule (qui signifie "comma‐separated"),
    # et qu'elle ne commence pas par une phrase de métadonnée (ex. "To cite", "HAL Id", "https://"), 
    # on l'ajoute au titre.
    while i < len(bloc):
        line = bloc[i].strip()
        # Si c'est le début d'un bloc “To cite”, on considère que le titre est terminé
        if line.lower().startswith("to cite"):
            break
        # Si la ligne contient une virgule, c'est probablement la première ligne des auteurs au format CSV
        if ',' in line:
            break
        # Sinon, on considère que c'est encore une ligne de titre
        if line:
            title_lines.append(line)
        i += 1

    titre = " ".join(title_lines).strip()

    # 2) EXTRACTION DES AUTEURS BRUTS (« raw_authors »)
    raw_authors = []

    # --- 2.a) Mode “comma‐separated” (plusieurs noms sur une ou deux lignes, séparés par virgule) ---
    if i < len(bloc) and ',' in bloc[i]:
        # Tant qu'on trouve des virgules, on collecte
        while i < len(bloc):
            line = bloc[i].strip()
            # Si ligne vide ou “To cite”, on arrête
            if not line or line.lower().startswith("to cite"):
                break
            # On ajoute la ligne si elle contient une virgule
            if ',' in line:
                raw_authors.append(line)
                i += 1
            else:
                break
        # À la fin, on joint tout et on split sur “,” pour obtenir chaque nom en brut
        joined = " ".join(raw_authors)
        noms = [n.strip() for n in joined.split(",") if n.strip()]

    else:
        # --- 2.b) Mode “one‐per‐line” (un auteur par ligne) ---
        # On continue à lire TANT QUE la ligne ressemble à un nom (Prénom Nom)
        # et ne ressemble pas à “To cite”, “HAL Id”, “https://”, ni un email.
        noms = []
        while i < len(bloc):
            line = bloc[i].strip()
            # Si ligne vide, ou “To cite” ou lien ou email, on arrête la boucle
            if not line or line.lower().startswith("to cite") or "http" in line or "@" in line:
                break

            # Critère simplifié “Prénom Nom” : 
            #  - la première lettre est une majuscule (A–Z ou accentuée) 
            #  - il y a au moins un espace (=> plusieurs mots)
            if re.match(r"^[A-ZÀ-ÖØ-öø-ÿ].*\s+.*$", line):
                # On nettoie la fin (retire chiffres, “*”, “∗”, “†”, etc.)
                cleaned = clean_author_name(line)
                if cleaned:
                    noms.append(cleaned)
                i += 1
                continue

            # Sinon on arrête
            break

    # À ce stade, `noms` contient la liste brute (raw) des auteurs pour cet article
    # Exemples de raw :
    #  - ["David Perera∗ 1", "Victor Letzelter∗ 1 2", "Théo Mariotte1", "…", "Slim Essid1", "Gaël Richard1"]
    #  - ou ["David Perera", "Victor Letzelter", "Théo Mariotte", "Adrien Cortés", "Mickael Chen", "Slim Essid", "Gaël Richard"]

    # filtrage “exact match” (mais on compare en forme normalisée)
    auteurs_valides = []
    for raw in noms:
        norm = normalize_str(raw)
        if norm in TARGET_NORMALIZED:
            # on prend la forme “propre” qui figure dans TARGET_AUTHORS
            # (on récupère l’original via un petit mapping inversé)
            for orig in TARGET_AUTHORS:
                if normalize_str(orig) == norm:
                    auteurs_valides.append(orig)
                    break

    # On stocke tout
    articles.append({
        "numero": num,
        "titre": titre,
        "raw_authors": noms,
        "membres": auteurs_valides
    })


# -------------------------- FIN DU PARSING --------------------------



# 3) DIAGNOSTIC : quels TARGET_AUTHORS n’ont JAMAIS été repérés en “raw” ?
found_raw_norm = set()
for art in articles:
    for raw in art["raw_authors"]:
        found_raw_norm.add(normalize_str(raw))

missing_norm = TARGET_NORMALIZED - found_raw_norm
missing_authors = [orig for orig in TARGET_AUTHORS if normalize_str(orig) in missing_norm]

print("\n" + "="*60)
print("→ Auteurs de TARGET_AUTHORS *jamais* extraits (même en brut) :")
if missing_authors:
    for a in sorted(missing_authors):
        print("   •", a)
else:
    print("   (Tous les TARGET_AUTHORS ont été repérés en brut au moins une fois.)")
print("="*60 + "\n")


# --- 5. Préparation des documents (titres) pour LDA --------------------------

documents = [art["titre"] for art in articles]
texts = [simple_preprocess(supprimer_stopwords(txt)) for txt in documents]

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=2, no_above=0.5)
corpus = [dictionary.doc2bow(text) for text in texts]

# --- 6. Entraînement du modèle LDA --------------------------------------------

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,
    passes=10,
    iterations=100
)
# ... (tout ce qui précède reste identique, jusqu'à la partie SciBERT)

# --- 7. Chargement SciBERT ou fallback mot-clé principal ----------------------

if TRANSFORMERS_AVAILABLE:
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model     = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        USE_SCIBERT = True
    except Exception as e:
        print(f"⚠️ Impossible de charger SciBERT ({e}). Basculer en mode mot-clé principal.")
        USE_SCIBERT = False
else:
    print("⚠️ transformers non disponible → fallback mot-clé principal.")
    USE_SCIBERT = False

if USE_SCIBERT:
    # --- 8a. Version SciBERT : extraire embeddings pour domaines & sous-thèmes ---
    # ⚠️ On passe bien les 3 arguments : dic, tokenizer, model
    domain_labels, emb_domains, subtheme_labels, emb_subthemes = extract_embeddings_from_dic(
        dic, tokenizer, model
    )

    # Construire pour chaque topic la liste (label, score)
    topic_candidates = []
    for topic_id in range(lda_model.num_topics):
        terms = lda_model.show_topic(topic_id, topn=10)
        mots_cles = " ".join([mot for mot, _ in terms])
        topic_emb = get_topic_embedding(mots_cles, tokenizer, model)

        sim_domains   = cosine_similarity(to_2d(topic_emb.numpy()), emb_domains.numpy())[0]
        sim_subthemes = cosine_similarity(to_2d(topic_emb.numpy()), emb_subthemes.numpy())[0]

        scored = []
        scored += list(zip(domain_labels, sim_domains))
        scored += list(zip(subtheme_labels, sim_subthemes))
        scored.sort(key=lambda x: x[1], reverse=True)
        topic_candidates.append(scored)

    # --- 8b. Appariement glouton global pour labels uniques ---
    all_triples = []
    for topic_id, candidates in enumerate(topic_candidates):
        for label, score in candidates:
            all_triples.append((topic_id, label, score))
    all_triples.sort(key=lambda x: x[2], reverse=True)

    assigned_label = {}
    used_labels    = set()
    for topic_id, label, score in all_triples:
        if topic_id not in assigned_label and label not in used_labels:
            assigned_label[topic_id] = label
            used_labels.add(label)

    # Si un topic reste sans label, lui donner le premier label disponible
    for topic_id in range(NUM_TOPICS):
        if topic_id not in assigned_label:
            for label, _ in topic_candidates[topic_id]:
                if label not in used_labels:
                    assigned_label[topic_id] = label
                    used_labels.add(label)
                    break
            else:
                fallback_label = topic_candidates[topic_id][0][0]
                assigned_label[topic_id] = fallback_label
                used_labels.add(fallback_label)

    Topic_names = [assigned_label[i] for i in range(NUM_TOPICS)]

    print("\n>> Attribution finale des labels (topic_id → label) :")
    for i, lbl in enumerate(Topic_names):
        print(f"   {i} → {lbl}")

else:
    # --- 8c. Fallback “mot-clé principal” (LDA) + éviter doublons ---
    raw_labels = [lda_model.show_topic(topic_id, topn=2) for topic_id in range(lda_model.num_topics)]
    initial_labels = [terms[0][0] for terms in raw_labels]
    counter = Counter(initial_labels)

    Topic_names = []
    used = set()
    for topic_id, terms in enumerate(raw_labels):
        premier = terms[0][0]
        deuxieme = terms[1][0] if len(terms) > 1 else None

        if counter[premier] == 1 and premier not in used:
            Topic_names.append(premier)
            used.add(premier)
        else:
            if deuxieme and deuxieme not in used:
                Topic_names.append(deuxieme)
                used.add(deuxieme)
            else:
                fallback = f"{premier}_{topic_id}"
                Topic_names.append(fallback)
                used.add(fallback)

    print("\n>> En mode fallback, on évite les doublons ainsi :")
    for i, lbl in enumerate(Topic_names):
        print(f"   {i} → {lbl}")


# --- 9. Reconstruction de doc_authors (filtrée) -----------------------------

doc_authors = []
for art in articles:
    doc_authors.append(art["membres"])

filtered_topics  = []
filtered_authors = []
for idx, auteurs in enumerate(doc_authors):
    if auteurs:
        dominant = max(lda_model.get_document_topics(corpus[idx]), key=lambda x: x[1])[0]
        filtered_topics.append(dominant)
        filtered_authors.append(auteurs)

# --- 10. Construction de la matrice TopicID × Auteur (complète) --------------

all_authors = sorted({a for auths in filtered_authors for a in auths})
cont_numeric = pd.DataFrame(0, index=list(range(NUM_TOPICS)), columns=all_authors)

for topic_id, authors in zip(filtered_topics, filtered_authors):
    for a in authors:
        cont_numeric.at[topic_id, a] += 1

# Supprimer les colonnes (auteurs) sans occurrence
cont_numeric = cont_numeric.loc[:, cont_numeric.sum(axis=0) > 0]

# --- 11. Vérification de cont_numeric avant l’AFC ----------------------------

if cont_numeric.shape[1] == 0:
    print("\n❌ ERREUR : après filtrage, cont_numeric n’a AUCUNE colonne.")
    print("             → Aucun membre dans TARGET_AUTHORS n’a été trouvé.")
    sys.exit(1)

# --- 12. Réaliser l’AFC sur cont_numeric --------------------------------------

ca   = CA(n_components=2, random_state=42).fit(cont_numeric)
rows = ca.row_coordinates(cont_numeric)    # index=topic_id, colonnes=(Dim1,Dim2)
cols = ca.column_coordinates(cont_numeric) # index=auteur, colonnes=(Dim1,Dim2)

print("\nIndex de cont_numeric (topic_ids) utilisé pour l’AFC :", cont_numeric.index.tolist())

# --- 13. Projection 2D et visualisation -------------------------------------

plt.figure(figsize=(14, 10))

# — Topics (cercles verts) —
plt.scatter(rows.iloc[:, 0], rows.iloc[:, 1],
            marker='o', edgecolors='green', facecolors='none', label='Topics')
for i in rows.index:
    label = Topic_names[i]
    plt.text(rows.at[i, 0], rows.at[i, 1], label, fontsize=10, color='green')

# — Auteurs (triangles rouges) —
plt.scatter(cols.iloc[:, 0], cols.iloc[:, 1],
            marker='^', edgecolors='red', facecolors='none', label='Auteurs')
for author in cols.index:
    plt.text(cols.at[author, 0], cols.at[author, 1], author, fontsize=9, color='red')

plt.axhline(0, linestyle='--', color='gray')
plt.axvline(0, linestyle='--', color='gray')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("AFC – Topics nommés par domaine/sous-thème & Auteurs (labels uniques)")
plt.legend(loc='best')
plt.tight_layout()
plt.show()
