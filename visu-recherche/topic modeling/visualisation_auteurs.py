#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import prince  # pip install prince scipy

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# ---------- CONFIGURATION ----------

INPUT_FILE = "../pdf_scraping/articles_de_publications_permanents_formatted.txt"
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
NUM_TOPICS = 10     # nombre de topics pour LDA
RADAR_DIR = "radar_charts"

# Crée les dossiers de sorties
os.makedirs(RADAR_DIR, exist_ok=True)

# ---------- 1) LECTURE & PARSING ----------

with open(INPUT_FILE, encoding="utf-8") as f:
    contenu = f.read()

# Regex pour repérer les en-têtes "Thèse N : ..."
header_re = re.compile(r'^Thèse\s*(\d+)\s*:\s*(.*?)\s*$', re.MULTILINE)
matches = list(header_re.finditer(contenu))

documents   = []   # liste de listes de tokens
doc_authors = []   # liste de listes d'auteurs
doc_ids     = []   # labels "Thèse N"

for idx, m in enumerate(matches):
    num   = int(m.group(1))
    titre = m.group(2).strip()
    start = m.end()
    end   = matches[idx+1].start() if idx+1 < len(matches) else len(contenu)
    section = contenu[start:end].strip()

    # Bloc auteurs / corps
    if "\n\n" in section:
        authors_block, body = section.split("\n\n", 1)
    else:
        authors_block, body = section, ""

    # Extraction des auteurs cibles
    auteurs = [
        name.strip().rstrip(',')
        for name in re.split(r',|\n', authors_block)
        if name.strip() in TARGET_AUTHORS
    ]
    if not auteurs:
        continue

    # Préparation du texte
    full_text = titre + " " + body.replace("\n", " ")
    tokens = [tok for tok in simple_preprocess(full_text) if tok not in STOPWORDS]

    doc_ids.append(f"Thèse {num}")
    documents.append(tokens)
    doc_authors.append(auteurs)

if not documents:
    print("Aucun document avec un auteur ciblé.")
    exit(1)

# ---------- 2) DICTIONNAIRE & CORPUS ----------

dictionary = corpora.Dictionary(documents)
dictionary.filter_extremes(no_below=2, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# ---------- 3) LDA & COHERENCE & PyLDAvis ----------

lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,
    passes=10,
    alpha="auto",
    per_word_topics=False
)

# Score de cohérence
coh_model = CoherenceModel(model=lda, texts=documents, dictionary=dictionary, coherence='c_v')
print("Score de cohérence (c_v):", coh_model.get_coherence())

# Visualisation interactive
vis_data = gensimvis.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(vis_data, "lda_visualisation.html")
print("PyLDAvis saved to lda_visualisation.html")

# ---------- 4) TOPIC DOMINANT PAR DOCUMENT ----------

doc_topics = []
for bow in corpus:
    tp = lda.get_document_topics(bow)
    dominant = max(tp, key=lambda x: x[1])[0]
    doc_topics.append(dominant)

# ---------- 5) MATRICE Topic × Auteur ----------

all_authors = sorted({a for sub in doc_authors for a in sub})
all_topics  = list(range(NUM_TOPICS))

# Nombre de docs du topic t où l'auteur a apparaît
cont_dict = {t: {a: 0 for a in all_authors} for t in all_topics}
for t, auths in zip(doc_topics, doc_authors):
    for a in auths:
        cont_dict[t][a] += 1

cont_df = pd.DataFrame.from_dict(cont_dict, orient='index', columns=all_authors)
# Retirer colonnes vides
cont_df = cont_df.loc[:, cont_df.sum(axis=0) > 0]

# ---------- 6) ANALYSE EN CORRESPONDANCE ----------

ca = prince.CA(
    n_components=2,
    n_iter=10,
    copy=True,
    check_input=True,
    engine='scipy',
    random_state=42
).fit(cont_df)

row_coords = ca.row_coordinates(cont_df)    # coords des topics
col_coords = ca.column_coordinates(cont_df) # coords des auteurs

# Préparer labels de topics (2 mots dominants)
topic_labels = {
    t: " / ".join([w for w, _ in lda.show_topic(t, topn=2)])
    for t in all_topics
}

# ---------- 7) PROJECTION 2D Topics & Auteurs ----------

plt.figure(figsize=(10, 7))
# Topics en vert
plt.scatter(row_coords[0], row_coords[1], c='green', marker='o', label='Topics')
for i, t in enumerate(row_coords.index):
    plt.text(row_coords.iloc[i,0], row_coords.iloc[i,1], topic_labels[t],
             fontsize=9, color='green')
# Auteurs en rouge
plt.scatter(col_coords[0], col_coords[1], c='red', marker='^', label='Auteurs')
for i, a in enumerate(col_coords.index):
    plt.text(col_coords.iloc[i,0], col_coords.iloc[i,1], a,
             fontsize=8, color='red')

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Analyse en Correspondance – Topics & Auteurs")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# ---------- 8) DIAGRAMMES RADAR PAR AUTEUR ----------

os.makedirs(RADAR_DIR, exist_ok=True)
categories = [topic_labels[t] for t in cont_df.index]
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for author in cont_df.columns:
    counts = cont_df[author].values.tolist()
    total = sum(counts)
    if total == 0: continue
    proportions = [c/total for c in counts]
    proportions += proportions[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, proportions, color='blue', linewidth=2)
    ax.fill(angles, proportions, color='blue', alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title(f"Distribution topics pour {author}", y=1.1)
    ax.set_ylim(0, max(proportions)*1.1)

    fname = author.replace(" ", "_") + ".png"
    fig.savefig(os.path.join(RADAR_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)

print("✓ Pipeline terminée. Radars dans", RADAR_DIR)
