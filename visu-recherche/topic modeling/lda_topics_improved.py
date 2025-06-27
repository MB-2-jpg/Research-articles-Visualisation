import re
from collections import defaultdict, Counter

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

# --- Configuration ---
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
NUM_TOPICS = 10  # nombre de topics

# --- 1. Lecture & parsing ---
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    contenu = f.read()

header_re = re.compile(r'^Thèse\s*(\d+)\s*:\s*(.*?)\s*$', re.MULTILINE)
matches = list(header_re.finditer(contenu))

documents = []
doc_authors = []

for idx, m in enumerate(matches):
    # bornes de la section
    start = m.end()
    end = matches[idx+1].start() if idx+1 < len(matches) else len(contenu)
    section = contenu[start:end].strip()

    # séparer bloc auteurs / texte
    if "\n\n" in section:
        authors_block, body = section.split("\n\n", 1)
    else:
        authors_block, body = section, ""

    # récupérer auteurs ciblés
    authors = [
        name.strip().rstrip(',')
        for name in re.split(r',|\n', authors_block)
        if name.strip() in TARGET_AUTHORS
    ]
    if not authors:
        continue

    title = m.group(2).strip()
    full_text = title + " " + body.replace("\n", " ")
    tokens = [t for t in simple_preprocess(full_text) if t not in STOPWORDS]

    documents.append(tokens)
    doc_authors.append(authors)

# --- 2. Dictionnaire & corpus pour LDA ---
dictionary = corpora.Dictionary(documents)
dictionary.filter_extremes(no_below=2, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# --- 3. Entraînement LDA ---
lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,
    passes=10,
    alpha="auto",
    per_word_topics=False
)

# --- 4. Coherence score et PyLDAvis ---
coherence_model = CoherenceModel(
    model=lda,
    texts=documents,
    dictionary=dictionary,
    coherence='c_v'
)
print("Coherence (c_v):", coherence_model.get_coherence())

vis_data = gensimvis.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(vis_data, 'lda_visualisation.html')

# --- 5. Topic dominant par document ---
doc_topics = []
for bow in corpus:
    topics_probs = lda.get_document_topics(bow)
    dominant = max(topics_probs, key=lambda x: x[1])[0]
    doc_topics.append(dominant)

# --- 6. Construction de la matrice Topic × Auteur ---
all_authors = sorted(TARGET_AUTHORS)
all_topics = list(range(NUM_TOPICS))

# Compter co-occurrences
cont_dict = {t: {a: 0 for a in all_authors} for t in all_topics}
for topic, authors in zip(doc_topics, doc_authors):
    for a in authors:
        cont_dict[topic][a] += 1

cont_df = pd.DataFrame.from_dict(cont_dict, orient='index', columns=all_authors)
cont_df = cont_df.loc[:, cont_df.sum(axis=0) > 0]

# --- 7. Analyse en Correspondance (Topic × Auteur) ---
ca = prince.CA(
    n_components=2,
    n_iter=10,
    copy=True,
    check_input=True,
    engine='scipy',
    random_state=42
).fit(cont_df)

row_coords = ca.row_coordinates(cont_df)    # topics
col_coords = ca.column_coordinates(cont_df) # auteurs

# --- 8. Préparer les labels de topics avec les mots dominants ---
topic_labels = {}
for t in all_topics:
    top_words = [w for w, _ in lda.show_topic(t, topn=2)]
    topic_labels[t] = " / ".join(top_words)

# --- 9. Visualisation 2D enrichie ---
plt.figure(figsize=(12, 8))

# Topics (cercles verts)
plt.scatter(row_coords[0], row_coords[1], marker='o', c='green', label='Topics')
for i, t in enumerate(row_coords.index):
    plt.text(
        row_coords.iloc[i, 0], row_coords.iloc[i, 1],
        topic_labels[t],
        fontsize=9, color='green'
    )

# Auteurs (triangles rouges)
plt.scatter(col_coords[0], col_coords[1], marker='^', c='red', label='Auteurs')
for i, auth in enumerate(col_coords.index):
    plt.text(
        col_coords.iloc[i, 0], col_coords.iloc[i, 1],
        auth,
        fontsize=8, color='red'
    )

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Analyse en Correspondance – Topics & Auteurs (mots dominants)")
plt.legend(loc='best')
plt.tight_layout()
plt.show()
