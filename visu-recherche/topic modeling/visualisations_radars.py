#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline complète : LDA → nomination des topics (SciBERT ou fallback) → AFC Topics×Auteurs → Radar charts par auteur
"""

import os
import re
from collections import Counter

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

# Tenter d’importer SciBERT et bibliothèques associées
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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

NUM_TOPICS = 10
RADAR_DIR = "radar_charts"


# Dictionnaire des domaines & sous-thèmes pour SciBERT (exemple)
dic = {
    "Statistics": {
        "Descriptive Statistics": {
            "description": "Summarizing and describing features of a dataset",
            "keywords": ["mean", "median", "mode", "variance", "standard deviation", "percentile", "quartile", "summary statistics"]
        },
        "Inferential Statistics": {
            "description": "Making predictions or inferences about a population from a sample",
            "keywords": ["confidence interval", "hypothesis testing", "p-value", "significance level", "sample", "population", "test statistic"]
        },
        "Parametric Estimation": {
            "description": "Estimating parameters of a distribution assuming a specific form",
            "keywords": ["maximum likelihood", "method of moments", "estimator", "bias", "consistency", "efficiency"]
        },
        "Hypothesis Testing": {
            "description": "Testing assumptions about data",
            "keywords": ["null hypothesis", "alternative hypothesis", "t-test", "chi-square test", "ANOVA", "F-test", "z-test"]
        },
        "Linear Regression": {
            "description": "Modeling linear relationship between dependent and independent variables",
            "keywords": ["regression coefficient", "intercept", "residual", "R-squared", "OLS", "predictor", "response"]
        },
        "Logistic Regression": {
            "description": "Regression model for binary outcomes",
            "keywords": ["logit", "odds ratio", "binary classification", "sigmoid", "maximum likelihood"]
        },
        "Time Series Analysis": {
            "description": "Analyzing data points ordered in time",
            "keywords": ["autoregressive", "moving average", "ARIMA", "seasonality", "trend", "stationarity", "lag", "forecasting"]
        },
        "Multivariate Analysis": {
            "description": "Analysis involving multiple variables simultaneously",
            "keywords": ["PCA", "factor analysis", "canonical correlation", "MANOVA", "cluster analysis"]
        },
        "Sampling Theory": {
            "description": "Methods of selecting representative samples",
            "keywords": ["random sampling", "stratified sampling", "sampling bias", "sampling distribution"]
        },
        "Bayesian Statistics": {
            "description": "Statistical inference using Bayes' theorem",
            "keywords": ["prior", "posterior", "likelihood", "Bayes factor", "Markov Chain Monte Carlo", "Gibbs sampling"]
        },
        "Non-parametric Statistics": {
            "description": "Statistical methods without assuming parameterized distribution",
            "keywords": ["rank test", "Wilcoxon", "Kruskal-Wallis", "Mann-Whitney", "kernel density"]
        },
        "Mathematical Statistics and Machine Learning": {
            "description": "Theoretical foundations bridging statistics and ML",
            "keywords": ["probability theory", "statistical learning theory", "risk minimization", "concentration inequalities"]
        },
        "Fairness and Bias in Statistics and Machine Learning": {
            "description": "Analyzing and mitigating bias and fairness issues",
            "keywords": ["algorithmic fairness", "bias detection", "equal opportunity", "disparate impact", "fairness metrics"]
        }
    },
    "Signal Processing": {
        "Digital Filtering": {
            "description": "Removing unwanted components from a signal",
            "keywords": ["low-pass filter", "high-pass filter", "band-pass filter", "FIR", "IIR", "filter design"]
        },
        "Fourier Transform": {
            "description": "Transforming signals between time and frequency domains",
            "keywords": ["FFT", "frequency spectrum", "frequency domain", "inverse Fourier transform", "spectral analysis"]
        },
        "Wavelet Transform": {
            "description": "Multi-resolution analysis of signals",
            "keywords": ["wavelets", "scaling function", "mother wavelet", "continuous wavelet transform", "discrete wavelet transform"]
        },
        "Time-Frequency Analysis": {
            "description": "Analyzing signals whose frequency content changes over time",
            "keywords": ["spectrogram", "short-time Fourier transform", "Wigner-Ville distribution", "Hilbert-Huang transform"]
        },
        "Spectral Analysis": {
            "description": "Estimating the power distribution over frequency",
            "keywords": ["power spectral density", "periodogram", "Welch method", "spectral leakage"]
        },
        "Signal Modeling": {
            "description": "Représentation mathématique des signaux",
            "keywords": ["autoregressive model", "moving average model", "ARMA", "state-space model", "signal models", "denoising", "source separation"]
        },
        "Noise Reduction": {
            "description": "Techniques pour réduire le bruit dans les signaux",
            "keywords": ["denoising", "Wiener filter", "median filter", "thresholding"]
        },
        "Sampling and Quantization": {
            "description": "Conversion de signaux continus en discrets",
            "keywords": ["Nyquist rate", "aliasing", "quantization error", "ADC", "DAC"]
        },
        "Adaptive Signal Processing": {
            "description": "Filtres qui adaptent leurs paramètres selon le signal",
            "keywords": ["LMS algorithm", "RLS algorithm", "adaptive filtering"]
        },
        "Signal Detection": {
            "description": "Identifier un signal dans le bruit",
            "keywords": ["matched filter", "energy detection", "false alarm rate", "detection probability"]
        },
        "Image Processing": {
            "description": "Manipulation et analyse d’images",
            "keywords": ["edge detection", "filtering", "segmentation", "morphological operations", "Fourier transform"]
        },
        "Audio Signal Processing": {
            "description": "Traitement des signaux audio",
            "keywords": ["speech recognition", "echo cancellation", "noise suppression", "audio coding", "speech processing", "machine listening", "MIR", "music information retrieval"]
        },
        "Distributed Processing": {
            "description": "Signal processing sur systèmes distribués/parallèles",
            "keywords": ["distributed algorithms", "parallel processing", "edge computing", "cloud processing"]
        }
    },
    "Machine Learning": {
        "Supervised Learning": {
            "description": "Apprentissage supervisé à partir de données étiquetées",
            "keywords": ["classification", "regression", "training set", "test set", "validation set"]
        },
        "Unsupervised Learning": {
            "description": "Apprentissage de motifs dans données non-étiquetées",
            "keywords": ["clustering", "dimensionality reduction", "association rules", "anomaly detection", "data decomposition"]
        },
        "Reinforcement Learning": {
            "description": "Apprentissage par renforcement",
            "keywords": ["agent", "environment", "policy", "reward", "Q-learning", "exploration-exploitation"]
        },
        "Deep Learning": {
            "description": "Apprentissage via réseaux neuronaux profonds",
            "keywords": ["neural networks", "convolutional neural networks", "recurrent neural networks", "backpropagation", "activation function"]
        },
        "Decision Trees": {
            "description": "Modèles arborescents pour classification/régression",
            "keywords": ["node", "leaf", "splitting criterion", "entropy", "Gini index"]
        },
        "Random Forests": {
            "description": "Ensemble d’arbres de décision",
            "keywords": ["bagging", "feature importance", "out-of-bag error"]
        },
        "Support Vector Machines": {
            "description": "Classifieurs marginaux",
            "keywords": ["kernel", "margin", "support vectors", "hyperplane", "C parameter", "kernel methods"]
        },
        "Clustering Algorithms": {
            "description": "Groupement similaire des points",
            "keywords": ["k-means", "hierarchical clustering", "DBSCAN", "centroid", "dendrogram"]
        },
        "Dimensionality Reduction": {
            "description": "Réduction de la dimensionnalité",
            "keywords": ["PCA", "t-SNE", "LDA", "feature extraction", "representation learning", "multiview learning", "multimodal processing"]
        },
        "Cross-validation": {
            "description": "Validation de modèle via k-fold, etc.",
            "keywords": ["k-fold", "leave-one-out", "train-test split"]
        },
        "Hyperparameter Optimization": {
            "description": "Recherche des meilleurs hyperparamètres",
            "keywords": ["grid search", "random search", "Bayesian optimization", "stochastic optimization", "convex analysis", "optimization theory", "optimization algorithms", "stochastic algorithms", "optimal control"]
        },
        "Feature Engineering": {
            "description": "Création de caractéristiques à partir de données brutes",
            "keywords": ["feature selection", "feature extraction", "scaling", "normalization"]
        },
        "Probabilistic Models": {
            "description": "Modèles basés sur des distributions de probabilité",
            "keywords": ["Hidden Markov Model", "Conditional Random Field", "Gaussian Mixture Model", "Bayesian networks", "Bayesian methods", "machine learning probability"]
        },
        "Natural Language Processing": {
            "description": "Traitement et analyse de données textuelles",
            "keywords": ["tokenization", "stemming", "lemmatization", "sentiment analysis", "word embeddings", "computational linguistics"]
        },
        "Sentiment Analysis": {
            "description": "Détection de sentiment dans le texte",
            "keywords": ["positive", "negative", "neutral", "opinion mining"]
        },
        "Machine Learning Fairness and Robustness": {
            "description": "Assurer équité et robustesse des modèles",
            "keywords": ["fairness", "bias", "explainable AI", "interpretable AI", "robust statistics", "bias mitigation"]
        },
        "Graphs and Graph-based Learning": {
            "description": "Apprentissage sur graph-structured data",
            "keywords": ["graph neural networks", "graph embeddings", "structured prediction", "graph supervised learning"]
        },
        "Large-scale and Distributed Machine Learning": {
            "description": "Techniques pour gérer gros volumes et computing distribué",
            "keywords": ["distributed computing", "parallel processing", "large-scale data analysis", "machine learning distributed processing"]
        },
        "Optimal Transport": {
            "description": "Cadre mathématique pour comparer distributions",
            "keywords": ["optimal transport", "Wasserstein distance", "earth mover's distance"]
        },
        "Inverse Problems": {
            "description": "Reconstruction d’entrées à partir de sorties observées",
            "keywords": ["inverse problems", "regularization", "deconvolution"]
        },
        "Human Pose Estimation": {
            "description": "Détection/analyse de poses humaines dans images/vidéos",
            "keywords": ["pose estimation", "keypoint detection", "computer vision"]
        },
        "Machine Listening and Music Information Retrieval (MIR)": {
            "description": "Traitement audio pour musique/son",
            "keywords": ["machine listening", "music information retrieval", "MIR", "audio classification", "speech processing", "handwriting recognition"]
        },
        "Speech Processing": {
            "description": "Traitement des signaux vocaux",
            "keywords": ["speech recognition", "speech synthesis", "speaker identification", "voice activity detection"]
        },
        "Computer Vision": {
            "description": "Extraction d’information depuis images/vidéos",
            "keywords": ["image recognition", "object detection", "segmentation", "feature extraction", "human pose estimation"]
        },
        "Rankings and Preferences": {
            "description": "Modèles/algorithmes pour apprentissage des préférences",
            "keywords": ["ranking", "preference learning", "recommendation systems"]
        }
    }
}


# ---------- FONCTIONS UTILITAIRES ----------

def to_2d(x):
    """
    Assure la forme 2D pour cosine_similarity.
    """
    if isinstance(x, torch.Tensor):
        return x if x.ndim == 2 else x.unsqueeze(0)
    import numpy as _np
    if isinstance(x, _np.ndarray):
        return x if x.ndim == 2 else x.reshape(1, -1)
    return x

def supprimer_stopwords(texte, langue='english'):
    """
    Supprime les stopwords d’un texte.
    """
    mots = word_tokenize(texte)
    stop_words = set(stopwords.words(langue))
    return ' '.join([mot for mot in mots if mot.lower() not in stop_words])

def build_embedding_text_for_domain(domain_name, subthemes):
    """
    Construit un texte pour un domaine : nom du domaine + sous-thèmes + keywords.
    """
    parts = [domain_name]
    for subtheme, infos in subthemes.items():
        parts.append(subtheme)
        parts.extend(infos.get("keywords", []))
    return " ".join(parts)

def build_embedding_text_for_subtheme(subtheme_name, infos):
    """
    Construit un texte pour un sous-thème : nom du sous-thème + keywords.
    """
    return " ".join([subtheme_name] + infos.get("keywords", []))

def get_topic_embedding(text, tokenizer, model):
    """
    Calcule l'embedding (CLS token) pour un texte via SciBERT.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :]  # shape = (1, hidden_size)
    return cls_emb.squeeze(0)                     # shape = (hidden_size,)

def extract_embeddings_from_dic(dic, tokenizer, model):
    """
    Pour chaque domaine et chaque sous-thème de `dic`, calcule un embedding SciBERT.
    Renvoie :
      - domain_labels (list[str])
      - domain_embs   (torch.Tensor, shape = (n_domains, hidden_size))
      - subtheme_labels (list[str])
      - subtheme_embs   (torch.Tensor, shape = (n_subthemes, hidden_size))
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


# ---------- 1) LECTURE & PARSING DU FICHIER FORMATÉ ----------

with open(INPUT_FILE, encoding="utf-8") as f:
    contenu = f.read()

header_re = re.compile(r'^Thèse\s*(\d+)\s*:\s*(.*?)\s*$', re.MULTILINE)
matches = list(header_re.finditer(contenu))

documents   = []
doc_authors = []

for idx, m in enumerate(matches):
    num   = int(m.group(1))
    titre = m.group(2).strip()
    start = m.end()
    end   = matches[idx+1].start() if idx+1 < len(matches) else len(contenu)
    section = contenu[start:end].strip()

    if "\n\n" in section:
        authors_block, body = section.split("\n\n", 1)
    else:
        authors_block, body = section, ""

    auteurs = []
    for part in re.split(r',|\n', authors_block):
        name = part.strip().rstrip(',')
        if name in TARGET_AUTHORS:
            auteurs.append(name)

    if not auteurs:
        continue

    full_text = titre + " " + body.replace("\n", " ")
    tokens = [tok for tok in simple_preprocess(full_text) if tok not in STOPWORDS]

    documents.append(tokens)
    doc_authors.append(auteurs)

if not documents:
    print("Aucun document avec un auteur ciblé.")
    exit(1)


# ---------- 2) CONSTRUCTION DICTIONNAIRE & CORPUS POUR LDA ----------

dictionary = corpora.Dictionary(documents)
dictionary.filter_extremes(no_below=2, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in documents]


# ---------- 3) ENTRAÎNEMENT DU MODÈLE LDA ----------

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

# Visualisation PyLDAvis
vis_data = gensimvis.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(vis_data, "lda_visualisation.html")
print("PyLDAvis saved to lda_visualisation.html")


# ---------- 4) TOPIC DOMINANT PAR DOCUMENT ----------

doc_topics = []
for bow in corpus:
    tp = lda.get_document_topics(bow)
    dominant = max(tp, key=lambda x: x[1])[0]
    doc_topics.append(dominant)


# ---------- 5) NOMINATION DES TOPICS (SciBERT ou fallback) ----------

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
    domain_labels, emb_domains, subtheme_labels, emb_subthemes = extract_embeddings_from_dic(dic, tokenizer, model)

    topic_candidates = []
    for topic_id in range(lda.num_topics):
        terms = lda.show_topic(topic_id, topn=10)
        mots_cles = " ".join([mot for mot, _ in terms])
        topic_emb = get_topic_embedding(mots_cles, tokenizer, model)

        sim_domains    = cosine_similarity(to_2d(topic_emb.numpy()), emb_domains.numpy())[0]
        sim_subthemes  = cosine_similarity(to_2d(topic_emb.numpy()), emb_subthemes.numpy())[0]

        scored = []
        scored += list(zip(domain_labels, sim_domains))
        scored += list(zip(subtheme_labels, sim_subthemes))
        scored.sort(key=lambda x: x[1], reverse=True)
        topic_candidates.append(scored)

    all_triples = []
    for tid, candidates in enumerate(topic_candidates):
        for lbl, score in candidates:
            all_triples.append((tid, lbl, score))
    all_triples.sort(key=lambda x: x[2], reverse=True)

    assigned_label = {}
    used_labels    = set()
    for tid, lbl, score in all_triples:
        if tid not in assigned_label and lbl not in used_labels:
            assigned_label[tid] = lbl
            used_labels.add(lbl)

    for tid in range(NUM_TOPICS):
        if tid not in assigned_label:
            for lbl, _ in topic_candidates[tid]:
                if lbl not in used_labels:
                    assigned_label[tid] = lbl
                    used_labels.add(lbl)
                    break
            else:
                fallback_lbl = topic_candidates[tid][0][0]
                assigned_label[tid] = fallback_lbl
                used_labels.add(fallback_lbl)

    Topic_names = [assigned_label[i] for i in range(NUM_TOPICS)]
    print("\n>> Attribution finale des labels topic_id → label :")
    for i, lbl in enumerate(Topic_names):
        print(f"   {i} → {lbl}")

else:
    raw_labels = [lda.show_topic(i, topn=2) for i in range(lda.num_topics)]
    initial_labels = [terms[0][0] for terms in raw_labels]
    counter = Counter(initial_labels)

    Topic_names = []
    used = set()
    for tid, terms in enumerate(raw_labels):
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
                fallback_lbl = f"{premier}_{tid}"
                Topic_names.append(fallback_lbl)
                used.add(fallback_lbl)

    print("\n>> En mode fallback, labels topic_id → mot unique :")
    for i, lbl in enumerate(Topic_names):
        print(f"   {i} → {lbl}")


# ---------- 6) CONSTRUCTION DE LA MATRICE TopicID × Auteur ----------

all_authors = sorted({a for sub in doc_authors for a in sub})
cont_numeric = pd.DataFrame(0, index=list(range(NUM_TOPICS)), columns=all_authors)

for topic_id, authors_list in zip(doc_topics, doc_authors):
    for a in authors_list:
        if a in all_authors:
            cont_numeric.at[topic_id, a] += 1

cont_numeric = cont_numeric.loc[:, cont_numeric.sum(axis=0) > 0]


# ---------- 7) RÉORDONNEMENT DES TOPICS PAR SIMILARITÉ ----------

topic_term_matrix = lda.get_topics()
distances = pdist(topic_term_matrix, metric='cosine')
link = linkage(distances, method='average')
order = leaves_list(link)

cont_numeric = cont_numeric.reindex(index=order)
Topic_names = [Topic_names[i] for i in order]


# ---------- 8) ANALYSE EN CORRESPONDANCE Topics × Auteurs ----------

ca = prince.CA(n_components=2, n_iter=10, copy=True, check_input=True, engine='scipy', random_state=42)
ca = ca.fit(cont_numeric)

rows = ca.row_coordinates(cont_numeric)     # DataFrame index=topic_id (réordonné)
cols = ca.column_coordinates(cont_numeric)   # DataFrame index=auteur

print("\nIndex de cont_numeric (topic_ids) utilisé pour l’AFC :", cont_numeric.index.tolist())


# ---------- 9) PROJECTION 2D Topics & Auteurs ----------

plt.figure(figsize=(14, 10))

# Topics (cercles verts)
plt.scatter(rows.iloc[:, 0], rows.iloc[:, 1], marker='o', edgecolors='green', facecolors='none', label='Topics')
for tid in rows.index:
    lbl = Topic_names[tid]
    plt.text(rows.at[tid, 0], rows.at[tid, 1], lbl, fontsize=10, color='green')

# Auteurs (triangles rouges)
plt.scatter(cols.iloc[:, 0], cols.iloc[:, 1], marker='^', edgecolors='red', facecolors='none', label='Auteurs')
for au in cols.index:
    plt.text(cols.at[au, 0], cols.at[au, 1], au, fontsize=9, color='red')

plt.axhline(0, linestyle='--', color='gray')
plt.axvline(0, linestyle='--', color='gray')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("AFC – Topics nommés & Auteurs")
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ---------- 10) DIAGRAMMES RADAR PAR AUTEUR (ORDRE OPTIMISÉ) ----------

os.makedirs(RADAR_DIR, exist_ok=True)
categories = [Topic_names[t] for t in cont_numeric.index]  # étiquettes ordonnées
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # ferme le polygone

categories[4]="Machine listning and (MIR)"

for author in cont_numeric.columns:
    counts = cont_numeric[author].values.tolist()
    total = sum(counts)
    if total == 0:
        continue  # pas de contributions pour cet auteur

    proportions = [c / total for c in counts]
    proportions += proportions[:1]  # ferme le polygone

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, proportions, color='blue', linewidth=2)
    ax.fill(angles, proportions, color='blue', alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title(f"Distribution des topics pour {author}", y=1.1)
    ax.set_ylim(0, max(proportions) * 1.1)

    safe_name = author.replace(" ", "_") + ".png"
    fig.savefig(os.path.join(RADAR_DIR, safe_name), dpi=150, bbox_inches='tight')
    plt.close(fig)

print(f"\n✓ Pipeline complète terminée.")
print(f"  - AFC affichée à l’écran")
print(f"  - Radars par auteur sauvegardés dans '{RADAR_DIR}/'")
print("  - Vue interactive PyLDAvis : lda_visualisation.html")
