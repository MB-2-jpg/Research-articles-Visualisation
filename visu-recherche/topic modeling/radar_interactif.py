#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline complète LDA + AFC + Radar interactif + Suggestion d’auteur le plus proche.

Usage :
    python pipeline_interactif_suggestion.py
"""

import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prince  # pip install prince scipy
from prince import CA

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
def to_2d(x):
    """
    Assure la forme 2D pour cosine_similarity.
    """
    if isinstance(x, torch.Tensor):
        return x if x.ndim == 2 else x.unsqueeze(0)
    if isinstance(x, np.ndarray):
        return x if x.ndim == 2 else x.reshape(1, -1)
    return x

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

# Tenter d’importer SciBERT et bibliothèques associées
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

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
NUM_TOPICS = 10       # Nombre de topics pour LDA
RADAR_DIR = "radar_charts"  # Dossier de sorties pour radars

# Dictionnaire des domaines & sous-thèmes pour SciBERT (optionnel)
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
            "description": "Transforming signals entre time & frequency",
            "keywords": ["FFT", "frequency spectrum", "frequency domain", "inverse Fourier transform", "spectral analysis"]
        },
        "Wavelet Transform": {
            "description": "Multi-resolution analysis of signals",
            "keywords": ["wavelets", "scaling function", "mother wavelet", "continuous wavelet transform", "discrete wavelet transform"]
        },
        "Time-Frequency Analysis": {
            "description": "Analyzing signals whose frequency content changes au fil du temps",
            "keywords": ["spectrogram", "short-time Fourier transform", "Wigner-Ville distribution", "Hilbert-Huang transform"]
        },
        "Spectral Analysis": {
            "description": "Estimating la power distribution over frequency",
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

# ---------- 1.bis) FILTRAGE EXACT (TITRE+CORPS) POUR DOCUMENTS & AUTHORS ----------

with open(INPUT_FILE, encoding="utf-8") as f:
    contenu_total = f.read()

header_re = re.compile(r'^Thèse\s*(\d+)\s*:\s*(.*?)\s*$', re.MULTILINE)
matches = list(header_re.finditer(contenu_total))

documents   = []
doc_authors = []

for idx, m in enumerate(matches):
    num   = int(m.group(1))
    titre = m.group(2).strip()
    start = m.end()
    end   = matches[idx+1].start() if idx+1 < len(matches) else len(contenu_total)
    section = contenu_total[start:end].strip()

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
    print("Aucun document (titre+corps) n'a d’auteur ciblé.")
    exit(1)

# ---------- 2) DICTIONNAIRE & CORPUS POUR LDA ----------

dictionary = corpora.Dictionary(documents)
dictionary.filter_extremes(no_below=2, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# ---------- 3) ENTRAINEMENT DU MODEL LDA ----------

lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,
    passes=10,
    alpha="auto",
    per_word_topics=False
)

coh_model = CoherenceModel(model=lda, texts=documents, dictionary=dictionary, coherence='c_v')
print("Score de cohérence (c_v):", coh_model.get_coherence())

# ---------- 4) NOMMAGE DES TOPICS (SciBERT ou fallback) ----------

if TRANSFORMERS_AVAILABLE:
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model     = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        USE_SCIBERT = True
    except Exception as e:
        print(f"⚠️ Impossible de charger SciBERT ({e}) → fallback mot-clé principal.")
        USE_SCIBERT = False
else:
    print("⚠️ transformers non disponible → fallback mot-clé principal.")
    USE_SCIBERT = False

if USE_SCIBERT:
    domain_labels, emb_domains, subtheme_labels, emb_subthemes = extract_embeddings_from_dic(dic, tokenizer, model)
    topic_candidates = []
    for topic_id in range(NUM_TOPICS):
        terms = lda.show_topic(topic_id, topn=10)
        mots_cles = " ".join([mot for mot, _ in terms])
        topic_emb = get_topic_embedding(mots_cles, tokenizer, model)
        sim_domains = cosine_similarity(to_2d(topic_emb.numpy()), emb_domains.numpy())[0]
        sim_subthemes = cosine_similarity(to_2d(topic_emb.numpy()), emb_subthemes.numpy())[0]
        scored = list(zip(domain_labels, sim_domains)) + list(zip(subtheme_labels, sim_subthemes))
        scored.sort(key=lambda x: x[1], reverse=True)
        topic_candidates.append(scored)
    all_triples = []
    for tid, candidates in enumerate(topic_candidates):
        for lbl, score in candidates:
            all_triples.append((tid, lbl, score))
    all_triples.sort(key=lambda x: x[2], reverse=True)
    assigned_label = {}
    used_labels = set()
    for tid, lbl, _ in all_triples:
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
else:
    raw_labels = [lda.show_topic(i, topn=2) for i in range(NUM_TOPICS)]
    initial = [terms[0][0] for terms in raw_labels]
    counter = Counter(initial)
    Topic_names = []
    used = set()
    for i, terms in enumerate(raw_labels):
        first = terms[0][0]
        second = terms[1][0] if len(terms) > 1 else None
        if counter[first] == 1 and first not in used:
            Topic_names.append(first)
            used.add(first)
        else:
            if second and second not in used:
                Topic_names.append(second)
                used.add(second)
            else:
                fallback_lbl = f"{first}_{i}"
                Topic_names.append(fallback_lbl)
                used.add(fallback_lbl)

print("\n>> Topic_labels finales :")
for i, lbl in enumerate(Topic_names):
    print(f"  {i} → {lbl}")

# ---------- 5) DOCUMENT → TOPIC DOMINANT ----------

doc_topics = []
for bow in corpus:
    tp = lda.get_document_topics(bow)
    dominant = max(tp, key=lambda x: x[1])[0]
    doc_topics.append(dominant)

# ---------- 6) CONSTRUCTION DE cont_numeric (TopicID × Auteur) ----------

all_authors = sorted({a for sub in doc_authors for a in sub})
cont_numeric = pd.DataFrame(0, index=list(range(NUM_TOPICS)), columns=all_authors)
for t, auths in zip(doc_topics, doc_authors):
    for a in auths:
        cont_numeric.at[t, a] += 1
cont_numeric = cont_numeric.loc[:, cont_numeric.sum(axis=0) > 0]

# ---------- 7) RÉORDONNEMENT DES TOPICS POUR RADAR & AFC ----------

topic_term_matrix = lda.get_topics()
distances = pdist(topic_term_matrix, metric='cosine')
link = linkage(distances, method='average')
order = leaves_list(link)

cont_numeric = cont_numeric.reindex(index=order)
Topic_names = [Topic_names[i] for i in order]

# ---------- 8) AFC Topics & Auteurs ----------

ca = CA(n_components=2, n_iter=10, copy=True, check_input=True, engine='scipy', random_state=42)
ca = ca.fit(cont_numeric)
row_coords = ca.row_coordinates(cont_numeric)
col_coords = ca.column_coordinates(cont_numeric)

plt.figure(figsize=(14, 10))
plt.scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1],
            marker='o', edgecolors='green', facecolors='none', label='Topics')
for tid in row_coords.index:
    lbl = Topic_names[tid]
    plt.text(row_coords.at[tid, 0], row_coords.at[tid, 1], lbl, fontsize=10, color='green')

plt.scatter(col_coords.iloc[:, 0], col_coords.iloc[:, 1],
            marker='^', edgecolors='red', facecolors='none', label='Auteurs')
for au in col_coords.index:
    plt.text(col_coords.at[au, 0], col_coords.at[au, 1], au, fontsize=9, color='red')

plt.axhline(0, linestyle='--', color='gray')
plt.axvline(0, linestyle='--', color='gray')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("AFC – Topics & Auteurs")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# ---------- 9) RADAR INTERACTIF & COLLECTION DES SOMMETS EXACTS ----------

base_angles = np.linspace(0, 2*np.pi, NUM_TOPICS, endpoint=False).tolist()
display_angles = [ (np.pi/2 - a) % (2*np.pi) for a in base_angles ]

click_points = []
collected = False

def plot_axes_and_instructions(ax):
    ax.clear()
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rmax(1.0)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_rlabel_position(135)
    ax.set_thetagrids(np.degrees(base_angles), Topic_names)
    ax.grid(True)
    ax.set_title(
        f"Cliquez {NUM_TOPICS} fois (un sommet par topic).\n"
        "Chaque clic définit exactement un sommet du polygone.\n"
        "Puis appuyez sur 'Entrée' pour confirmer ou quittez dès N clics.",
        va='bottom'
    )

def draw_collected_points(ax):
    for theta, r in click_points:
        ax.plot(theta, r, 'ro')

def finalize_polygon(ax):
    assigned = [None]*NUM_TOPICS
    used = set()
    for i, disp in enumerate(display_angles):
        diffs = []
        for j, (theta_c, r_c) in enumerate(click_points):
            d = abs(theta_c - disp)
            d = min(d, 2*np.pi - d)
            diffs.append((d, j))
        diffs.sort(key=lambda x: x[0])
        for _, jmin in diffs:
            if jmin not in used:
                assigned[i] = jmin
                used.add(jmin)
                break

    theta_vertices = []
    r_vertices = []
    for i in range(NUM_TOPICS):
        idx = assigned[i]
        theta_c, r_c = click_points[idx]
        r_c = min(max(r_c, 0.0), 1.0)
        theta_vertices.append(theta_c)
        r_vertices.append(r_c)

    theta_loop = theta_vertices + theta_vertices[:1]
    r_loop = r_vertices + r_vertices[:1]

    ax.plot(theta_loop, r_loop, color='blue', linewidth=2)
    ax.fill(theta_loop, r_loop, color='blue', alpha=0.25)

    annotation = "\n".join([f"{Topic_names[i]}: {r_vertices[i]:.3f}" for i in range(NUM_TOPICS)])
    ax.text(0.5, -0.15, annotation, transform=ax.transAxes, fontsize=9,
            va='top', ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

    return r_vertices

def on_click(event):
    global collected
    if collected:
        return
    if event.inaxes is None:
        return
    theta_click = event.xdata
    r_click = event.ydata
    if theta_click is None or r_click is None:
        return
    r_norm = min(max(r_click, 0.0), 1.0)
    click_points.append((theta_click, r_norm))
    plot_axes_and_instructions(ax)
    draw_collected_points(ax)
    fig.canvas.draw_idle()
    if len(click_points) == NUM_TOPICS:
        collected = True
        plt.pause(0.2)
        user_values = finalize_polygon(ax)
        fig.canvas.draw_idle()

def on_key(event):
    if event.key == "enter":
        plt.close(fig)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
plot_axes_and_instructions(ax)
cid1 = fig.canvas.mpl_connect('button_press_event', on_click)
cid2 = fig.canvas.mpl_connect('key_press_event', on_key)
plt.tight_layout()
plt.show()

# ---------- 10) POST-INTERACTION : calcul dist et suggestion auteur ----------

if len(click_points) < NUM_TOPICS:
    print(f"\nSeulement {len(click_points)} clic(s); polygone non finalisé.")
    exit(0)

assigned = [None]*NUM_TOPICS
used = set()
for i, disp in enumerate(display_angles):
    diffs = []
    for j, (theta_c, r_c) in enumerate(click_points):
        d = abs(theta_c - disp)
        d = min(d, 2*np.pi - d)
        diffs.append((d, j))
    diffs.sort(key=lambda x: x[0])
    for _, jmin in diffs:
        if jmin not in used:
            assigned[i] = jmin
            used.add(jmin)
            break

user_values = []
for i in range(NUM_TOPICS):
    idx = assigned[i]
    r_c = click_points[idx][1]
    r_c = min(max(r_c, 0.0), 1.0)
    user_values.append(r_c)

user_df = pd.DataFrame({"Topic": Topic_names, "Valeur": user_values})
user_df.to_csv("radar_projections_interactif.csv", index=False, encoding='utf-8')
print("\n→ Projections utilisateur sauvegardées dans 'radar_projections_interactif.csv'")

author_profiles = {}
for author in cont_numeric.columns:
    vec = cont_numeric[author].values.astype(float)
    total = vec.sum()
    if total > 0:
        author_profiles[author] = vec / total

uv = np.array(user_values)
distances = {author: np.linalg.norm(uv - profile) for author, profile in author_profiles.items()}
best_author = min(distances, key=lambda a: distances[a])
print(f"\nAuteur proposé (distance minimale) : {best_author} (dist = {distances[best_author]:.4f})")

best_profile = author_profiles[best_author]
theta_author = base_angles + base_angles[:1]
r_author = best_profile.tolist() + [best_profile[0]]

fig2, ax2 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax2.set_theta_offset(np.pi/2)
ax2.set_theta_direction(-1)
ax2.set_rmax(1.0)
ax2.set_rticks([0.2,0.4,0.6,0.8,1.0])
ax2.set_rlabel_position(135)
ax2.set_thetagrids(np.degrees(base_angles), Topic_names)
ax2.plot(theta_author, r_author, color='green', linewidth=2)
ax2.fill(theta_author, r_author, color='green', alpha=0.25)
ax2.set_title(f"Radar de l’auteur proposé : {best_author}", va='bottom')
plt.tight_layout()
plt.show()

author_df = pd.DataFrame({"Topic": Topic_names, "Valeur": best_profile})
author_df.to_csv("radar_projections_auteur.csv", index=False, encoding='utf-8')
print(f"\n→ Projections de {best_author} sauvegardées dans 'radar_projections_auteur.csv'")

print("\n✓ Pipeline terminée !")
