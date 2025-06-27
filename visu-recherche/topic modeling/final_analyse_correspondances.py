# -*- coding: utf-8 -*-
"""
AFC Topics–Auteurs avec nomination automatique des libellés (domaines & sous-thèmes),
et fallback unique en mode « mot-clé principal » si SciBERT ne peut pas se charger
(évite que deux topics aient le même nom).
Adapté pour extraire correctement les auteurs à partir d’un fichier de la forme :
    Thèse 1 : Titre…
    Auteur 1, Auteur 2, …
    Auteur_n+1, Auteur_n+2

    To cite this version: …
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


# --- 1. Fonctions utilitaires ---------------------------------------------------

def to_2d(x):
    """
    Assure la forme 2D pour cosine_similarity, qu'il s'agisse d'un torch.Tensor ou d'un numpy.ndarray.
    """
    if isinstance(x, torch.Tensor):
        return x if x.ndim == 2 else x.unsqueeze(0)
    import numpy as np
    if isinstance(x, np.ndarray):
        return x if x.ndim == 2 else x.reshape(1, -1)
    return x

def supprimer_stopwords(texte, langue='english'):
    """
    Enlève les stopwords d'un texte.
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


# --- 2. Définition des domaines et sous-thèmes ----------------------------------

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
            "description": "Techniques to reduce noise in signals",
            "keywords": ["denoising", "Wiener filter", "median filter", "thresholding"]
        },
        "Sampling and Quantization": {
            "description": "Converting continuous signals to discrete",
            "keywords": ["Nyquist rate", "aliasing", "quantization error", "ADC", "DAC"]
        },
        "Adaptive Signal Processing": {
            "description": "Filters that adjust parameters based on signal properties",
            "keywords": ["LMS algorithm", "RLS algorithm", "adaptive filtering"]
        },
        "Signal Detection": {
            "description": "Identifying signals in noise",
            "keywords": ["matched filter", "energy detection", "false alarm rate", "detection probability"]
        },
        "Image Processing": {
            "description": "Manipulation and analysis of images",
            "keywords": ["edge detection", "filtering", "segmentation", "morphological operations", "Fourier transform"]
        },
        "Audio Signal Processing": {
            "description": "Processing of sound signals",
            "keywords": ["speech recognition", "echo cancellation", "noise suppression", "audio coding", "speech processing", "machine listening", "MIR", "music information retrieval"]
        },
        "Distributed Processing": {
            "description": "Signal processing on distributed systems and parallel architectures",
            "keywords": ["distributed algorithms", "parallel processing", "edge computing", "cloud processing"]
        }
    },
    "Machine Learning": {
        "Supervised Learning": {
            "description": "Learning from labeled data",
            "keywords": ["classification", "regression", "training set", "test set", "validation set"]
        },
        "Unsupervised Learning": {
            "description": "Learning patterns from unlabeled data",
            "keywords": ["clustering", "dimensionality reduction", "association rules", "anomaly detection", "data decomposition"]
        },
        "Reinforcement Learning": {
            "description": "Learning by interacting with environment and receiving rewards",
            "keywords": ["agent", "environment", "policy", "reward", "Q-learning", "exploration-exploitation"]
        },
        "Deep Learning": {
            "description": "Learning using deep neural networks",
            "keywords": ["neural networks", "convolutional neural networks", "recurrent neural networks", "backpropagation", "activation function"]
        },
        "Decision Trees": {
            "description": "Tree-structured models for classification and regression",
            "keywords": ["node", "leaf", "splitting criterion", "entropy", "Gini index"]
        },
        "Random Forests": {
            "description": "Ensemble of decision trees",
            "keywords": ["bagging", "feature importance", "out-of-bag error"]
        },
        "Support Vector Machines": {
            "description": "Margin-based classifiers",
            "keywords": ["kernel", "margin", "support vectors", "hyperplane", "C parameter", "kernel methods"]
        },
        "Clustering Algorithms": {
            "description": "Grouping similar data points",
            "keywords": ["k-means", "hierarchical clustering", "DBSCAN", "centroid", "dendrogram"]
        },
        "Dimensionality Reduction": {
            "description": "Reducing number of features",
            "keywords": ["PCA", "t-SNE", "LDA", "feature extraction", "representation learning", "multiview learning", "multimodal processing"]
        },
        "Cross-validation": {
            "description": "Model validation technique",
            "keywords": ["k-fold", "leave-one-out", "train-test split"]
        },
        "Hyperparameter Optimization": {
            "description": "Finding best model parameters",
            "keywords": ["grid search", "random search", "Bayesian optimization", "stochastic optimization", "convex analysis", "optimization theory", "optimization algorithms", "stochastic algorithms", "optimal control"]
        },
        "Feature Engineering": {
            "description": "Creating features from raw data",
            "keywords": ["feature selection", "feature extraction", "scaling", "normalization"]
        },
        "Probabilistic Models": {
            "description": "Models based on probability distributions",
            "keywords": ["Hidden Markov Model", "Conditional Random Field", "Gaussian Mixture Model", "Bayesian networks", "Bayesian methods", "machine learning probability"]
        },
        "Natural Language Processing": {
            "description": "Processing and analysis of text data",
            "keywords": ["tokenization", "stemming", "lemmatization", "sentiment analysis", "word embeddings", "computational linguistics"]
        },
        "Sentiment Analysis": {
            "description": "Determining sentiment in text",
            "keywords": ["positive", "negative", "neutral", "opinion mining"]
        },
        "Machine Learning Fairness and Robustness": {
            "description": "Ensuring models are fair, interpretable, and robust",
            "keywords": ["fairness", "bias", "explainable AI", "interpretable AI", "robust statistics", "bias mitigation"]
        },
        "Graphs and Graph-based Learning": {
            "description": "Learning on graph-structured data",
            "keywords": ["graph neural networks", "graph embeddings", "structured prediction", "graph supervised learning"]
        },
        "Large-scale and Distributed Machine Learning": {
            "description": "Techniques for handling large datasets and distributed computing",
            "keywords": ["distributed computing", "parallel processing", "large-scale data analysis", "machine learning distributed processing"]
        },
        "Optimal Transport": {
            "description": "Mathematical framework for comparing probability distributions",
            "keywords": ["optimal transport", "Wasserstein distance", "earth mover's distance"]
        },
        "Inverse Problems": {
            "description": "Recovering inputs from observed outputs in ill-posed problems",
            "keywords": ["inverse problems", "regularization", "deconvolution"]
        },
        "Human Pose Estimation": {
            "description": "Detecting and analyzing human poses from images or videos",
            "keywords": ["pose estimation", "keypoint detection", "computer vision"]
        },
        "Machine Listening and Music Information Retrieval (MIR)": {
            "description": "Audio signal processing and analysis for music and sound",
            "keywords": ["machine listening", "music information retrieval", "MIR", "audio classification", "speech processing", "handwriting recognition"]
        },
        "Speech Processing": {
            "description": "Processing spoken language signals",
            "keywords": ["speech recognition", "speech synthesis", "speaker identification", "voice activity detection"]
        },
        "Computer Vision": {
            "description": "Extracting information from images and videos",
            "keywords": ["image recognition", "object detection", "segmentation", "feature extraction", "human pose estimation"]
        },
        "Rankings and Preferences": {
            "description": "Models and algorithms for ranking data and preference learning",
            "keywords": ["ranking", "preference learning", "recommendation systems"]
        }
    }
}

# --- 3. Paramètres utilisateur ------------------------------------------------

THESIS_FILE    = "../pdf_scraping/articles_de_publications_permanents_formatted.txt"
NUM_TOPICS     = 6   # Ajustez si besoin
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


# --- 4. Lecture & parsing des thèses (titres + auteurs) ------------------------

with open(THESIS_FILE, 'r', encoding='utf-8') as f:
    contenu_theses = f.read().strip()

# On repère toutes les occurrences "Thèse N : Titre…"
header_pattern = re.compile(r"Thèse\s+(\d+)\s*:\s*(.+)")
lines = contenu_theses.split("\n")

theses = []
i = 0
while i < len(lines):
    # Cherche la ligne "Thèse N : Titre"
    match = header_pattern.match(lines[i])
    if match:
        num   = int(match.group(1))
        title = match.group(2).strip()

        # Collecter ensuite les lignes suivantes jusqu'à une ligne vide ou "To cite"
        author_lines = []
        j = i + 1
        while j < len(lines):
            line = lines[j].strip()
            if (not line) or line.lower().startswith("to cite"):
                break
            author_lines.append(line)
            j += 1

        # Concaténer toutes les lignes d'auteurs et les séparer par virgule
        author_block = " ".join(author_lines)
        # Les noms se terminent tous par une virgule (sauf peut-être le dernier), on split donc sur ","
        raw_names = [n.strip() for n in author_block.split(",") if n.strip()]
        # Filtrer uniquement ceux présents dans TARGET_AUTHORS
        auteurs = [n for n in raw_names if n in TARGET_AUTHORS]

        theses.append({
            "numero": num,
            "titre": title,
            "membres": auteurs
        })
        # Passer j à la prochaine ligne après les auteurs
        i = j
    else:
        i += 1

# Vérifier qu’on a effectivement extrait au moins une thèse avec auteurs
print(f">>> Nombre de thèses extraites : {len(theses)}")
non_vides = sum(1 for t in theses if t["membres"])
print(f">>> Parmi elles, {non_vides} comportent au moins un auteur cible.")

if non_vides == 0:
    raise RuntimeError(
        "❌ Aucune thèse ne contenait un auteur de TARGET_AUTHORS. "
        "Vérifiez le format du fichier ou la liste TARGET_AUTHORS."
    )

# Préparer la liste des documents (titres uniquement) pour LDA
documents_theses = [t["titre"] for t in theses]
texts_theses = [
    simple_preprocess(supprimer_stopwords(txt))
    for txt in documents_theses
]

dictionary_theses = corpora.Dictionary(texts_theses)
dictionary_theses.filter_extremes(no_below=2, no_above=0.5)
corpus_theses = [dictionary_theses.doc2bow(text) for text in texts_theses]

# --- 5. Entraînement du modèle LDA --------------------------------------------

lda_model = LdaModel(
    corpus=corpus_theses,
    id2word=dictionary_theses,
    num_topics=NUM_TOPICS,
    random_state=42,
    passes=10,
    iterations=100
)

# --- 6. Chargement SciBERT ou fallback sur mot-clé principal ------------------

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
    # Extraire embeddings SciBERT pour domaines & sous-thèmes
    domain_labels, emb_domains, subtheme_labels, emb_subthemes = extract_embeddings_from_dic(dic, tokenizer, model)

    # --- 7. Détermination des candidats (label, score) pour chaque topic ----
    topic_candidates = []  # liste de listes de tuples (label, score)
    for topic_id in range(lda_model.num_topics):
        terms = lda_model.show_topic(topic_id, topn=10)
        mots_cles = " ".join([mot for mot, _ in terms])
        topic_emb = get_topic_embedding(mots_cles, tokenizer, model)

        # Similarités aux domaines
        sim_domains = cosine_similarity(
            to_2d(topic_emb.numpy()),
            emb_domains.numpy()
        )[0]

        # Similarités aux sous-thèmes
        sim_subthemes = cosine_similarity(
            to_2d(topic_emb.numpy()),
            emb_subthemes.numpy()
        )[0]

        # Construire liste fusionnée : (label, score) pour domaines et sous-thèmes
        scored = []
        scored += list(zip(domain_labels, sim_domains))
        scored += list(zip(subtheme_labels, sim_subthemes))
        # Trier par score décroissant
        scored.sort(key=lambda x: x[1], reverse=True)
        topic_candidates.append(scored)

    # --- 8. Appariement glouton global pour labels uniques ------------------

    # Rassembler tous les triplets (topic_id, label, score)
    all_triples = []
    for topic_id, candidates in enumerate(topic_candidates):
        for label, score in candidates:
            all_triples.append((topic_id, label, score))

    # Trier par score décroissant
    all_triples.sort(key=lambda x: x[2], reverse=True)

    assigned_label = {}  # topic_id -> label
    used_labels    = set()

    # Attribution gloutonne
    for topic_id, label, score in all_triples:
        if topic_id not in assigned_label and label not in used_labels:
            assigned_label[topic_id] = label
            used_labels.add(label)

    # S’assurer que chaque topic ait un label
    for topic_id in range(NUM_TOPICS):
        if topic_id not in assigned_label:
            # Attribuer le premier candidat non utilisé
            for label, _ in topic_candidates[topic_id]:
                if label not in used_labels:
                    assigned_label[topic_id] = label
                    used_labels.add(label)
                    break
            else:
                # En dernier recours, on prend le premier candidat brut
                fallback_label = topic_candidates[topic_id][0][0]
                assigned_label[topic_id] = fallback_label
                used_labels.add(fallback_label)

    # Construire Topic_names dans l’ordre des IDs
    Topic_names = [assigned_label[i] for i in range(NUM_TOPICS)]

    print("\n>> Attribution finale des labels (topic_id → label) :")
    for i, lbl in enumerate(Topic_names):
        print(f"   {i} → {lbl}")

else:
    # --- Bloc fallback unique : mot-clé principal + prévention des doublons ---
    # On récupère pour chaque topic les deux premiers mots
    raw_labels = [
        lda_model.show_topic(topic_id, topn=2)  # deux premiers mots
        for topic_id in range(lda_model.num_topics)
    ]
    # raw_labels[i][0][0] = premier mot du topic i, raw_labels[i][1][0] = deuxième mot si disponible

    # Étape 1 : lister tous les premiers mots
    initial_labels = [terms[0][0] for terms in raw_labels]
    counter = Counter(initial_labels)

    # Étape 2 : créer Topic_names en corrigeant les doublons
    Topic_names = []
    used = set()
    for topic_id, terms in enumerate(raw_labels):
        premier = terms[0][0]
        deuxieme = terms[1][0] if len(terms) > 1 else None

        # Si le premier mot n'est pas dupliqué ET pas encore utilisé, on l'utilise
        if counter[premier] == 1 and premier not in used:
            Topic_names.append(premier)
            used.add(premier)
        else:
            # Sinon, on tente le deuxième mot
            if deuxieme and deuxieme not in used:
                Topic_names.append(deuxieme)
                used.add(deuxieme)
            else:
                # En dernier recours, concaténer l'ID pour garantir l'unicité
                fallback = f"{premier}_{topic_id}"
                Topic_names.append(fallback)
                used.add(fallback)

    print("\n>> En mode fallback, on évite les doublons ainsi :")
    for i, lbl in enumerate(Topic_names):
        print(f"   {i} → {lbl}")


# --- 9. Reconstruction de doc_authors (filtrée) -----------------------------

doc_authors = []
for t in theses:
    # On garde uniquement les membres figurant dans TARGET_AUTHORS
    membres = t["membres"]
    doc_authors.append(membres)

filtered_topics  = []
filtered_authors = []
for idx, auteurs in enumerate(doc_authors):
    if auteurs:
        dominant = max(lda_model.get_document_topics(corpus_theses[idx]), key=lambda x: x[1])[0]
        filtered_topics.append(dominant)
        filtered_authors.append(auteurs)

# --- 10. Construction de la matrice TopicID × Auteur (complète) --------------

all_authors = sorted({a for auths in filtered_authors for a in auths})
cont_numeric = pd.DataFrame(
    0,
    index=list(range(NUM_TOPICS)),
    columns=all_authors
)

for topic_id, authors in zip(filtered_topics, filtered_authors):
    for a in authors:
        cont_numeric.at[topic_id, a] += 1

# Supprimer seulement les colonnes (auteurs) sans occurrence
cont_numeric = cont_numeric.loc[:, cont_numeric.sum(axis=0) > 0]

# --- 11. Vérification de cont_numeric avant d’appeler l’AFC ---------------

if cont_numeric.shape[1] == 0:
    print("\n❌ ERREUR : après filtrage, cont_numeric n’a AUCUNE colonne.")
    print("             → Cela signifie qu’aucun membre dans TARGET_AUTHORS n’a été trouvé.")
    print("             Vérifiez :")
    print("               • la liste TARGET_AUTHORS")
    print("               • le parsing du fichier articles_de_publications…")
    sys.exit(1)

# --- 12. Réaliser l’AFC sur cont_numeric --------------------------------------

ca   = CA(n_components=2, random_state=42).fit(cont_numeric)
rows = ca.row_coordinates(cont_numeric)    # DataFrame : index=topic_id, colonnes=(Dim1, Dim2)
cols = ca.column_coordinates(cont_numeric) # DataFrame : index=auteur, colonnes=(Dim1, Dim2)

print("\nIndex de cont_numeric (topic_ids) utilisé pour l’AFC :", cont_numeric.index.tolist())

# --- 13. Projection 2D et visualisation -------------------------------------

plt.figure(figsize=(14, 10))

# — Topics (cercles verts) —
plt.scatter(rows.iloc[:, 0], rows.iloc[:, 1], edgecolors='green', facecolors='none', label='Topics')
for i in rows.index:
    label = Topic_names[i]
    if i>4  or i==7:
          plt.text(rows.at[i, 0], rows.at[i, 1], label, fontsize=15, color='green')
    else:
        plt.text(rows.at[i, 0], rows.at[i, 1], label, fontsize=11, color='green')

# — Auteurs (triangles rouges) —
plt.scatter(cols.iloc[:, 0], cols.iloc[:, 1],
             edgecolors='red', facecolors='none', label='Auteurs')
j=0
for author in cols.index:
    j+=1
    if j==4  or j==3 or j==6 or j==8 or j==1:
           plt.text(cols.at[author, 0], cols.at[author, 1], author, fontsize=15, color='red')

    elif j!= 2:

         plt.text(cols.at[author, 0], cols.at[author, 1], author, fontsize=8, color='red')


plt.axhline(0, linestyle='--', color='gray')
plt.axvline(0, linestyle='--', color='gray')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("AFC – Topics nommés par domaine/sous-thème & Auteurs (labels uniques)")
plt.legend(loc='best')
plt.tight_layout()
plt.show()
