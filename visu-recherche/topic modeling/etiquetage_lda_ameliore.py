
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# Lemmatisation
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]

def tokenize_label(label):
    tokens = [w.lower() for w in label.split()]
    return lemmatize_tokens(tokens)

def to_2d(x):
    return x if len(x.shape) == 2 else x.reshape(1, -1)


def supprimer_stopwords(texte, langue='english'):
    mots = word_tokenize(texte)
    stop_words = set(stopwords.words(langue))
    mots_filtrés = [mot for mot in mots if mot.lower() not in stop_words]
    return ' '.join(mots_filtrés)

def run_lda():
    global dic

    filepath_theses = "web_scraping/s2a_thèses_1.txt"
    with open(filepath_theses, 'r', encoding='utf-8') as file:
        contenu = file.read()

    pattern = re.findall(
        r"Thèse\s*(\d+)\s*:(.*?)\s*(?:@members:\s*(.*?))?(?=\s+Thèse|\Z)",
        contenu,
        re.DOTALL
    )

    theses = []
    for num, title, membre_str in pattern:
        theses.append({
            "numero": int(num),
            "titre": title.strip()
        })

    documents = [these['titre'] for these in theses]
    textes = [simple_preprocess(supprimer_stopwords(doc)) for doc in documents]

    dictionary = corpora.Dictionary(textes)
    corpus = [dictionary.doc2bow(text) for text in textes]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=12,
        random_state=42,
        passes=10,
        iterations=100
    )
    return lda_model


def get_topic_word_weights(lda_model, topic_id):
    """
    Récupère tous les mots avec leurs poids pour un topic donné.
    """
    return lda_model.show_topic(topic_id, topn=50)  # topn assez grand pour couvrir le topic

def score_label_against_topic(label_words, topic_word_weights):
    """
    label_words: liste des mots du label (domaine ou sous-thème)
    topic_word_weights: liste de (mot, poids) du topic

    Retourne un score = somme des poids des mots du label présents dans le topic
    """
    # Création d'un dict pour accès rapide poids
    topic_dict = dict(topic_word_weights)
    score = 0.0
    for w in label_words:
        if w in topic_dict:
            score += topic_dict[w]
    return score

def enriched_score_label_vs_topic(label_words, topic_word_weights):
    label_vecs = [word_vectors[w] for w in label_words if w in word_vectors]
    score = 0.0

    for word, weight in topic_word_weights:
        if word in word_vectors:
            word_vec = word_vectors[word]
            sims = [cosine_similarity(to_2d(word_vec), to_2d(lv))[0][0] for lv in label_vecs]
            if sims:
                sim_score = max(sims)
                score += sim_score * weight

    return score

from gensim.models import KeyedVectors

# Exemple avec GloVe (préchargé dans le format word2vec)
word_vectors = KeyedVectors.load_word2vec_format("glove.6B.100d.txt", binary=False)

def embedding_similarity_score(label_words, topic_word_weights):
    label_vecs = [word_vectors[w] for w in label_words if w in word_vectors]
    topic_scores = []

    for word, weight in topic_word_weights:
        if word in word_vectors:
            word_vec = word_vectors[word]
            sims = [cosine_similarity(to_2d(word_vec), to_2d(lv))[0][0] for lv in label_vecs]
            if sims:
                topic_scores.append(max(sims) * weight)  # ou `mean(sims)`
    
    return sum(topic_scores)


def tokenize_label(label):
    """
    Simple tokenisation : découpe label en mots lowercase
    """
    return [w.lower() for w in label.split()]

def assign_labels_to_topics(lda_model, dic, topn=3):
    domaines = list(dic.keys())
    sous_themes = [st for sublist in dic.values() for st in sublist]

    for topic_id in range(lda_model.num_topics):
        topic_words = get_topic_word_weights(lda_model, topic_id)

        # Score pour chaque domaine
        scores_domaines = []
        for d in domaines:
            d_tokens = tokenize_label(d)
            score = score_label_against_topic(d_tokens, topic_words)
            scores_domaines.append((d, score))
        scores_domaines.sort(key=lambda x: x[1], reverse=True)

        # Score pour chaque sous-thème
        scores_sous_themes = []
        for st in sous_themes:
            st_tokens = tokenize_label(st)
            score = score_label_against_topic(st_tokens, topic_words)
            scores_sous_themes.append((st, score))
        scores_sous_themes.sort(key=lambda x: x[1], reverse=True)

        # Affichage
        topic_top_words = " ".join([w for w, p in topic_words[:10]])
        print(f"\n🧠 Topic {topic_id}: {topic_top_words}")
        print("Top Domaines possibles:")
        for d, s in scores_domaines[:topn]:
            print(f" - {d} (score: {s:.3f})")
        print("Top Sous-thèmes possibles:")
        for st, s in scores_sous_themes[:topn]:
            print(f" - {st} (score: {s:.3f})")



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
      "description": "Mathematical representation of signals",
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





lda_model = run_lda()
assign_labels_to_topics(lda_model, dic, topn=3)
