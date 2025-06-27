

from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re



Topic_names=[]

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

def to_2d(x):
    return x if len(x.shape) == 2 else x.reshape(1, -1)


def supprimer_stopwords(texte, langue='english'):
    mots = word_tokenize(texte)
    stop_words = set(stopwords.words(langue))
    mots_filtrés = [mot for mot in mots if mot.lower() not in stop_words]
    return ' '.join(mots_filtrés)

def build_embedding_text_for_domain(domain_name, subthemes):
    """
    Construire un texte riche pour un domaine en concaténant:
    - Le nom du domaine
    - Les noms des sous-thèmes
    - Les keywords des sous-thèmes
    """
    texts = [domain_name]
    for subtheme, infos in subthemes.items():
        texts.append(subtheme)
        keywords = infos.get("keywords", [])
        texts.extend(keywords)
    return " ".join(texts)

def build_embedding_text_for_subtheme(subtheme_name, infos):
    """
    Construire un texte riche pour un sous-thème en concaténant:
    - Le nom du sous-thème
    - Ses keywords
    """
    keywords = infos.get("keywords", [])
    return " ".join([subtheme_name] + keywords)

def get_topic_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0)

def extract_embeddings_from_dic(dic, tokenizer, model):
    """
    Pour chaque domaine et chaque sous-thème dans dic:
    - Calculer un embedding
    - Retourner deux listes: embeddings_domains, embeddings_subthemes
    Et leurs labels correspondants
    """
    domain_labels = []
    domain_embeddings = []
    subtheme_labels = []
    subtheme_embeddings = []

    for domain_name, subthemes in dic.items():
        domain_text = build_embedding_text_for_domain(domain_name, subthemes)
        domain_emb = get_topic_embedding(domain_text, tokenizer, model)
        domain_labels.append(domain_name)
        domain_embeddings.append(domain_emb)

        for subtheme_name, infos in subthemes.items():
            subtheme_text = build_embedding_text_for_subtheme(subtheme_name, infos)
            subtheme_emb = get_topic_embedding(subtheme_text, tokenizer, model)
            subtheme_labels.append(subtheme_name)
            subtheme_embeddings.append(subtheme_emb)

    return (domain_labels, torch.stack(domain_embeddings),
            subtheme_labels, torch.stack(subtheme_embeddings))


def main():
    global dic

    filepath_theses = "../web_scraping/s2a_thèses_1.txt"
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
    

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    domain_labels, emb_domains, subtheme_labels, emb_subthemes = extract_embeddings_from_dic(dic, tokenizer, model)

    for topic_id in range(lda_model.num_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=10)
        mots_cles = " ".join([mot for mot, _ in topic_terms])
        print(f"\n🧠 Topic {topic_id}: {mots_cles}")

        topic_emb = get_topic_embedding(mots_cles, tokenizer, model)

        sim_domains = cosine_similarity(to_2d(topic_emb.numpy()), emb_domains.numpy())[0]
        sim_subthemes = cosine_similarity(to_2d(topic_emb.numpy()), emb_subthemes.numpy())[0]

        top_domain_idx = sim_domains.argsort()[::-1][:2]  # top 2 domaines
        top_subtheme_idx = sim_subthemes.argsort()[::-1][:3]  # top 3 sous-thèmes
        Topic_names.append[top_domain_idx[-1]]
        print("Top Domaines possibles:")
        for idx in top_domain_idx:
            print(f" - {domain_labels[idx]} (score: {sim_domains[idx]:.3f})")

        print("Top Sous-thèmes possibles:")
        for idx in top_subtheme_idx:
            print(f" - {subtheme_labels[idx]} (score: {sim_subthemes[idx]:.3f})")


if __name__ == "__main__":
    main()
