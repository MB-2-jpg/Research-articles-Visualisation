from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm

stop_words_eng = stopwords.words('english') +stopwords.words('frensh')


vectorizer = TfidfVectorizer(
    stop_words=stop_words_eng,
    max_df=10000,
    min_df=1,
    max_features=1000
)


def conversion_txt_string(nom_fichier):
    try:
        with open(nom_fichier, "r", encoding="utf-8") as fichier:
            contenu = fichier.read()
        return contenu
    except FileNotFoundError:
        return "Erreur : fichier non trouvé."
    except Exception as e:
        return f"Erreur : {e}"
tempo = conversion_txt_string("../web_scraping/s2a_thèses_with_pdf.txt")

documents = tempo.split("\n\n\n\n\n Thèse ")



with open("../web_scraping/s2a_thèses_with_pdf.txt", "r", encoding="utf-8") as f:
    contenu = f.read()

words = word_tokenize(contenu)

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Optionally remove punctuation
words = [word for word in words if word not in string.punctuation]

# Remove stopwords
filtered_words = [word for word in words if word.lower() not in stop_words]

# Join words back into a string (optional)
contenu = ' '.join(filtered_words)

words = word_tokenize(contenu)

# Load English stopwords
stop_words = set(stopwords.words('french'))

# Optionally remove punctuation
words = [word for word in words if word not in string.punctuation]

# Remove stopwords
filtered_words = [word for word in words if word.lower() not in stop_words]

# Join words back into a string (optional)
contenu = ' '.join(filtered_words)


print(len(contenu))
documents = contenu.split("Thèse")

print(len(documents))
X = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
#print(feature_names)




dense_matrix = X.toarray()


feature_names = vectorizer.get_feature_names_out()


df_tfidf = pd.DataFrame(dense_matrix, columns=feature_names)
#print(df_tfidf)


# Nombre de topics à extraire
n_topics = 13

# Appliquer NMF sur la matrice TF-IDF
nmf_model = NMF(n_components=n_topics, random_state=42)
W = nmf_model.fit_transform(X)  # Matrice des poids des documents par topic
H = nmf_model.components_  

# Récupérer les mots les plus significatifs par topic
n_top_words = 10

for topic_idx, topic in enumerate(H):
    print(f"\nTopic {topic_idx + 1}:")
    top_words_idx = topic.argsort()[:-n_top_words - 1:-1]  # Indices des top mots
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
    print(", ".join(top_words))


# Afficher les proportions des topics dans chaque document
for i, doc_topic in enumerate(W):
    print(f"\nDocument {i + 1}:")
    for topic_idx, prob in enumerate(doc_topic):
        print(f"Topic {topic_idx + 1}: {prob:.3f}")



# Nombre de mots à afficher par topic
n_top_wordwords = word_tokenize(contenu)

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Optionally remove punctuation
words = [word for word in words if word not in string.punctuation]

# Remove stopwords
filtered_words = [word for word in words if word.lower() not in stop_words]

# Join words back into a string (optional)
contenu = ' '.join(filtered_words)

# Affichage des nuages de mots pour tous les topics
fig, axes = plt.subplots(1, n_topics, figsize=(20, 5))  # Crée un graphique avec n_topics sous-graphes

for topic_idx, topic in enumerate(H):
    # Extraire les mots les plus importants par topic
    top_words_idx = topic.argsort()[:-n_top_words - 1:-1]  # Indices des top mots
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
    top_word_weights = topic[top_words_idx]  # Poids des mots associés au topic
    
    # Créer un dictionnaire de mots et poids pour le nuage de mots
    word_freq = dict(zip(top_words, top_word_weights))
    
    # Créer le nuage de mots
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    # Afficher le nuage de mots sur le graphique correspondant
    axes[topic_idx].imshow(wordcloud, interpolation="bilinear")
    axes[topic_idx].axis('off')
    axes[topic_idx].set_title(f"Topic {topic_idx + 1}")

# Afficher le graphique avec tous les nuages de mots
plt.tight_layout()
plt.show()
topics = []
for topic_idx, topic in enumerate(H):
    top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    topics.append(top_words)

# Convertir les documents en liste de mots (pour `gensim`)
texts = [doc.split() for doc in documents]

# Créer un dictionnaire Gensim
dictionary = gensim.corpora.Dictionary(texts)

# Créer le corpus Gensim
corpus = [dictionary.doc2bow(text) for text in texts]

# Calcul de la cohérence avec `gensim`
coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model.get_coherence()

print(f"Cohérence des topics: {coherence_score}")


coherence_scores = []
topic_range = range(1, 21)  # de 1 à 15 topics

for n_topics in tqdm(topic_range):
    # Appliquer NMF
    nmf_model = NMF(n_components=n_topics, random_state=42)
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_
    
    # Extraire les top mots par topic
    topics = []
    for topic in H:
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(top_words)
    
    # Calcul de cohérence
    coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()
    coherence_scores.append(coherence)

# Tracer la courbe
plt.figure(figsize=(10, 6))
plt.plot(topic_range, coherence_scores, marker='o')
plt.title("Score de cohérence en fonction du nombre de topics")
plt.xlabel("Nombre de topics")
plt.ylabel("Score de cohérence (c_v)")
plt.xticks(topic_range)
plt.grid(True)
plt.show()
