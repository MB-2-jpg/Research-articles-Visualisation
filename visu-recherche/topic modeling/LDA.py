
from gensim import corpora, models
from gensim.utils import simple_preprocess
import os
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def supprimer_stopwords(texte, langue='english'):
    mots = word_tokenize(texte)
    stop_words = set(stopwords.words(langue))
    mots_filtrés = [mot for mot in mots if mot.lower() not in stop_words]
    return ' '.join(mots_filtrés)

if __name__ == '__main__':
    # Chargement du fichier
    with open("../pdf_scraping/articles_de_publications_permanents.txt", 'r', encoding='utf-8') as file:
        contenu = file.read()

    # Séparation des documents
    documents = contenu.split("\n Thèse ")

    # Prétraitement
    texts = [
        simple_preprocess(supprimer_stopwords(doc))
        for doc in documents
        if len(doc.strip()) > 10
    ]

    # Dictionnaire et corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Calcul des scores de cohérence
    coherence_scores = []
    topic_range = range(5,20)

    for num_topics in topic_range:
        lda_model = LdaModel(corpus=corpus,
                             id2word=dictionary,
                             num_topics=num_topics,
                             random_state=42,
                             passes=10,
                             iterations=100)
        
        coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        score = coherence_model.get_coherence()
        coherence_scores.append(score)
        print(f"Topics: {num_topics} | Cohérence: {score:.4f}")

    # Tracer la courbe
    plt.plot(topic_range, coherence_scores, marker='o')
    plt.xlabel('Nombre de topics')
    plt.ylabel('Score de cohérence (c_v)')
    plt.title('Cohérence vs Nombre de topics')
    plt.grid(True)
    plt.show()






"""

# 4. Entraînement du modèle LDA
lda_model = models.LdaModel(corpus=corpus,
                            id2word=dictionary,
                            num_topics=13,
                            random_state=42,
                            passes=10,
                            alpha='auto',
                            eta='auto')

# 5. Affichage des topics
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

pyLDAvis.save_html(vis_data, 'lda_visualisation_13.html')
"""