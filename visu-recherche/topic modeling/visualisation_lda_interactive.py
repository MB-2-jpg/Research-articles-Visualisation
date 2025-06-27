

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


def langageDetection(string_txt):
    lang = detect(string_txt)
    if lang == "en" :
        lang= "english"
    elif lang == "fr":
        lang = "french"
    return lang


def nettoyer_texte(texte):
    # Forcer l'anglais si tes textes sont majoritairement en anglais
    langue = langageDetection(texte)
    if langue not in ['english', 'french']:
        langue = 'english'  # fallback

    stop_words = set(stopwords.words(langue))
    stop_words.update(['the','of','and','in','is','to','for','that','et','we','by','let','xi','set','log','with','that','are','it','our','this','kx','rx','be','ad','such','one'
    ,'two'])

    # Preprocessing avec simple_preprocess directement
    mots = simple_preprocess(texte, deacc=True)
    mots_filtrés = [mot for mot in mots if mot not in stop_words and len(mot)>2]
    
    return mots_filtrés


# Chargement du fichier
if __name__=='__main__':
    with open("web_scraping/s2a_thèses.txt", 'r', encoding='utf-8') as file:

        contenu = file.read()

# Séparation des documents
    documents = contenu.split("\n Thèse ")

# Prétraitement
    texts = [
        nettoyer_texte(doc)
        for doc in documents
        if len(doc.strip()) > 10
    ]

# Dictionnaire et corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

# Calcul des scores de cohérence
    coherence_scores = []
    num_topics= 12

    lda_model = LdaModel(corpus=corpus,
                             id2word=dictionary,
                             num_topics=12,
                             random_state=42,
                             passes=10,
                             iterations=100)
        
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    score = coherence_model.get_coherence()
    coherence_scores.append(score)

    print(f"Topics: {num_topics} | Cohérence: {score:.4f}")


# 5. Affichage des topics
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

    pyLDAvis.save_html(vis_data, 'lda_visualisation__12.html')
