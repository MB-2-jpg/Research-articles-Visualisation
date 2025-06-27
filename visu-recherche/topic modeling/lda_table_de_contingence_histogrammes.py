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
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    pattern = re.findall(
    r"Thèse\s*(\d+)\s*:(.*?)\s*(?:@members:\s*(.*?))?(?=\s+Thèse|\Z)", 
    contenu, 
    re.DOTALL)

    # Formatage en liste de dictionnaires
    theses = []
    for num, title, members in pattern:
        theses.append({
        "numero": int(num),
        "titre": title.strip(),
        "membres": [m.strip() for m in members.split(",")] if members else []
        })

    
    # Prétraitement
    documents = [thèse['titre'] for thèse in theses]
    membres = [thèse['membres'] for thèse in theses]
    print(membres[:5])
    print(documents[:5])
    texts = [
        simple_preprocess(supprimer_stopwords(doc))
        for doc in documents
    ]

    # Dictionnaire et corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

'''
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


'''
# 4. Entraînement du modèle LDA pour assigner à chaque document un topic dominant

num_topics = 12  # Choisissez le nombre de topics que vous souhaitez

lda_model = LdaModel(corpus=corpus,
                             id2word=dictionary,
                             num_topics=num_topics,
                             random_state=42,
                             passes=10,
                             iterations=100)

L = [0]

for bow in corpus:
    topic_distribution = lda_model.get_document_topics(bow)
    # Récupère le topic avec la plus haute probabilité
    dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
    L.append(dominant_topic)

"""

# Affichage des topics pour un numbre de topics donné
lda_model = models.LdaModel(corpus=corpus,
                            id2word=dictionary,
                            num_topics=13,
                            random_state=42,
                            passes=10,
                            alpha='auto',
                            eta='auto')


vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

pyLDAvis.save_html(vis_data, 'lda_visualisation_13.html')
"""

#Tableau de contingence : 
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


# Accumulateur (membre → topic → somme)
acc = defaultdict(lambda: defaultdict(float))

# Remplissage : on ajoute le coefficient du membre pour le topic assigné à chaque doc
for doc_id, topic_id in enumerate(L[1:]):
    for membre in membres[doc_id]:
        acc[membre][topic_id] += 1

acc_filtered = {membre: topics for membre, topics in acc.items() if len(topics) > 5}
print(acc_filtered)
  
# Transformation en DataFrame
df_filtered = pd.DataFrame.from_dict(acc_filtered, orient='index').fillna(0).sort_index()
df=pd.DataFrame.from_dict(acc, orient='index').fillna(0).sort_index()

# Table de contingence 
df.columns.name = 'Topic'
df.index.name = 'Membre'
# Ajuster les options d'affichage pour montrer toutes les lignes et colonnes
pd.set_option('display.max_rows', None)  # Afficher toutes les lignes
pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
pd.set_option('display.width', None)  # Pas de limite de largeur
pd.set_option('display.max_colwidth', None)  # Pas de limite de largeur des colonnes

print(df)


# Histogramme : Nombre de contributeurs par Topic
topic_counts = df.sum(axis=0)  # Somme des coefficients par topic, représentant le nombre de membres par topic
plt.figure(figsize=(10, 6))
sns.barplot(x=topic_counts.index, y=topic_counts.values, palette='viridis')
plt.title('Histogramme : Nombre de Membres par Topic')
plt.xlabel('Topic')
plt.ylabel('Nombre de Membres')
plt.show()

# Histogramme : Distribution de la contribution topics par membre filtré avec au moins 5 contributions.
df_filtered_columns = df_filtered.columns.astype(int)

# Palette de couleurs : une couleur par topic
cmap = plt.colormaps.get_cmap('tab20')  # juste le nom
colors = [cmap(i / len(df_filtered.columns)) for i in range(len(df_filtered.columns))]


# Initialisation du graphique
fig, ax = plt.subplots(figsize=(15, 8))

# Empilage des barres
bottom = np.zeros(len(df_filtered))
x = np.arange(len(df_filtered))
for i, topic in enumerate(df.columns):
    heights = df_filtered[topic].values
    ax.bar(x, heights, bottom=bottom, color=colors[i], label=f'Topic {topic}')
    bottom += heights  # empilement

# Mise en forme
ax.set_xticks(x)
ax.set_xticklabels(df_filtered.index, rotation=90)
ax.set_ylabel("Nombre de contributions")
ax.set_title("Contributions par membre ")
ax.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()