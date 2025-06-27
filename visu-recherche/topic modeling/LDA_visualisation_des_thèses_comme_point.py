from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel, CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
from collections import defaultdict
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def supprimer_stopwords(texte, langue1='french',langue2='english'):
    mots = word_tokenize(texte)
    stop_words = set(stopwords.words(langue1))
    mots_filtrés = [mot for mot in mots if mot.lower() not in stop_words]
    stop_words = set(stopwords.words(langue2))
    mots_filtrés = [mot for mot in mots if mot.lower() not in stop_words]

    return ' '.join(mots_filtrés)

if __name__ == '__main__':
    # Chargement du texte
    with open("../web_scraping/s2a_thèses_with_pdf.txt", 'r', encoding='utf-8') as file:
        contenu = file.read()

    documents = contenu.split("\n\n\n\n\n Thèse ")
    texts = [simple_preprocess(supprimer_stopwords(doc)) for doc in documents if len(doc.strip()) > 10]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    num_topics = 14  # Choisis ici un nombre de topics optimal (ou après analyse de la courbe)

    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         random_state=42,
                         passes=10,
                         iterations=100)

    # Obtenir les distributions de topics pour chaque document
    def get_topic_vectors(model, corpus, num_topics):
        topic_distributions = []
        dominant_topics = []
        for bow in corpus:
            dist = model.get_document_topics(bow, minimum_probability=0.0)
            sorted_dist = sorted(dist, key=lambda x: x[0])
            vector = [prob for _, prob in sorted_dist]
            topic_distributions.append(vector)
            # topic dominant (celui avec prob max)
            dominant_topic = max(dist, key=lambda x: x[1])[0]
            dominant_topics.append(dominant_topic)
        return np.array(topic_distributions), dominant_topics

    topic_vectors, dominant_topics = get_topic_vectors(lda_model, corpus, num_topics)

    # Réduction en 2D
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(topic_vectors)
    
    
    
    S = tokenizer.tokenize(S)
 
Auteurs = []
marker = False
i =0
while i< len(S) :
    aut=[]
    word = S[i]
    if word == "@Auteurs" and marker == False:
        marker = True
        i+=1
    if marker == True :
        if word =="@Auteurs":
            marker == False
            Auteurs.append(aut)
            aut =[]
        else :
        
            name =""
            while S[i]!= ",":
                name += " " +S[i]
                 
                S.pop(i)
            
            S.insert(i, name)          
            i+=1
            aut.append(name)
    # Associe chaque topic à une liste d’indices de documents
    documents_par_topic = defaultdict(list)
    Dictionnaire_auteurs_thèmes={}
    for i, topic_id in enumerate(dominant_topics):
        documents_par_topic[topic_id].append(i)
        for j in Auteurs[i]:
            if not(j in Dictionnaire_auteurs_thèmes.keys):
              Dictionnaire_auteurs_thèmes[j]=[i]
            else :
                Dictionnaire_auteurs_thèmes[j].append(i)


    print(documents_par_topic)



    # Mots-clés dominants par topic
    topic_keywords = {
        topic_id: ", ".join([word for word, _ in lda_model.show_topic(topic_id, topn=2)])
        for topic_id in range(num_topics)
    }

    # Affichage
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                          c=dominant_topics, cmap='tab10', s=40, alpha=0.8)

    # Ajouter les labels des mots-clés (près du barycentre de chaque topic)
    for topic_id in range(num_topics):
        indices = [i for i, t in enumerate(dominant_topics) if t == topic_id]
        if indices :
            x_mean = np.mean(embedding[indices, 0])
            y_mean = np.mean(embedding[indices, 1])
            plt.text(x_mean, y_mean,
                     f'Topic {topic_id}\n{topic_keywords[topic_id]}',
                     fontsize=10, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    plt.title("Thèses visualisées selon leur distribution thématique (colorées par topic dominant)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.colorbar(scatter, ticks=range(num_topics), label='Topic ID')
    plt.show()
