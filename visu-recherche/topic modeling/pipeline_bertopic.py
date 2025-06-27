
# -*- coding: utf-8 -*-
"""
Topic Modeling sans  utilisé la librairie BERTopic 
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import random
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import euclidean
import sentence_transformers
from numpy import dot
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import numpy as np
import json
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import InvertedRBO

#################FONCTIONS AUXILLIAIRES UTILES###########
def dist_cos(a,b): #distance cosinus entre deux vecteur a et b
    return 1- dot(a, b)/(norm(a)*norm(b))

def core_distances(data, nb_voisin):
    nbrs = NearestNeighbors(n_neighbors=nb_voisin, metric='euclidean').fit(data)
    distances, _ = nbrs.kneighbors(data)
    core_distances = distances[:, -1]  # distance au k-ième voisin
    return core_distances

def distance_clusters_avec_densite(cluster1, cluster2, data, core_dists): # calcule de la distance-densité de deux
    distances = []
    for i in cluster1:
        for j in cluster2:
            euclidean = np.linalg.norm(data[i] - data[j])
            temp = max(core_dists[i], core_dists[j], euclidean)  # distance-densité
            distances.append(temp)
    return np.mean(distances) # on envoie la moyenne des distances mais peut etre replace par le min

# Calcul de la distance entre deux clusters



# Étape 1 : Données d'exemple (à remplacer par les documents S2A)


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
documents = documents[1:]


# Étape 2 : Embedding des documents
print("[1/5] Génération des embeddings...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(documents, show_progress_bar=True)

# Étape 3 : Réduction de dimensions avec UMAP
print("[2/5] Réduction de dimension avec UMAP...")
umap_model = umap.UMAP(n_neighbors=2, n_components=5, min_dist=0.0,metric= 'cosine')
reduced_embeddings = umap_model.fit_transform(embeddings)

# Étape 4 : Clustering
'''
def hierarchical_clustering(data, k):
    # Étape 1 : Calculer la matrice de liaison (linkage matrix)
    Z = linkage(data, method='ward')  # 'ward' minimise la variance intra-cluster

    # Étape 2 : Créer des clusters en coupant le dendrogramme
    labels = fcluster(Z, k, criterion='maxclust')

    # Étape 3 : Tracer le dendrogramme (Pas obligé)
    #plt.figure(figsize=(10, 7))
    #dendrogram(Z, truncate_mode='level', p=5)
    #plt.title("Dendrogramme du Clustering Hiérarchique")
    #plt.xlabel("Indices des échantillons")
    #plt.ylabel("Distance")
    #plt.show()

    return labels
'''

print("[3/5] Clustering avec CHA (clustering hierarchique ascendant) ...")


def distance_clusters(cluster1, cluster2):
    # Moyenne des distances entre tous les points des deux clusters (Average linkage)
    distances = [euclidean(p1, p2) for p1 in cluster1 for p2 in cluster2]
    return np.mean(distances)
list_taille= []


def clustering_hierarchique_ascendant(data, nb_clusters):
    from itertools import combinations

    clusters = [[i] for i in range(len(data))]
    min_distance_threshold = 0
    max_distance_threshold = 100  # pour éviter des boucles infinies
 
    increment = 1

    while len(clusters) > nb_clusters:
        list_taille.append(len(clusters))
        to_merge = []
        used = set()
      
        print("taille")
        print(list_taille)
        # Rechercher toutes les paires dont la distance est < seuil
        for i, j in combinations(range(len(clusters)), 2):
            if i in used or j in used:
                continue
            dist = distance_clusters([data[idx] for idx in clusters[i]],
                                     [data[idx] for idx in clusters[j]])
            if dist < min_distance_threshold:
                to_merge.append((i, j))
                used.add(i)
                used.add(j)

        if not to_merge:
            min_distance_threshold += increment
            if min_distance_threshold > max_distance_threshold:
                break
            continue
        new_clusters = []
        merged_indices = set()
        for i, j in to_merge:
            new_clusters.append(clusters[i] + clusters[j])
            merged_indices.update([i, j])

        for idx, cluster in enumerate(clusters):
            if idx not in merged_indices:
                new_clusters.append(cluster)

        clusters = new_clusters

    # Attribution des labels finaux
    labels = [0] * len(data)
    for topic_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = topic_id

    return labels

def clustering_hierarchique_ascendant_avec_densite(data, nb_clusters, nb_voisin=5):
    from copy import deepcopy

    clusters = [[i] for i in range(len(data))]
    core_dists = core_distances(data, nb_voisin)


    while len(clusters) > nb_clusters:
        min_dist = float("inf")
        pair = None

        # Recherche du meilleur couple à fusionner
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = distance_clusters_avec_densite(clusters[i], clusters[j], data, core_dists)
                if dist < min_dist:
                    min_dist = dist
                    pair = (i, j)

        # Fusionner les deux clusters trouvés
        i, j = pair
        new_cluster = clusters[i] + clusters[j]
        clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j]
        clusters.append(new_cluster)
        list_taille.append(len(clusters))

    # Création des labels finaux
    labels = [0] * len(data)
    for topic_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = topic_id

    return labels

nb_topics = 12
print(list_taille)
clusters = clustering_hierarchique_ascendant(reduced_embeddings, nb_topics)
#clusters = clustering_hierarchique_ascendant_avec_densite(reduced_embeddings, nb_topics)
plt.figure(figsize=(20, 6))
plt.plot(range(len(list_taille)), list_taille, marker='o', linestyle='-')
plt.title("Évolution du nombre de clusters au cours du CHA")
plt.xlabel("Itération")
plt.ylabel("Nombre de clusters")
plt.grid(True)
plt.show()
'''
print("[3/5] Clustering avec HDBSCAN...")
cluster_model = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', prediction_data=True)
clusters = cluster_model.fit_predict(reduced_embeddings)
'''

# Étape 5 : Regrouper les documents par topic
print("[4/5] Regroupement des documents par topic...")
topic_docs = defaultdict(list)
for doc, topic in zip(documents, clusters):
    topic_docs[topic].append(doc)

# Liste de stopwords français personnalisée

# Étape 6 : Extraction de mots-clés par TF-IDF


custom_stopwords = list(text.ENGLISH_STOP_WORDS.union([
    'le', 'la', 'les', 'de', 'des', 'du', 'un', 'une', 'et', 'en', 'dans', 'au', 'aux', 'ce', 'ces', 'ça',
    'pour', 'pas', 'par', 'sur', 'se', 'plus', 'ou', 'avec', 'tout', 'mais', 'comme', 'si', 'sans', 'être',
    'cette', 'son', 'sa', 'ses', 'on', 'il', 'elle', 'ils', 'elles', 'nous', 'vous', 'je', 'tu', 'mon', 'ma',
    'mes', 'ton', 'ta', 'tes', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs', 'y', 'donc'
]))

print("[5/5] Extraction des mots-clés par topic...")
topic_keywords = {}
def generate_topic_json(topic_keywords, tfidf_models, output_path="model_output.json"):
    topics_json = []

    for topic_id, keywords in topic_keywords.items():
        tfidf_data = tfidf_models[topic_id]
        vectorizer = tfidf_data["vectorizer"]
        tfidf_matrix = tfidf_data["matrix"]

        words = vectorizer.get_feature_names_out()
        tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        word_scores = list(zip(words, tfidf_scores))

        sorted_keywords = sorted(word_scores, key=lambda x: x[1], reverse=True)
        topic_dict = {
            "topic_id": topic_id,
            "keywords": [[w, float(s)] for w, s in sorted_keywords[:10]]
        }
        topics_json.append(topic_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(topics_json, f, ensure_ascii=False, indent=2)
tfidf_models = {}

for topic, docs in topic_docs.items():
    if topic == -1:  # Ignore les documents non assignés
        continue
    vectorizer = TfidfVectorizer(stop_words=custom_stopwords)
    tfidf_matrix = vectorizer.fit_transform(docs)
    tfidf_models[topic] = { "vectorizer": vectorizer,
    "matrix": tfidf_matrix
}
    words = vectorizer.get_feature_names_out()
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    # Associe chaque mot à son score moyen
    word_scores = list(zip(words, tfidf_scores))
    
    # Trie les mots par score décroissant et garde les 3 meilleurs
    sorted_keywords = sorted(word_scores, key=lambda x: x[1], reverse=True)
    topic_keywords[topic] = [w for w, _ in sorted_keywords]
# Affichage des résultats
#generate_topic_json(topic_keywords, tfidf_models, output_path="model1.json")


print("\n=== Résultats ===")



for topic, documents in topic_docs.items():
    if topic == -1:
        label = "Topic non assigné"
    else:
        keywords = topic_keywords.get(topic, [])

        label = ", ".join(keywords[:3])  # Affiche les 3 premiers mots-clés comme label

    for doc in documents:
        preview = doc[:60].replace('\n', ' ') + "..." if len(doc) > 60 else doc
        print(f"{preview} --> {label}")
# Visualisation des clusters
print("Affichage de la visualisation 2D...")
plt.figure(figsize=(8, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='tab10', s=100)
plt.title("Clustering des documents")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()


# Calculer la similarité intra-cluster pour chaque topic
def calculate_coherence_scores(embeddings, clusters):
    coherence_scores = {}

    for topic in set(clusters):
        if topic == -1:
            continue  # Ignorer les documents non assignés

        # Sélectionner les embeddings des documents appartenant à ce topic
        topic_embeddings = [embeddings[i] for i in range(len(clusters)) if clusters[i] == topic]

        if len(topic_embeddings) > 1:
            # Calculer la matrice de similarité cosinus
            similarity_matrix = cosine_similarity(topic_embeddings)

            # Calculer la similarité moyenne, en ignorant la diagonale (similarité de 1 avec soi-même)
            coherence_score = similarity_matrix.sum() - len(topic_embeddings)  # Soustraire les 1s de la diagonale
            coherence_score /= (len(topic_embeddings) * (len(topic_embeddings) - 1))  # Normaliser

            coherence_scores[topic] = coherence_score
        else:
            print("Ce topic n'a qu'une thèse a son actif:")
            print(topic)
            coherence_scores[topic] = 1  # Si un seul document, la cohérence est parfaite

    return coherence_scores

# Calculer les scores de cohérence
coherence_scores = calculate_coherence_scores(embeddings, clusters)

# Visualiser les scores de cohérence
plt.figure(figsize=(10, 6))
sns.barplot(x=list(coherence_scores.keys()), y=list(coherence_scores.values()), palette="viridis")
plt.title("Score de Cohérence par Topic")
plt.xlabel("Topic")
plt.ylabel("Score de Cohérence")
plt.xticks(rotation=45)
plt.show()

# Liste pour stocker les résultats
topk_values = list(range(2, 100))  # de 2 à 40 mots
diversity_scores = []

# Liste des topics extraits, triés par topic
topics_list = [words for _, words in topic_keywords.items() if len(words) >= 2]

# Boucle pour calculer le score pour chaque topk
for k in topk_values:
    # Vérifie qu'on a assez de mots pour chaque topic
    topics_k = [words[:k] for words in topics_list if len(words) >= k]
    
    if len(topics_k) == 0:
        diversity_scores.append(None)  # ou 0 si tu préfères
        continue
    
    metric = TopicDiversity(topk=k)
    model_output = {"topics": topics_k}
    score = metric.score(model_output)
    diversity_scores.append(score)

# Tracer la courbe
plt.figure(figsize=(10, 5))
plt.plot(topk_values, diversity_scores, marker='o', linestyle='-')
plt.title("Évolution du Topic Diversity en fonction de top-k")
plt.xlabel("Nombre de mots-clés top-k")
plt.ylabel("Score de diversité des topics")
plt.grid(True)
plt.tight_layout()
plt.show()


inverted_rbo = InvertedRBO(topk=10)
metric = TopicDiversity(topk=10) # Initialize metric




topics_list = [words[:40] for _, words in topic_keywords.items() if len(words) >= 2]
model_output = {"topics": topics_list}
topic_diversity_score = metric.score(model_output)
rbo_score = inverted_rbo.score(model_output)
print(f"Score de diversité des topics avec Octis : {topic_diversity_score:.4f}")
print(f"Inverted RBO Score avec Octis: {rbo_score}")