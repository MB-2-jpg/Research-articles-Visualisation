
import pandas as pd

from collections import defaultdict, Counter
import prince  # for Correspondence Analysis (CA)
from keybert import KeyBERT
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk
import spacy
from gensim.parsing.preprocessing import STOPWORDS as STOP_ENGLISH
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from prince import CA
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from langdetect import detect
import re

def fix_hyphenated_words(text):
    # Supprime les césures en fin de ligne : "repre-\nsentation" => "representation"
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Supprime aussi les césures invisibles (PDF) ou les mots cassés sans tiret
    text = re.sub(r'(\w+)\s*\n\s*(\w+)', r'\1 \2', text)
    return text

nlp_en = spacy.load('en_core_web_sm')
nlp_fr = spacy.load('fr_core_news_sm')




STOP_French = set(stopwords.words('french'))
stopwords_custom = [
    # Termes académiques génériques
    'pages', 'doi', 'org', 'preprint', 'exp', 'shot', 'fine', 'trained', 'tasks', 
    'proc', 'ismir', 'latent', 'self', 'labels', 'trained', 'label', 'datasets',
    # Français parasites
    'cette', 'plus', 'modèle', 'mots', 'réduction', 'présente', 'ainsi', 'étude', 
    'analyse', 'figure', 'proposé', 'thèse', 'approche', 'résultats',
    # Maths parasites
    'exists', 'proof', 'lemma', 'theorem', 'proposition', 'assumption', 'convergence',
    # NLP/ML parasites
    'trained', 'representation', 'supervised', 'network', 'attention'
]
STOP_French.update(stopwords_custom)
STOP_ALL = STOP_ENGLISH.union(STOP_French)


def preprocess(documents):
    """
    Preprocesses a list of raw text documents and returns tokenized, lemmatized, and cleaned documents.
    Each document is returned as a list of words ready to be used for dictionary/corpus creation.
    """
    
    # Étape 1 : Nettoyage texte brut et tokenisation simple
    def clean_text(text):
        text = fix_hyphenated_words(text)
        return simple_preprocess(text, deacc=True)
    
    data_tokens = [
    [w for w in clean_text(doc) if w not in STOP_ALL]
    for doc in documents]

    # Étape 2 : Bigrammes et trigrammes
    threshold = max(5, min(100, int(len(documents) / 10)))
    bigram = Phrases(data_tokens, min_count=5, threshold=threshold)
    trigram = Phrases(bigram[data_tokens], threshold=threshold)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    # Étape 3 : Stopwords, n-gram, lemmatisation
    def process_doc(tokens):
        # Filtrer stopwords globaux
        tokens = [w for w in tokens if w not in STOP_ALL]
    
        # Créer une string simple pour détecter langue
        text_for_lang = " ".join(tokens)
        try:
            lang = detect(text_for_lang)
        except:
            lang = 'en'  # fallback en anglais si échec detection
    
        # Choix du modèle spaCy selon la langue détectée
        if lang == 'fr':
            nlp_model = nlp_fr
            stop_words_local = set(stopwords.words('french'))
        else:
            nlp_model = nlp_en
            stop_words_local = set(stopwords.words('english'))
    
        # Appliquer bigrammes et trigrammes
        tokens = bigram_mod[tokens]
        tokens = trigram_mod[tokens]

        # Lemmatisation et filtre POS
        doc = nlp_model(" ".join(tokens))
        return [
        token.lemma_ 
        for token in doc
        if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']
        and token.lemma_ not in stop_words_local
        and len(token.lemma_) > 2
    ]

    processed_docs = [process_doc(tokens) for tokens in data_tokens]

    print("Exemple document :", processed_docs[0][:20])  # à commenter ensuite
    print(documents[0])  # texte brut
    print(data_tokens[0])
    return processed_docs

def filter_articles_by_topic(lda_model, corpus_bow, articles_dict, doc2hal_id, topic_idx, threshold=0.3):
    """
    Filtrer les articles associés à un topic donné si leur probabilité dépasse un seuil.
    """
    filtered_articles = {}
    for doc_idx, bow in enumerate(corpus_bow):
        doc_topics = lda_model.get_document_topics(bow, minimum_probability=0)
        prob = dict(doc_topics).get(topic_idx, 0)
        if prob >= threshold:
            hal_id = doc2hal_id[doc_idx]
            if hal_id in articles_dict:
                filtered_articles[hal_id] = articles_dict[hal_id].copy()
                filtered_articles[hal_id]['topic_prob'] = prob
    return filtered_articles

def frequence_keywords(filtered_articles):
    keywords = {}
    for hal_id, info in filtered_articles.items():
        keys = info.get('keywords',[])
        if keys is not None :
            for key in keys : 
                keywords[key] = keywords.get(key,0) + 1
    return keywords

def concat_title_keywords(filtered_articles):
    """
    Prend un dict d'articles (comme filtered) et retourne un dict
    avec hal_id comme clé et la concaténation 'title + keywords' en texte.
    """
    concatenated = {}
    for hal_id, info in filtered_articles.items():
        title = info.get('title', '')
        keywords = info.get('keywords', [])
        keywords_text = ' '.join(keywords) if keywords else ''
        concatenated_text = f"{title} {keywords_text}".strip()
        concatenated[hal_id] = concatenated_text
    return concatenated

from sklearn.feature_extraction.text import CountVectorizer

def extract_ngrams_list(texts_list, n=2,min_freq=1):
    vectorizer = CountVectorizer(ngram_range=(n, n), min_df=min_freq)
    X = vectorizer.fit_transform(texts_list)
    freqs = X.sum(axis=0).A1
    ngrams = vectorizer.get_feature_names_out()
    ngram_freq = dict(zip(ngrams, freqs))
    ngram_freq = dict(sorted(ngram_freq.items(), key=lambda item: item[1], reverse=True))
    return ngram_freq




def extract_ngrams(texts_dict, n=2, min_freq=1):
    """
    Extrait les n-grammes les plus fréquents à partir d'un dict de textes.
    """
    documents = list(texts_dict.values())
    vectorizer = CountVectorizer(ngram_range=(n, n), min_df=min_freq)
    X = vectorizer.fit_transform(documents)
    freqs = X.sum(axis=0).A1
    ngrams = vectorizer.get_feature_names_out()
    ngram_freq = dict(zip(ngrams, freqs))
    ngram_freq = dict(sorted(ngram_freq.items(), key=lambda item: item[1], reverse=True))
    return ngram_freq

def remove_stopword_bigrams(ngram_freq):
    """
    Supprime les n-grammes composés uniquement de stopwords (anglais).
    """
    filtered = {}
    for ngram, freq in ngram_freq.items():
        words = ngram.split()
        # Si au moins un des mots n'est pas un stopword, on garde
        if any(word.lower() in STOP_ALL for word in words):
            continue
        filtered[ngram] = freq
    return filtered

def filter_ngrams_by_lda_keywords(lda_model, dictionary, topic_id, ngram_freq, topn_keywords=20):
    """
    Filtre et ordonne les n-grammes selon la présence des top mots clés LDA.
    """
    topic_terms = lda_model.get_topic_terms(topicid=topic_id, topn=topn_keywords)
    topic_keywords = [dictionary[id_word] for id_word, prob in topic_terms]
    filtered_ngrams = []
    seen_ngrams = set()
    for kw in topic_keywords:
        for ngram, freq in ngram_freq.items():
            if kw.lower() in ngram.lower() and ngram not in seen_ngrams:
                filtered_ngrams.append((ngram, freq))
                seen_ngrams.add(ngram)
    return filtered_ngrams

def visualisation_bigrams(num_topics):
    """
    Visualisation des bigrams associés aux topics LDA avec une analyse en correspondance (CA).
    """
      # --- Configuration des topics ---
    
    results = main(num_topics)
    lda_model = results["lda"]  
    corpus = results["corpus"]
    dictionary = results["dictionary"]
    documents = results["documents"]


    # 1. Filtrage des articles par topic
    filtered_articles = [filter_articles_by_topic(
        lda_model, corpus, DIC_HALIDS, {i: hal_id for i, hal_id in enumerate(hal_list)}, topic_idx, threshold=0.1) for topic_idx in range(num_topics)]
    
    # 2. Concaténation titre + keywords
    #concatenated_texts = [concat_title_keywords(filtered_articles[i]) for i in range(num_topics)]

    frequence_keys = [ frequence_keywords(article) for article in filtered_articles]

    for i in range(num_topics):
        print(sorted(frequence_keys[i].items(), key= lambda x: x[1],reverse = True)[:6])
    exit()
    # 3. Extraction des bigrammes
    bigrams_freq = [extract_ngrams(text, n=2, min_freq=1) for text in concatenated_texts]   
    bigrams_freq = [remove_stopword_bigrams(bigram) for bigram in bigrams_freq]
    brut_ngrams = [list(bigrams_freq[i].keys()) for i in range(num_topics)]
   
  

    # 4. Filtrage des n-grammes par mots-clés LDA
    filtered_ngrams = [filter_ngrams_by_lda_keywords(lda_model, dictionary, i, bigrams, topn_keywords=200) for i, bigrams in enumerate(bigrams_freq)]

     # --- Construction de la matrice de contingence Topic × Bigram ---
    bigram_list = sorted(set(bg for topic_bigrams in brut_ngrams for bg in topic_bigrams))

    cont_dict = {t: {bg: 0 for bg in bigram_list} for t in range(num_topics)}

    for topic_id, topic_bigrams in enumerate(bigrams_freq):
        for bigram, freq in topic_bigrams.items():
            cont_dict[topic_id][bigram] += freq  # Présence du bigramme parmi les keywords du topic
    
    for i in range (num_topics) : 
        print(f'Bigrammes les plus fréquents pour le topic{i}:\n')
        sorted(cont_dict[i].items(), key=lambda x: x[1], reverse=True)
        print(sorted(cont_dict[i].items(), key=lambda x: x[1], reverse=True)[:5])

        
    cont_df = pd.DataFrame.from_dict(cont_dict, orient='index', columns=bigram_list)
    cont_df = cont_df.loc[:, cont_df.sum(axis=0) > 0]  # Ne garder que les colonnes non nulles


    # DataFrame avec les topics en index, bigrammes en colonnes
    cont_df = pd.DataFrame.from_dict(cont_dict, orient='index', columns=bigram_list)

    # Optionnel : ne garder que les bigrams qui apparaissent quelque part

    cont_df = cont_df.loc[cont_df.sum(axis=1) > 0, :]
    cont_df = cont_df.loc[:, cont_df.sum(axis=0) > 0]

    from adjustText import adjust_text



    # --- Analyse en Correspondance ---
    ca = CA(n_components=2, n_iter=30, copy=True, engine='scipy', random_state=42).fit(cont_df)
    row_coords = ca.row_coordinates(cont_df)    # Topics
    col_coords = ca.column_coordinates(cont_df) # Bigrams

    # --- Affichage amélioré ---
    plt.figure(figsize=(16, 12))
    plt.grid(True, linestyle='--', alpha=0.3)

    # Scatter pour Topics
    plt.scatter(row_coords[0], row_coords[1], marker='o', c='darkgreen', edgecolors='black', s=80, label='Topics')

    # Scatter pour Bigrams
    plt.scatter(col_coords[0], col_coords[1], marker='^', c='crimson', edgecolors='black', s=60, label='Bigrams')

    # Labels avec gestion de chevauchement
    texts = []
    for i, lbl in enumerate(row_coords.index):
        texts.append(plt.text(row_coords.iloc[i, 0], row_coords.iloc[i, 1], f"Topic {lbl}", fontsize=11, color='darkgreen'))

    for i, lbl in enumerate(col_coords.index):
        texts.append(plt.text(col_coords.iloc[i, 0], col_coords.iloc[i, 1], lbl, fontsize=9, color='crimson'))

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),expand_text=(2.0, 3.0))  # Plus d'espace autour des étiquettes
    words = [lda_model.show_topic(topic_id, topn=10) for topic_id in range(num_topics)]  # topn=30 mots par topic
    keywords = [", ".join([word for word, prob in topic_words]) for topic_words in words]
    final_keywords = "\n".join([f"Topic {i}: {kw}" for i, kw in enumerate(keywords)])

    plt.figtext(0.4, 0.8, final_keywords, horizontalalignment='right', fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.axhline(0, color='gray', linestyle='--', lw=0.8)
    plt.axvline(0, color='gray', linestyle='--', lw=0.8)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.title("Analyse en Correspondance – Topics & Bigrams", fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.show()

          
def main(num_topics,documents,dic_halids, hal_list):
    # --- 0. Préparation des données ---
    

    #nltk.download('stopwords')
   

    doc_authors = []

    documents_processed = preprocess(documents)
    print(documents_processed[:2])

    for hal_id in hal_list:   
        doc_authors.append(dic_halids[hal_id].get("authors", []))

    # --- 2. Construire le dictionnaire et le corpus pour LDA ---
    dictionary = corpora.Dictionary(documents_processed)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in documents_processed]

    # --- 3. Entraîner le modèle LDA ---
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha="auto",
        per_word_topics=False
    )

    # --- 4. Attribuer à chaque document son topic dominant ---

    doc_topics = []
    for bow in corpus:
        topics_probs = lda.get_document_topics(bow)
        dominant_topic = max(topics_probs, key=lambda x: x[1])[0]
        doc_topics.append(dominant_topic)

    topic_doc = [[] for _ in range(num_topics)]
    for i, topic in enumerate(doc_topics):
        topic_doc[topic].append(i)


    # --- Afficher les mots-clés associés à chaque topic ---
    print("\nMots-clés par topic :")
    for topic_id in range(num_topics):
        print(f"Topic {topic_id} : ", end='')
        words = lda.show_topic(topic_id, topn=30)  # topn=30 mots par topic
        keywords = ", ".join([word for word, prob in words])
        print(keywords)

    # 1. Score de cohérence c_v
    """coherence_model = CoherenceModel(
        model=lda,
        texts=documents_processed,
        dictionary=dictionary,
        coherence='c_v'
    )
    print("Score de cohérence (c_v) :", coherence_model.get_coherence())
"""
    return {
        "lda": lda,
        "corpus": corpus,
        "dictionary": dictionary,
        "HAL_LIST": hal_list,
        "documents": documents_processed,
        "doc_authors": doc_authors,
        "doc_topics": doc_topics,   
    }

def table_contingence():
    num_topics = 12  # Nombre de topics à extraire
    results = main(num_topics)
    lda = results["lda"]
    corpus = results["corpus"]
    dictionary = results["dictionary"]
    doc_authors = results["doc_authors"]
    doc_topics = results["doc_topics"]  
    HAL_LIST = results["HAL_LIST"]

    # 3. Matrice de contingence Thèse × Auteur
    all_authors = sorted(TARGET_AUTHORS)
    cont_df = pd.DataFrame(
        [
            {auth: (1 if auth in doc_authors[i] else 0) for auth in all_authors}
            for i in range(len(doc_authors))
    ],
    index=[f"Thèse {i+1}" for i in range(len(doc_authors))],
    columns=all_authors
)
# éventuellement supprimer les auteurs jamais présents
    cont_df = cont_df.loc[:, cont_df.sum(axis=0) > 0]

# éventuellement supprimer les auteurs absents
    cont_df = cont_df.loc[:, cont_df.sum(axis=0) > 0]

# 4. Analyse en Correspondance
    ca = CA(
    n_components=2,
    n_iter=15,
    copy=True,
    check_input=True,
    engine='scipy',       # moteur valide
    random_state=42
    ).fit(cont_df)

    row_coords = ca.row_coordinates(cont_df)
    col_coords = ca.column_coordinates(cont_df)

# 5. Projection 2D
    plt.figure(figsize=(15, 8))

# Thèses
    plt.scatter(row_coords[0], row_coords[1], marker='o', c='blue', label='Thèses')
    for i, lbl in enumerate(row_coords.index):
        plt.text(row_coords.iloc[i, 0], row_coords.iloc[i, 1], lbl, fontsize=8, color='blue')

# Auteurs
    plt.scatter(col_coords[0], col_coords[1], marker='^', c='red', label='Auteurs')
    for i, lbl in enumerate(col_coords.index):
        plt.text(col_coords.iloc[i, 0], col_coords.iloc[i, 1], lbl, fontsize=7, color='red')

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Analyse en Correspondance – Thèses & Auteurs")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# --- 3. Matrice de contingence Topic × Auteur ---
    all_authors = sorted(TARGET_AUTHORS)
    all_topics = list(range(num_topics))

# Construire un dict[topic][auteur] = nombre de docs du topic où l'auteur apparaît
    cont_dict = {t: {a: 0 for a in all_authors} for t in all_topics}
    for topic, authors in zip(doc_topics, doc_authors):
        for a in authors:
            cont_dict[topic][a] += 1

# DataFrame avec les topics en index, auteurs en colonnes
    cont_df = pd.DataFrame.from_dict(cont_dict, orient='index', columns=all_authors)

# Optionnel : ne garder que les auteurs qui apparaissent
    cont_df = cont_df.loc[:, cont_df.sum(axis=0) > 0]

# --- 4. Analyse en Correspondance sur Topic × Auteur ---
    ca = CA(
        n_components=2,
    n_iter=30,
    copy=True,
    check_input=True,
    engine='scipy',
    random_state=42
    ).fit(cont_df)

    row_coords = ca.row_coordinates(cont_df)    # ici, ce sont les topics
    col_coords = ca.column_coordinates(cont_df) # ici, les auteurs

# --- 5. Projection 2D Topics (cercles) & Auteurs (triangles) ---
    plt.figure(figsize=(12, 8))
# Topics
    plt.scatter(row_coords[0], row_coords[1], marker='o', c='green', label='Topics')
    for i, lbl in enumerate(row_coords.index):
        plt.text(row_coords.iloc[i, 0], row_coords.iloc[i, 1], f"Topic {lbl}", fontsize=9, color='green')
# Auteurs
    plt.scatter(col_coords[0], col_coords[1], marker='^', c='red', label='Auteurs')
    for i, lbl in enumerate(col_coords.index):
        plt.text(col_coords.iloc[i, 0], col_coords.iloc[i, 1], lbl, fontsize=8, color='red')

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Analyse en Correspondance – Topics & Auteurs")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def visualisation(results,authors=None):

    lda = results['lda']
    corpus = results['corpus']
    dictionary = results['dictionary']
    
    vis_data = gensimvis.prepare(lda, corpus, dictionary)

    #if authors : 
    pyLDAvis.save_html(vis_data,"static/lda_vis.html")
    #else : pyLDAvis.save_html(vis_data, f'lda_visualisation__automatise_.html')

"""
if __name__ == "__main__":

    # --- Configuration des topics ---
    num_topics = 12  # Nombre de topics à extraire
    # Exécution du pipeline principal
    results = main(num_topics)
    lda = results["lda"]
    corpus = results["corpus"]
    dictionary = results["dictionary"]
    HAL_LIST = results["HAL_LIST"]

    # 1. Filtrage des articles par topic
    filtered =[ filter_articles_by_topic(
        lda, corpus, DIC_HALIDS, {i: hal_id for i, hal_id in enumerate(HAL_LIST)}, topic_idx, threshold=0.1) for topic_idx in range(num_topics) ]

    for i in range(num_topics):
        print(f"\nTopic {i} : {len(filtered[i])} articles")
        for hal_id, info in filtered[i].items():
            print(f"{hal_id}: {info['title']} (probabilité: {info['topic_prob']:.4f})")

    titles_raw = []
    total_keywords = []

# 1. Récupération des titres et mots-clés
    for i in range(num_topics):
        topic_titles = [info['title'] for hal_id, info in filtered[i].items() if 'keywords' in info]
        titles_raw.extend(topic_titles)
    
        topic_keywords = []
        for hal_id, info in filtered[i].items():
            if 'keywords' in info:
                keywords = info['keywords']  # O n suppose que les mots-clés sont séparés par des espaces
                if keywords:
                    topic_keywords.extend(keywords)
        total_keywords.append(topic_keywords)

# 2. Extraction des bigrammes
    bigrams_freq = [extract_ngrams_list(text, n=2, min_freq=2) for text in titles_raw]
    bigrams_freq = [remove_stopword_bigrams(bigram) for bigram in bigrams_freq]

# 3. Comptage des mots-clés
    total_keywords_dict = []
    for keywords in total_keywords:
        keyword_count = {}
        for key in keywords:
            keyword_count[key] = keyword_count.get(key, 0) + 1
        total_keywords_dict.append(keyword_count)

       
            # 4. Trier les mots-clés par fréquence décroissante pour chaque topic
    sorted_keywords_by_topic = []

    for keyword_count in total_keywords_dict:
        sorted_keywords = sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)
        sorted_keywords_by_topic.append(sorted_keywords)

# Exemple d'affichage des top keywords par topic :
    for i, keywords in enumerate(sorted_keywords_by_topic):
        print(f"Topic {i+1}:")
        for bigrams in bigrams_freq[i].items()[:10]: # Affichage top bigrams
            print(f"  {bigrams[0]}: {bigrams[1]}")
        for keyword, freq in keywords:
            print(f"  {keyword}: {freq}")
            


    # 4. Filtrage des n-grammes par mots-clés LDA
    filtered_ngrams = [filter_ngrams_by_lda_keywords(lda, dictionary, topic_id=i, ngram_freq=bigrams, topn_keywords=20) for i, bigrams in enumerate(bigrams_freq)]

    for i, filtered in enumerate(filtered_ngrams):
        print(f"\nFiltered n-grams for topic {i}:")
        for ngram, freq in filtered[:10]:
            print(f"{ngram}: {freq}")

"""



if __name__ == "__main__":
  from config import prepare_corpus
  documents, dic_halids, hal_list, authors= prepare_corpus()
  target_authors= set(authors)  # Dictionnaire des auteurs cibles

  #visualisation_bigrams()
  #table_contingence()
  results = main(6,documents,dic_halids,hal_list)
  visualisation(results,authors)

