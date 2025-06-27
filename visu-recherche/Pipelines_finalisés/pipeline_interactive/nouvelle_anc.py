  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construire un dictionnaire (clé -> [auteurs, keywords, année, titre])
à partir des entrées BibTeX présentes dans article.txt
"""
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
import re
import json
import unicodedata
from pathlib import Path
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import io
nltk.download('stopwords')
BIB_FILE = Path("../../pdf_scraping/article.txt")



# --------------------------------------------------------------------
# 1. Auteur·e·s d'intérêt (forme canonique)
# --------------------------------------------------------------------
CANONICAL_AUTHORS = {
    "Roland Badeau", "Pascal Bianchi", "Philippe Ciblat",
    "Stephan Clémençon", "Florence d'Alché-Buc", "Slim Essid",
    "Olivier Fercoq", "Pavlo Mozharovskyi", "Geoffroy Peeters",
    "Gaël Richard", "François Roueff", "Maria Boritchev",
    "Radu Dragomir", "Mathieu Fontaine", "Ekhiñe Irurozki",
    "Yann Issartel", "Hicham Janati", "Ons Jelassi",
    "Matthieu Labeau", "Charlotte Laclau",
    "Laurence Likforman-Sulem", "Yves Grenier",
}
list_test=[]
# --------------------------------------------------------------------
# 2. Petites fonctions utilitaires
# --------------------------------------------------------------------

def run_topic_modeling(keywords_chosen: list[str]=["graph" , "stochastic" , "neural" ,"audio", "signal","vision", "machine learning", "source", "separation", "listening"], years: list[int]=[2020,2023]):
    # normalisation simple de tous les prénoms et noms canoniques
    def strip_accents(text: str) -> str:
        """Enlève les accents, renvoie en minuscules."""
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return text.lower()


    CANONICAL_TOKENS = {
        full: {strip_accents(t) for t in full.replace("-", " ").split()}
        for full in CANONICAL_AUTHORS
    }

    # --------------------------------------------------------------------
    # 3. Lecture du fichier BibTeX
    # --------------------------------------------------------------------
    bibtex_text = BIB_FILE.read_text(encoding="utf-8")

    # On découpe les entrées sur un « @ » suivi d’un type, jusqu’à la
    # accolade fermante correspondante (équilibrage approximatif)
    entry_pattern = re.compile(r"@(?:article|inproceedings)\{[\s\S]+?\n\}")
    entries = entry_pattern.findall(bibtex_text)

    # --------------------------------------------------------------------
    # 4. Extraction des informations pour chaque entrée
    # --------------------------------------------------------------------
    def extract_field(entry: str, field: str) -> str | None:
        """Retourne le contenu du champ BibTeX (sans les accolades), ou None."""
        m = re.search(
            rf"{field}\s*=\s*\{{\s*([\s\S]*?)\s*\}}\s*,?", entry, re.IGNORECASE
        )
        return m.group(1).strip() if m else None


    database: dict[str, list] = {}
   
    for entry in entries:
        # clé BibTeX = texte entre '{' et la première virgule
        key_match = re.match(r"@\w+\{([^,]+),", entry)
        if not key_match:
            continue
        bib_key = key_match.group(1).strip()

        # champs utiles
        authors_raw = extract_field(entry, "author") or ""
        title = extract_field(entry, "title") or ""
        keywords_raw = extract_field(entry, "keywords") or ""
        year_raw = extract_field(entry, "year") or ""

        # ----------------------------------------------------------------
        # 4.a Détection des auteur·e·s d'intérêt
        # ----------------------------------------------------------------
        authors_found: list[str] = []
        authors_norm = strip_accents(authors_raw)
        for full_name, tokens in CANONICAL_TOKENS.items():
            if any(token in authors_norm for token in tokens):
                authors_found.append(full_name)

        # ----------------------------------------------------------------
        # 4.b Nettoyage des mots-clés et de l'année
        # ----------------------------------------------------------------
        keywords = (
            [kw.strip() for kw in keywords_raw.split(";") if kw.strip()]
            if keywords_raw
            else []
        )
        try:
            year = int(year_raw)
        except ValueError:
            year = None  # année manquante ou mal formée

        # ----------------------------------------------------------------
        # 4.c Ajout au dictionnaire
        # ----------------------------------------------------------------
        database[bib_key] = [authors_found, keywords, year, title]
        list_test.append(bib_key)
    

    def strip_accents(text: str) -> str:
        """Enlève les accents et passe en minuscules."""
        import unicodedata
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return text.lower()

    def filtrage(database: dict, intervalle_annee: list[int], mots_cles: list[str]) -> dict:
        """
        Filtre les articles publiés entre les deux années (incluses)
        et contenant au moins un mot-clé (en forme canonique, même partielle).
        """
        an_deb = int(intervalle_annee[0])
        an_fin = int(intervalle_annee[1])
        mots_cles_canon = [strip_accents(kw) for kw in mots_cles]

        filtré = {}
        for key, (auteurs, keywords, année, titre) in database.items():
            if année is None or not (an_deb <= année <= an_fin):
                continue

            keywords_canon = [strip_accents(kw) for kw in keywords]
            
            match = any(
                filtre in kw_canon
                for kw_canon in keywords_canon
                for filtre in mots_cles_canon
            )
            if match:
                filtré[key] = [auteurs, keywords, année, titre]
        
        return filtré
    database = filtrage(database, years, keywords_chosen)


    def choose_from_my_list_of_keywords(kd):
        L = [strip_accents(k) for k in keywords_chosen ]
        wd =strip_accents(kd)
        for i in range(len(L)):
            if wd in L[i].split(" ")  or wd in L[i].split("-"):
                return keywords_chosen[i]
        return False









    from sklearn.feature_extraction.text import CountVectorizer
    from gensim import corpora, models
    import nltk
    from collections import defaultdict
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    # --------------------------------------------------------------------
    # 5. Préparer le corpus textuel
    # --------------------------------------------------------------------
    stop_words = set(stopwords.words("english"))
    documents = []
    doc_keys = []

    for key, (authors, keywords, _, title) in database.items():
        text = " ".join(keywords + [title])
        tokens = [
            word.lower()
            for word in re.findall(r"\b\w+\b", text)
            if word.lower() not in stop_words and len(word) > 2
        ]
        documents.append(tokens)
        doc_keys.append(key)

    # Créer dictionnaire et corpus pour Gensim
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # --------------------------------------------------------------------
    # 6. Appliquer LDA pour extraire les topics
    # --------------------------------------------------------------------
    NUM_TOPICS = 3
    lda_model = models.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=10, random_state=42)

    # Pour voir les topics :
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")

    # --------------------------------------------------------------------
    # 7. Associer chaque document à sa distribution de topics
    # --------------------------------------------------------------------
    doc_topic_distributions = {
        key: lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
        for i, key in enumerate(doc_keys)
    }
    k = len(keywords_chosen)
    # --------------------------------------------------------------------
    # 8. Associer chaque auteur à la distribution moyenne des topics
    # --------------------------------------------------------------------
    author_topic_counts = defaultdict(lambda: [0.0] * NUM_TOPICS)
    author_doc_counts = defaultdict(int)
    kwrds_topic_counts = defaultdict(lambda: [0.0] * NUM_TOPICS)
    kwrds_doc_counts = defaultdict(int)
    for key, (authors, kwrds, _, _) in database.items():
        topic_dist = doc_topic_distributions[key] 
        print(topic_dist) # liste (topic_id, score)
        for author in authors:
            for topic_id, score in topic_dist:
                author_topic_counts[author][topic_id] += score
            author_doc_counts[author] += 1
        
        for kwrd in kwrds[0:15]:
           # if(choose_from_my_list_of_keywords(kwrd)):
            #kwrd = choose_from_my_list_of_keywords(kwrd)
            for topic_id, score in topic_dist:
                kwrds_topic_counts[kwrd][topic_id] += score
            kwrds_doc_counts[kwrd] += 1
            #else : kwrds.pop(kwrds.index(kwrd))
    # Moyenne normalisée par auteur
    author_topic_distribution = {}
    for author, topic_sums in author_topic_counts.items():
        print(author, topic_sums)
        count = author_doc_counts[author]
        if count > 0:
            percentages = [round(v / count, 4)+0.001 for v in topic_sums]
            author_topic_distribution[author] = percentages


    i=0
    kwrds_topic_distribution = {}
    L=[]
    for kw, topic_sums in kwrds_topic_counts.items():
        print(kw, topic_sums)

        if(kw) and i<15:
            i+=1
            count = kwrds_doc_counts[kw]
            if count > 0 :
                percentages = [round(v / count, 4)  +0.001 for v in topic_sums]

                
                if not(percentages in L):
                     kwrds_topic_distribution[kw] = percentages
                     L.append(percentages)
    print(kwrds_topic_distribution)

    
    # --------------------------------------------------------------------
    # 9. Sauvegarde (optionnelle)
    # --------------------------------------------------------------------
    with open("author_topic_distribution.json", "w", encoding="utf-8") as f:
        json.dump(author_topic_distribution, f, indent=2, ensure_ascii=False)

    import pandas as pd
    import matplotlib.pyplot as plt
    import prince

    # --------------------------------------------------------------------
    # 10. Construire une DataFrame pour l’analyse des correspondances
    # --------------------------------------------------------------------
    df = pd.DataFrame.from_dict(author_topic_distribution, orient='index')
    df.columns = [f"Topic {i}" for i in range(df.shape[1])]
    df.index.name = "Author"
    


    df2 = pd.DataFrame.from_dict(kwrds_topic_distribution, orient='index')
    df2.columns = [f"Topic {i}" for i in range(df2.shape[1])]
    df2.index.name = "Keyword"
    # --------------------------------------------------------------------
    # 11. Fonction d’étiquetage automatique des topics
    # --------------------------------------------------------------------
    # topic_modeling.py

    # --------------------------------------------------------------------
    # 12. Analyse factorielle des correspondances
    # --------------------------------------------------------------------
    ca = prince.CA(n_components=2, n_iter=10, copy=True, check_input=True, engine='sklearn')

    ca = ca.fit(df)
  
    # Coordonnées
    row_coords = ca.row_coordinates(df)     # auteurs
    col_coords = ca.column_coordinates(df)  # topics
    print(ca.row_coordinates(df) )
    def calculate_position(kw):
        sum=0
        x=0
        y=0
        for topic_id, score in topic_dist:
                sum += kwrds_topic_counts[kw][topic_id]
        for topic_id, score in topic_dist:
                x += kwrds_topic_counts[kw][topic_id]*col_coords.iloc[topic_id, 0]/sum
        for topic_id, score in topic_dist:
                y += kwrds_topic_counts[kw][topic_id]*col_coords.iloc[topic_id, 1]/sum


        return x,y
         
    ca = prince.CA(n_components=3, n_iter=10, copy=True, check_input=True, engine='sklearn')
    ca2 = ca.fit(df2)
    row_coords2 = ca2.row_coordinates(df2)     
    col_coords2 = ca2.column_coordinates(df2)  
    print(ca2.row_coordinates(df2) )
    fig, ax = plt.subplots(figsize=(10, 8))

    # Afficher auteurs
    ax.scatter(row_coords[0], row_coords[1], color='blue', label='Auteurs')
   # ax.scatter(row_coords2[0], row_coords2[1], color='green', label='kwrds')
    for i, name in enumerate(row_coords.index):
        ax.text(row_coords.iloc[i, 0], row_coords.iloc[i, 1], name, color='blue', fontsize=8)
    #for i, name in enumerate(row_coords2.index):
       # ax.text(row_coords2.iloc[i, 0], row_coords2.iloc[i, 1], name, color='green', fontsize=8)
   
    def est_proche_de_la_liste(point , liste):
        var =False
        if not L:
            return False
        for p in liste:
            if (p[0]-point[0])**2+((p[1]-point[1]))**2 <0.3:
                   var = True
        return var


    i=0 
    L=[]
    for kw, topic_sums in kwrds_topic_counts.items():
       
        if i<10  and  not(est_proche_de_la_liste([calculate_position(kw)[0], calculate_position(kw)[1]] , L)): 
            ax.text(calculate_position(kw)[0], calculate_position(kw)[1], kw, color='green', fontsize=8)
            i+=1
            L.append([calculate_position(kw)[0],calculate_position(kw)[1]])
           
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    ax.set_title("Analyse en correspondances : Auteurs et keywords")
    ax.legend()
    plt.savefig("result_nouvelle_anc.png")
   # plt.show()
    
    ax.legend()
    plt.tight_layout()






#run_topic_modeling()