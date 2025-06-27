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
BIB_FILE = Path("../pdf_scraping/article.txt")

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

def run_topic_modeling(keywords_chosen: list[str]=["audio", "signal","vision", "machine learning", "source", "separation", "listening"], years: list[int]=[2020,2023]):
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
    def entropy(probs):
        """Entropie en base 2 d'une distribution de probabilité."""
        probs = np.array(probs)
        probs = probs[probs > 0]  # éviter log(0)
        return -np.sum(probs * np.log2(probs))
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
    print(database)

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
        an_deb, an_fin = intervalle_annee
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

    # --------------------------------------------------------------------
    # 8. Associer chaque auteur à la distribution moyenne des topics
    # --------------------------------------------------------------------
    author_topic_counts = defaultdict(lambda: [0.0] * NUM_TOPICS)
    author_doc_counts = defaultdict(int)

    for key, (authors, _, _, _) in database.items():
        topic_dist = doc_topic_distributions[key]  # liste (topic_id, score)
        for author in authors:
            for topic_id, score in topic_dist:
                author_topic_counts[author][topic_id] += score
            author_doc_counts[author] += 1

    # Moyenne normalisée par auteur
    author_topic_distribution = {}
    for author, topic_sums in author_topic_counts.items():
        count = author_doc_counts[author]
        if count > 0:
            percentages = [round(v / count, 4) for v in topic_sums]
            author_topic_distribution[author] = percentages

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

    # --------------------------------------------------------------------
    # 11. Fonction d’étiquetage automatique des topics
    # --------------------------------------------------------------------
    # topic_modeling.py



    def get_topic_labels(lda_model, num_keywords=10):
        """Retourne une étiquette par topic (mot-clé le plus associé non encore pris)."""
        used_keywords = set()
        topic_labels = []

        for topic_id in range(lda_model.num_topics):
            # Récupère les top mots-clés du topic
            words = [word for word, _ in lda_model.show_topic(topic_id, topn=num_keywords)]
            # Prend le premier mot non encore utilisé
            for word in words:
                if word not in used_keywords:
                    topic_labels.append(word)
                    used_keywords.add(word)
                    break
            else:
                topic_labels.append(f"Topic {topic_id}")  # fallback

        return topic_labels


    def get_topic_labels_from_keywords(database, doc_topic_distributions, num_topics=5):
        """
        Étiquette les topics avec les mots-clés BibTeX les plus associés,
        sans duplication (même en forme canonique).
        """
        from collections import defaultdict

        def strip_accents(text: str) -> str:
            import unicodedata
            text = unicodedata.normalize("NFD", text)
            text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
            return text.lower()

        keyword_topic_weights = defaultdict(lambda: [0.0] * num_topics)

        # Accumuler les scores des mots-clés pour chaque topic
        for doc_key, (authors, keywords, year, title) in database.items():
            topic_dist = dict(doc_topic_distributions[doc_key])  # {topic_id: score}
            for kw in keywords:
                for topic_id, score in topic_dist.items():
                    keyword_topic_weights[kw][topic_id] += score

        used_keywords_canon = set()
        topic_labels = [""] * num_topics

        for topic_id in range(num_topics):
            best_kw = None
            best_score = -1.0
            for kw, scores in keyword_topic_weights.items():
                kw_canon = strip_accents(kw)
                if kw_canon in used_keywords_canon:
                    continue
                if scores[topic_id] > best_score:
                    best_score = scores[topic_id]
                    best_kw = kw
            if best_kw:
                topic_labels[topic_id] = best_kw
                used_keywords_canon.add(strip_accents(best_kw))
            else:
                topic_labels[topic_id] = f"Topic {topic_id}"

        return topic_labels

    # Calcul de l'entropie des documents
    doc_entropies = {
        doc_key: entropy([p for _, p in topic_dist])
        for doc_key, topic_dist in doc_topic_distributions.items()
    }
    # Calcul de l'entropie des auteurs (sur leur distribution moyenne de topics)
    author_entropies = {
        author: entropy(distribution)
        for author, distribution in author_topic_distribution.items()
    }

    # Renommer les colonnes (topics) par des mots-clés associés
    topic_labels = get_topic_labels_from_keywords(database, doc_topic_distributions,NUM_TOPICS)
    df.columns = topic_labels

    # --------------------------------------------------------------------
    # 12. Analyse factorielle des correspondances
    # --------------------------------------------------------------------
    ca = prince.CA(n_components=2, n_iter=10, copy=True, check_input=True, engine='sklearn')

    ca = ca.fit(df)

    # Coordonnées
    row_coords = ca.row_coordinates(df)     # auteurs
    col_coords = ca.column_coordinates(df)  # topics

    # --------------------------------------------------------------------
    # 13. Affichage dans un même plan
    # --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    # Afficher auteurs
    ax.scatter(row_coords[0], row_coords[1], color='blue', label='Auteurs')
    for i, name in enumerate(row_coords.index):
        ax.text(row_coords.iloc[i, 0], row_coords.iloc[i, 1], name, color='blue', fontsize=8)

    # Afficher topics
    ax.scatter(col_coords[0], col_coords[1], color='red', label='Topics')
    for i, name in enumerate(col_coords.index):
        ax.text(col_coords.iloc[i, 0], col_coords.iloc[i, 1], name, color='red', fontsize=10, weight='bold')

    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    ax.set_title("Analyse en correspondances : Auteurs et Topics")
    ax.legend()
    plt.tight_layout()
    plt.show()





    # 📊 Diagramme en barres des entropies des auteurs (triées)
    sorted_authors = sorted(author_entropies.items(), key=lambda x: x[1], reverse=True)
    authors, entropies = zip(*sorted_authors)

    plt.figure(figsize=(12, 6))
    plt.bar(authors, entropies, color='salmon')
    plt.title("Entropie des distributions de topics par auteur")
    plt.xlabel("Auteur")
    plt.ylabel("Entropie (bits)")
    plt.xticks(rotation=90)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    from collections import defaultdict
    import pandas as pd

    # 1. Construire un dictionnaire Auteur → Mots-clés utilisés
    author_keywords = defaultdict(lambda: defaultdict(int))

    for _, (authors, keywords, _, _) in database.items():
        for author in authors:
            for kw in keywords:
                kw_clean = strip_accents(kw.lower())
                author_keywords[author][kw_clean] += 1

    # 2. Transformer en DataFrame
    df = pd.DataFrame.from_dict(author_keywords, orient='index').fillna(0).astype(int)
    df.index.name = "Author"

if __name__ == "__main__":
    run_topic_modeling()
