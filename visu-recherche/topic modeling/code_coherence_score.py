from gensim import corpora
from gensim.models import CoherenceModel

# Supposons que `documents` est une liste de documents (chaque document étant une liste de tokens)
# et que `topics` est la liste des thèmes extraits par LSA (liste de listes de mots)

# Créer un dictionnaire
dictionary = corpora.Dictionary(documents)

# Créer un corpus en bag-of-words à partir de ce dictionnaire
corpus_bow = [dictionary.doc2bow(doc) for doc in documents]

# Calculer la coherence en utilisant, par exemple, le measure "c_v"
coherence_model = CoherenceModel(topics=topics, texts=documents, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model.get_coherence()

print("Coherence Score :", coherence_score)
