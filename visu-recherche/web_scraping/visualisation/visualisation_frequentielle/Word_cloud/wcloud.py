import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

url = 'https://fr.wikipedia.org/wiki/Trou_noir'  # Remplacez par l'URL des sites de l'encadrant


response = requests.get(url)
web_content = response.content

mask = np.array(Image.open("star.jpg")) # La forme du Nuage

soup = BeautifulSoup(web_content, 'html.parser')
texte = soup.get_text()

mots_bannis = set(STOPWORDS)
mots_bannis.update(["très","au","aux",
  "un", "une", "de", "d'", "l'", "le", "la", "des", "les", "et", "en", "à",
    "on", "du", "ce", "cette", "ces", "pour", "avec", "sans", "par", "dans",
    "sur", "comme", "mais", "ou", "si", "quand", "où", "qui", "que", "quoi",
    "comment", "pourquoi", "car", "donc", "or", "ni", "ne", "n'", "pas", "plus",
    "moins", "tout", "tous", "toute", "toutes", "aucun", "aucune", "chaque",
    "autre", "autres", "son", "sa", "ses", "leur", "leurs", "mon", "ma", "mes",
    "ton", "ta", "tes", "notre", "nos", "votre", "vos", "leur", "leurs", "ceux",
    "celle", "celles", "celui", "ceux-ci", "celle-ci", "celles-ci", "celui-ci",
    "ceux-là", "celle-là", "celles-là", "celui-là", "même", "mêmes", "aucun",
    "aucune", "aucuns", "aucunes", "quel", "quelle", "quels", "quelles", "quelque",
    "quelques", "tel", "telle", "tels", "telles", "tant", "autant", "certains",
    "certaines", "certain", "certaine", "certes", "divers", "diverse", "diverses",
    "différents", "différentes", "quelconque", "quelconques", "nul", "nulle",
    "nuls", "nulles", "n'importe", "n'importe quel", "n'importe quelle",
    "n'importe quels", "n'importe quelles", "n'importe lequel", "n'importe laquelle",
    "n'importe lesquels", "n'importe lesquelles", "n'importe qui", "n'importe quoi",
    "n'importe où", "n'importe quand", "n'importe comment", "n'importe pourquoi",
    "n'importe comment", "n'importe pourquoi", "n'importe quel", "n'importe quelle",
    "n'importe quels", "n'importe quelles", "n'importe lequel", "n'importe laquelle",
    "n'importe lesquels", "n'importe lesquelles", "n'importe qui", "n'importe quoi",
    "n'importe où", "n'importe quand", "n'importe comment", "n'importe pourquoi",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "me", "te", "se",
    "moi", "toi", "soi", "y", "en", "suis", "es", "est", "sommes", "êtes", "sont",
    "serai", "seras", "sera", "serons", "serez", "seront", "étais", "étais", "était",
    "étions", "étiez", "étaient", "fus", "fus", "fut", "fûmes", "fûtes", "furent",
    "sois", "soit", "soyons", "soyez", "soient", "fusse", "fusses", "fût", "fussions",
    "fussiez", "fussent", "serais", "serait", "serions", "seriez", "seraient",
    "ai", "as", "a", "avons", "avez", "ont", "aurai", "auras", "aura", "aurons",
    "aurez", "auront", "avais", "avais", "avait", "avions", "aviez", "avaient",
    "eus", "eus", "eut", "eûmes", "eûtes", "eurent", "aie", "aies", "ait", "ayons",
    "ayez", "aient", "eusse", "eusses", "eût", "eussions", "eussiez", "eussent",
    "aurais", "aurait", "aurions", "auriez", "auraient"
])
# Créer le wordcloud
wordcloud = WordCloud(width=800, height=400, mask=mask,stopwords=mots_bannis,background_color='snow').generate(texte)


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Masquer les axes
plt.show()