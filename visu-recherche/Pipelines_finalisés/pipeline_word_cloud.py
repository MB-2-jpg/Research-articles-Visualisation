import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize  import RegexpTokenizer 
import spacy
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
import re
import os

try: # on ignore la verification ssl qui peut poser probleme
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



# Télécharger les stop words 
nltk.download('stopwords')
nltk.download('punkt_tab')

def langageDetection(string_txt):
    lang = detect(string_txt)
    if lang == "en" :
        lang= "english"
    elif lang == "fr":
        lang = "french"
    return lang

une_lettre_digit = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
     'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]
def supprimer_stopwords(texte):
    # Tokenizer le texte
    langue = langageDetection(texte)
    mots = word_tokenize(texte)
    # Charger les stop words français

    stop_words = set(stopwords.words(langue))

    # Filtrer les mots qui ne sont pas des stop words
    mots_filtrés = [mot for mot in mots if mot.lower() not in stop_words ]
    print("mots_filtered:",mots_filtrés)
    # Rejoindre les mots filtrés en une seule chaîne de caractères
    texte_filtré = ' '.join(mots_filtrés)
    return texte_filtré

def scrape_page(url,output_file):
    try:
        # Envoyer une requête GET
        response = requests.get(url)
        response.raise_for_status()  # Lève une exception en cas d'erreur HTTP
        
        # Analyser le contenu HTML avec BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
        for span in soup.find_all('span'):
            span.string = ''  # Supprime le contenu de la balise span
        
        # Supprimer les balises inutiles (<script>, <style>, etc.)
        for script in soup(["script", "style"]):
            script.extract()

        # Extraire uniquement le conteneur du contenu principal (article)
        content_div = soup.find("div", id="mw-content-text")
        if content_div:
            # Extraire tous les paragraphes de l'article
            paragraphs = content_div.find_all("p")
            # Concaténer le texte des paragraphes avec 2 sauts de ligne entre eux
            page_text = "\n\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

        else:
            page_text = soup.get_text(separator='\n')
        
        # Si on veut tout extraire, directement:
        # page_text = soup.get_text(separator='\n')

        # Remplacer les occurrences de 3 sauts de ligne ou plus par exactement 2 sauts de ligne
        page_text = re.sub(r'\n{3,}', '\n\n', page_text)
        
        # Sauvegarder le texte dans un fichier
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(page_text)
        
        print("Les informations de la page ont été stockées dans " + output_file)
    
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du scraping: {e}")


# Exemple d'utilisation
url = "https://s2a.telecom-paris.fr/publications/"  # L'URL à scraper
url = "https://fr.wikipedia.org/wiki/S%C3%A9rie_de_Fourier"
#scrape_page(url,"file.txt")




# this is for tokenization using regular expressions   
#Structure generale : 
exp_reg = r"\w+|,"
 
#exemple d'expression regulière #exp_reg = r"[\w']+"  
# cette expression dénote l'ensemble des mots contenant des apostrophes 
#exp_reg = r'\d+'   
#cette expression dénote l'ensemble des numéros (one or more digit) #exp_reg = r'\w+|\d+' 
#les deux #Si on veut diviser le texte selon les espaces et la ponctuation, on utilise gaps=True 
#tokenizer = RegexpTokenizer(r'\s+', gaps=True) 
tokenizer = RegexpTokenizer(exp_reg) 
tokens = tokenizer.tokenize("Hello , c'est moi ") 
#print(tokens)


nom_fichier = "../web_scraping/s2a_thèses_1.txt"

def conversion_txt_string(nom_fichier):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of this script
    fichier_path = os.path.join(script_dir, "..", "web_scraping", "s2a_thèses_1.txt")
    fichier_path = os.path.normpath(fichier_path)  # clean up the path
    try:
        with open(nom_fichier, "r", encoding="utf-8") as fichier:
            contenu = fichier.read()
        return contenu
    except FileNotFoundError:
        return "Erreur : fichier non trouvé."
    except Exception as e:
        return f"Erreur : {e}"

fichier_string = supprimer_stopwords(conversion_txt_string(nom_fichier))
S= conversion_txt_string(nom_fichier)
S = tokenizer.tokenize(S)
 
Auteurs = []
marker = False
i =0
name =""
n =len(S)
while i< len(S)-2 :
    print(i)
    aut=[]
    word = S[i]
    if marker == False:
       if word == "members" :
    
            marker = True
            i+=1
            name =""

       else : 
            i+=1
    if marker == True :
            if word =="Thèse":
                marker == False
                Auteurs.append(aut)
                aut =[]
                i+=1
            else :
            
                
                while S[i] != ",":
                    name += " " +S[i]
                    
                    S.pop(i)
                print(name)
                S.insert(i, name)          
                i+=2
                aut.append(name)
                name = ""




tokenizer = RegexpTokenizer(exp_reg) 
tokens = S


nlp_fr = spacy.load("fr_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

def normalize_word(word, lang="fr"):
    """
    Renvoie la forme canonique d'un mot en supprimant les majuscules, le pluriel et la féminisation.
    
    :param word: Mot à normaliser (str)
    :param lang: Langue du mot ("fr" pour français, "en" pour anglais)
    :return: Mot normalisé (str)
    """
    # Convertir en minuscule
    word = word.lower()

    # Sélectionner le modèle spaCy en fonction de la langue
    if lang == "fr":
        doc = nlp_fr(word)
    elif lang == "en":
        doc = nlp_en(word)
    else:
        raise ValueError("Langue non supportée. Utilisez 'fr' pour français ou 'en' pour anglais.")

    # Lemmatisation (récupération de la forme canonique du mot)
    return doc[0].lemma_


tokens = [normalize_word(w) for w in tokens]
print(tokens)
def vocabularisation(tokens): #tokens est le résultat de la tokenisation du texte
    vocab_occ = {}
    liste_mots=[]
    for elt in tokens :
        if elt.casefold() in vocab_occ :
            vocab_occ[elt.casefold()] += 1
        elif elt.casefold() not in une_lettre_digit:
            vocab_occ[elt.casefold()] = 1
            liste_mots.append(elt.casefold())
    return liste_mots,vocab_occ


liste ,vocab = vocabularisation(tokens)
print(vocab)

nombre_mini_occ = 15

def amelioration_vocab(vocab):
    new_vocab = {}
    for elt in vocab :
        if vocab[elt]>=nombre_mini_occ:
            new_vocab[elt]=vocab[elt]
    return new_vocab


wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(amelioration_vocab(vocab))

# Affichage du nuage de mots
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # Supprimer les axes
plt.show()

'''
#L'autre code est mieux pour la gestion d'erreurs
from bs4 import BeautifulSoup
import requests

url = 'https://fr.wikipedia.org/wiki/Trou_noir'

page = requests.get(url)

if page.status_code != 200:
    print("Erreur lors du téléchargement de la page")

soup = BeautifulSoup(page.text, 'html.parser') '''

'''




# Extraire les titres:
titres = soup.find_all('h2')
for titre in titres:
    print(titre.get_text(strip=True))

# Extraire les lien: (peut être utile pour le crawling)

liens = soup.find_all('a')
for lien in liens:
    href = lien.get('href')
    texte = lien.get_text(strip=True)
    print(f"Texte: {texte} - Lien: {href}")

'''
