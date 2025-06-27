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
                    if word =="Thèse":
                        marker == False
                        Auteurs.append(aut)
                        aut =[]
                        i+=1
                        break
                    name += " " +S[i]
                    
                    S.pop(i)
                print(name)
                S.insert(i, name)          
                i+=2
                aut.append(name)
                name = ""


Dict ={}
print(len(Auteurs))
for i in range(30):
    print(i)
    Dict[i] = Auteurs[i]