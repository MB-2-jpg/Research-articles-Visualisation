import ssl
from bs4 import BeautifulSoup
from urllib.request import urlopen 
import re
import codecs
import latexcodec  # juste pour enregistrer l'encodage "latex+"

print("\n\n\n\n\n LOAAAAD DATA IMPOOOOOOOOOOOORTED")
import requests

import requests


def remove_dashes(s):
    return s.replace("-", "")

def normalize_author_name(name: str) -> str:
    return ' '.join(name.strip().lower().split())

def clear_acolades(text):
    """
    Supprime uniquement les accolades { } mais garde leur contenu.
    
    :param text: La chaîne de caractères à nettoyer.
    :return: La chaîne de caractères sans accolades mais avec leur contenu intact.
    """
    return text.replace('{', '').replace('}', '').strip()

def scrap():

    url = 'https://s2a.telecom-paris.fr/publications/#2024'

    # Contourner la vérification SSL si besoin
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    # Ouvrir l'URL avec le contexte SSL désactivé
    U_client = urlopen(url, context=context)
    page_html = U_client.read()
    U_client.close()  # Fermer la connexion après lecture

    # Parser le contenu HTML avec BeautifulSoup
    page_soup = BeautifulSoup(page_html, "html.parser")

    # Supprimer les balises inutiles (<script>, <style>, etc.)
    for script in page_soup(["script", "style"]):
        script.extract()
    
    return page_soup




def extract_hal_id(Dictionary, bibliography, type):
        if bibliography:
            c = 0
            for li in bibliography.find_all("li"):
                text = li.get_text(" ", strip=False)  # Récupérer le texte proprement
                L = text.split("\n")
                #print(" ---------------------\n------------------------\n---------------------\n")
                #print(text)
        
                # Nettoyer la liste L en retirant les éléments vides
                L = [token.strip() for token in L if token.strip() != ""]

                pdf_link = None  # Initialiser pdf_link à None
                a = li.find("a", href=True)
                if a : 
                    pdf_link = a['href'] if a['href'].endswith(".pdf") else None
                    if pdf_link is None : 
                        a = li.find_next("a", href=True)
                        pdf_link = a['href'] if a and a['href'].endswith(".pdf") else None
                    
                #if pdf_link is not None:
                   # print(f"PDF link trouvé à travers a : {pdf_link}")

                S = []  # Liste des auteurs ou éléments avant "BibTeX"
                hal_id = None
                keywords = []
                for i, token in enumerate(L):
                    if token == "BibTeX":
                        S = [t.strip().rstrip(',') for t in L[:i-1]]
                    elif token.lower().startswith("hal_id"):
                    # Extraire le hal_id entre accolades
                        match = re.search(r'\{(.+?)\}', token)
                        if match:
                            hal_id = match.group(1).strip()
                            #print(f"HAL ID trouvé : {hal_id}")

                    

                    elif token.lower().startswith("keywords"):
                    # Extraire les keywords entre accolades
                        match = re.search(r'\{(.+)\}', token)
                        #print(f"Token keywords : {token}")
                        if match:
                            try  : 
                                keywords = [codecs.decode(codecs.encode(clear_acolades(k.strip()),"utf-8"),"latex") for k in match.group(1).split(';')]
                                keywords = [key.lower() for key in keywords]
                                keywords = [remove_dashes(key) for key in keywords]
                            except Exception as e:
                                #print(f"Erreur lors de l'extraction des mots-clés : {e}")
                                keywords = []
                        else:
                            keywords = []
                    elif token.lower().startswith("pdf"):
                        if pdf_link is None : 
                            match = re.search(r'\{(.+?)\}', token)
                            if match: 
                                pdf_tst = match.group(1).strip()
                                pdf_link = match.group(1).strip() if pdf_tst.endswith(".pdf") else None
                            #    print(f"\n\n\nPDF link trouvé FALLBACK: {pdf_link}\n\n\n")
                           # else : 
                             #   print("\n\n\nNo PDF link found in this entry.\n\n\n")
                    #elif token.lower().startswith("url"):
                        #if pdf_link is None:  # Si pdf_link n'a pas été trouvé précédemment
                            #match = re.search(r'\{(.+?)\}', token)
                            #if match:
                                #url_link = match.group(1).strip()
                                #if url_link.endswith(".pdf"):
                                 #   pdf_link = url_link
                                #print(f"URL link trouvé : {url_link}")
                            #else : print("\n\n\nNo PDF link found in this entry.\n\n\n")
                    # Si un hal_id a été trouvé, on l’associe à ses keywords
                    elif token.lower().startswith("year"):
                        match = re.search(r'\{(.+?)\}', token)
                        if match:
                            year = match.group(1).strip()
                            #print(f"Année trouvée : {year}")
                        else:
                            year = None
                    
                    if len(S) > 1 : 
                        authors = [ normalize_author_name(author) for author in S[1:]]
                    else : authors = None
                    
                    if hal_id is not None:
                        Dictionary[str(hal_id)] = {"keywords": keywords if keywords else None, "title": S[0], "authors": authors, "pdf_link": pdf_link , "type" : "", "year" : year }
                        if type is not None :
                            Dictionary[str(hal_id)]["type"] = type
                            #print(f"Type de publication ajouté : {type}")
           

def extract_all(Dictionary, page_soup): 
    for type in page_soup.find_all("h3", class_="bibliography"):
        type_name = type.get_text(strip=True)
        #print(f"Type de publication : {type_name}")
        bibliography = type.find_next("ol", class_="bibliography")
        extract_hal_id(Dictionary, bibliography, type_name)
        #print(f"Nombre d'articles extraits pour le type '{type_name}': {len(Dictionary)}\n")
    #print(Dictionary)
        
       


def load_data():
    Dic = {}
    page_soup = scrap()
    extract_all(Dic, page_soup)
    #print(Dic)  # Afficher le dictionnaire final
    return Dic



if __name__ == "__main__":
    Dic = load_data()
    print(len(Dic))
   