import ssl
from bs4 import BeautifulSoup
from urllib.request import urlopen 
import re




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


                S = []  # Liste des auteurs ou éléments avant "BibTeX"
                hal_id = None
                keywords = []
                pdf_link = None
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
                        match = re.search(r'\{(.+?)\}', token)
                        if match:
                            keywords = [k.strip() for k in match.group(1).split(';')]
                        else:
                            keywords = []
                    elif token.lower().startswith("pdf"):
                        match = re.search(r'\{(.+?)\}', token)
                        if match: 
                            pdf_link = match.group(1).strip()
                            #print(f"PDF link trouvé : {pdf_link}")
                    elif token.lower().startswith("url"):
                        match = re.search(r'\{(.+?)\}', token)
                        if match:
                            url_link = match.group(1).strip()
                            #print(f"URL link trouvé : {url_link}")
                    # Si un hal_id a été trouvé, on l’associe à ses keywords
                    elif token.lower().startswith("year"):
                        match = re.search(r'\{(.+?)\}', token)
                        if match:
                            year = match.group(1).strip()
                            #print(f"Année trouvée : {year}")
                        else:
                            year = None
                    if hal_id is not None:
                        Dictionary[str(hal_id)] = {"keywords": keywords if keywords else None, "title": S[0], "authors": S[1:] if len(S) > 1 else None, "pdf_link": pdf_link if pdf_link else url_link, "type" : "", "year" : year }
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
    load_data()