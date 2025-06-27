import ssl
from bs4 import BeautifulSoup
from urllib.request import urlopen 
import re






        



def load_keywords():

    Dic = {}
    # URL de la page Wikipédia sur les chatons
    URL = 'https://s2a.telecom-paris.fr/publications/#2024'

    # Contourner la vérification SSL si besoin
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    # Ouvrir l'URL avec le contexte SSL désactivé
    U_client = urlopen(URL, context=context)
    page_html = U_client.read()
    U_client.close()  # Fermer la connexion après lecture

    # Parser le contenu HTML avec BeautifulSoup
    page_soup = BeautifulSoup(page_html, "html.parser")



    # Supprimer les balises inutiles (<script>, <style>, etc.)
    for script in page_soup(["script", "style"]):
        script.extract()


    for b in page_soup.find_all("ol", class_="bibliography"):
        if b:
            c = 0
            for li in b.find_all("li"):
                text = li.get_text(" ", strip=False)  # Récupérer le texte proprement
                L = text.split("\n")
                print(" ---------------------\n------------------------\n---------------------\n")
                print(text)
        
                # Nettoyer la liste L en retirant les éléments vides
                L = [token.strip() for token in L if token.strip() != ""]


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
                            print(f"HAL ID trouvé : {hal_id}")

                    elif token.lower().startswith("keywords"):
                    # Extraire les keywords entre accolades
                        match = re.search(r'\{(.+?)\}', token)
                        if match:
                            keywords = [k.strip() for k in match.group(1).split(';')]
                        else:
                            keywords = []

            # Si un hal_id a été trouvé, on l’associe à ses keywords
                if hal_id is not None:
                    Dic[str(hal_id)] = {"keywords": keywords if keywords else None, "title": S[0], "authors": S[1:] if len(S) > 1 else None}

    return Dic