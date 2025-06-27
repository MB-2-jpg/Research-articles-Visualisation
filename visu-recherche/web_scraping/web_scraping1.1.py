import requests
from bs4 import BeautifulSoup
import re

def scrape_page(url):
    try:
        # Envoyer une requête GET
        response = requests.get(url)
        response.raise_for_status()  # Lève une exception en cas d'erreur HTTP
        
        # Analyser le contenu HTML avec BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
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
        with open("page_content.txt", "w", encoding="utf-8") as file:
            file.write(page_text)
        
        print("Les informations de la page ont été stockées dans 'page_content.txt'.")
    
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du scraping: {e}")

# Exemple d'utilisation
url = "https://fr.wikipedia.org/wiki/Trou_blanc"  # L'URL à scraper
scrape_page(url)