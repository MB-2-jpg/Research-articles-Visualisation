import requests
from bs4 import BeautifulSoup

def scrape_page(url):
    try:
        # Envoyer une requête GET
        response = requests.get(url)
        response.raise_for_status()  # Lève une exception en cas d'erreur HTTP
        
        # Analyser le contenu HTML avec BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extraire le texte complet de la page (vous pouvez adapter cette étape selon vos besoins)
        page_text = soup.get_text(separator='\n')
        
        # Sauvegarder le texte dans un fichier
        with open("page_content.txt", "w", encoding="utf-8") as file:
            file.write(page_text)
        
        print("Les informations de la page ont été stockées dans 'page_content.txt'.")
    
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du scraping: {e}")

# Exemple d'utilisation
url = "https://s2a.telecom-paris.fr/publications/"  # L'URL à scraper
scrape_page(url)

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