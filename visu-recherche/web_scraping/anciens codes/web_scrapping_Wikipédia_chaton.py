import ssl
from bs4 import BeautifulSoup
from urllib.request import urlopen 

# URL de la page Wikipédia sur les chatons
URL = 'https://fr.wikipedia.org/wiki/Chaton_(animal)'

# Contourner la vérification SSL
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



# Extraire uniquement les balises <p> et récupérer leur texte
paragraphs = page_soup.find_all("p")  # Trouver toutes les balises <p>
text_content = "\n\n".join(p.get_text(strip=True) for p in paragraphs)  # Récupérer le texte des <p>

'''Afficher le texte récupéré
print(text_content_cleaned)


 Ecrire le texte récupéré dans un fichier txt

file_path = "chaton_wikipedia.txt"
with open(file_path, "w", encoding="utf-8") as file:
    file.write(text_content)

print(f"Le contenu a été sauvegardé dans {file_path}") '''