import ssl
from bs4 import BeautifulSoup
from urllib.request import urlopen 

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

bibliography = page_soup.find("ol", class_="bibliography")

for tag in bibliography.find_all(["div", "button"]):
        tag.decompose()  # Supprime complètement l'élément du HTML

# Extraire le texte et les liens
references = []
if bibliography:
    for li in bibliography.find_all("li"):
        text = li.get_text(" ", strip=True)  # Récupérer le texte proprement
        link = li.find("a")["href"] if li.find("a") else None  # Récupérer le lien s'il existe
        references.append((text, link))

print(references[0][0])

file_path = "s2a_thèses.txt"
with open(file_path, "w", encoding="utf-8") as file:
    c=1
    for reference in references:
         file.write("\n Thèse :"+str(c))
         file.write(reference[0])
         c+=1
file.close()
         
    