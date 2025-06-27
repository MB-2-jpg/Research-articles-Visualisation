import ssl
from bs4 import BeautifulSoup
from urllib.request import urlopen
import fitz
import requests
import io

def extract_text_from_pdf_url(pdf_url):
    # Télécharger le PDF en tant que contenu binaire
    response = requests.get(pdf_url)

    try:
        response.raise_for_status()  # Vérifie que la requête s'est bien déroulée
    except requests.exceptions.HTTPError as e:
        print(f"Erreur HTTP pour l'URL {pdf_url}: {e}")
        # Vous pouvez choisir de renvoyer une chaîne vide, None, ou lever une exception personnalisée
        return False
    
    # Créer un objet BytesIO à partir du contenu du PDF
    pdf_stream = io.BytesIO(response.content)
    
    # Ouvrir le PDF depuis le flux
    document = fitz.open("pdf", pdf_stream)
    
    text = ""
    # Parcourir chaque page du PDF
    for page in document:
        text += page.get_text()
    return text



# URL de la page web
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


references=[]
for b in page_soup.find_all("ol", class_="bibliography"):
     if b :
        for tag in b.find_all(["div", "button"]):
            tag.decompose()  # Supprime complètement l'élément du HTML

        for li in b.find_all("li"):
            if li :
                text = li.get_text(" ", strip=True)  # Récupérer le texte proprement
                if li.find("a"):
                    link = li.find("a")["href"]
                    lien_presence= True
                else : 
                    lien_presence = False
                references.append((text, lien_presence,link))

print(len(references))
print(references[0][1])


file_path = "s2a_thèses_with_pdf.txt"
with open(file_path, "w", encoding="utf-8") as file:
    c=1
    for reference in references:
        file.write("\n\n\n\n\n Thèse "+str(c)+" :\n")
        file.write(reference[0])
         
        if reference[1]:
            file.write("\n Contenu PDF : \n")
            text=extract_text_from_pdf_url(reference[2])
            if type(text) is bool :
                c+=1
                continue
            else : 
                file.write(text)

        c+=1
file.close()






