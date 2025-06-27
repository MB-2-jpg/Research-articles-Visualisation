

import fitz  # PyMuPDF
import requests
import io
import re
import tempfile
from requests.exceptions import RequestException

print(" EXTRAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT")
BRUIT = """
•  Recherche\n
•  Accéder directement au contenu\n
•  Pied de page\n
[image] [image]\n
•  Documentation\n
•  FR\n
Français (FR)\n
Anglais (EN)\n
•  \n
Se connecter\n
•  Portail HAL Télécom Paris\n
•  Recherche\n
Loading...\n
Recherche avancée\n
Information de documents\n
Titres \n
○  Titres\n
○  Sous-titre\n
○  Titre de l'ouvrage\n
○  Titre\n
"""






def extract_text_from_pdf_url(pdf_url):
    # Télécharger le PDF en tant que contenu binaire
    response = requests.get(pdf_url)
    response.raise_for_status()  # Vérifie que la requête s'est bien déroulée
    
    # Créer un objet BytesIO à partir du contenu du PDF
    pdf_stream = io.BytesIO(response.content)
    
    # Ouvrir le PDF depuis le flux
    document = fitz.open("pdf", pdf_stream)
    
    text = ""
    # Parcourir chaque page du PDF
    for page in document:
        text += page.get_text()
    return text



def extract_abstract_from_pdf(pdf_url):

    print(f" Beginning extraction \n")
    if pdf_url is None : 
        return " No PDF link found for this entry."
       # Télécharge le PDF
    try:
        # ⏱️ Limite le temps d'attente à 10s
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
    except RequestException as e:
        return f"Erreur lors du téléchargement du PDF : {e}"

   
    # Sauvegarde temporaire du PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_pdf_path = tmp_file.name

    # Extraction de texte avec fitz
    try:
        doc = fitz.open(tmp_pdf_path)
    except Exception as e:
        return f"Erreur lors de l'ouverture du PDF avec fitz: {e}"
  

    text = ""
    
    # On lit généralement les 1ères pages pour l'abstract
    for page in doc:
        text += page.get_text()
    #print("text", text)
    doc.close()
    match = re.search(r'(ABSTRACT|Résumé)\s*(.+?)\s*(?=(INTRODUCTION|1\. INTRODUCTION|I\. INTRODUCTION))', 
                      text, re.IGNORECASE | re.DOTALL)

    if match:
        #print(" YEEEEEEEEEEEEEEEEEEES \n \n \n ABSTRAAAAAAAAAAAACT FOUUUUND")
        
        abstract = match.group(2).strip()

        if abstract.strip().startswith(BRUIT.strip()):  
            return "No abstract found in the PDF."
        
        txt_utile = abstract.split(" ")
        return " ".join(txt_utile[:200])
    
    return "No abstract found in the PDF."
        # Si pas d'abstract détecté, retourner le début jusqu'à "INTRODUCTION"
        #match_intro = re.search(r'^(.+?)\s*(INTRODUCTION|1\. INTRODUCTION|I\. INTRODUCTION)', 
                           #     text, re.IGNORECASE | re.DOTALL)
        #if match_intro:
            #return match_intro.group(1).strip()
        #else:
            # Dernier recours : retourner les 1500 premiers caractères nettoyés
            #return text[:1500].strip()
        


    


# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple d'utilisation de la fonction extract_text_from_pdf_url
    pdf_url = "https://inria.hal.science/hal-01219637v1/file/Magron-WASPAA-2015.pdf"  # Remplacez par l'URL réelle du PDF
    contenu_pdf = extract_abstract_from_pdf(pdf_url)
    print("contenu_pdf", contenu_pdf)

