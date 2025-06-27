import ssl
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
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

file_path = "web_scraping/s2a_thèses_1.txt"




with open(file_path, "w", encoding="utf-8") as file:
    for b in page_soup.find_all("ol", class_="bibliography"):
        c=0
        if b:
            for tag in b.find_all(["div", "button"]):
                tag.decompose()  # Supprime complètement l'élément du HTML

            for li in b.find_all("li"):
                if li:
                    c+=1
                    before_nodes = []

                    after_nodes = []

                    found_br = False
                    for tag in li.find_all(['div','pre','button']):
                        tag.decompose()  # Supprime complètement l'élément du HTML   
                     
                    for elem in li.contents:
                        if isinstance(elem, Tag) and elem.name == 'br':
                            if not found_br:
                                found_br = True
                                continue  # on ignore les balises <br> elles-mêmes
                        if not found_br:
                            before_nodes.append(elem)
                        else:
                            after_nodes.append(elem)

# Créer des nouveaux fragments HTML à partir des deux parties
                    before_html = ''.join(str(e) for e in before_nodes)
                    after_html = ''.join(str(e) for e in after_nodes)
                    before_soup = BeautifulSoup(before_html, 'html.parser')
                    after_soup = BeautifulSoup(after_html, 'html.parser')


# Parser ces fragments pour utiliser .get_text() proprement
            
                    title = before_soup.get_text("", strip=True)
                    members = after_soup.get_text(" ", strip=True).split(",")
                    file.write("\n Thèse "+str(c)+" :")
                    file.write(title)
                    file.write("\n@members: ")
                    
                    for member in members: #[:-4] if len(members) > 4 else members:
                        m=member.strip()
                        if len(m.split(" ")) !=2:
                           file.write(' '.join(m.split(" ")[:2]))
                           break
                        file.write(m)
                        file.write(", ")


                    
