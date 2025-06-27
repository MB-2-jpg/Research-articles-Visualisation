
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re



URL = 'https://s2a.telecom-paris.fr/members/'


u_client=urlopen(URL)
page_html=u_client.read()
u_client.close()


file_path = "s2a_members.txt"
page_soup=BeautifulSoup(page_html,'html.parser')


relevant_content = page_soup.find("section",class_= "page__content", itemprop = "text")

if relevant_content : 
    divs = relevant_content.find_all("div", class_= "col-8")
    if divs :
        c=1
        with open(file_path, "w", encoding="utf-8") as file:

            file.write("Members and Topics:\n\n")

            for div in divs :

                topic_tag= div.find('div')
                if topic_tag:

                    out_text = topic_tag.get_text("", strip=True)
                    topic_tag.decompose()
                    in_text = div.get_text("", strip=False)

                else:
                    print (" fail to get the content of the topic div")

                in_text = re.sub(r'\s+', ' ', in_text).strip()
                out_text = re.sub(r'\s+', ' ', out_text).strip()

                file.write(f"Member {c} : {in_text}\n")
                file.write(f"Topics : {out_text}\n")

                c+=1
    else : 
       print("Error divs not found")
else :
    print(" error : section not found ")








