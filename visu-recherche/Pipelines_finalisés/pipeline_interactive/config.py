#from load_data import load_data
from filter_parameters import filtered_data, filtered_data_dic
from extract_pdf import extract_abstract_from_pdf
import concurrent.futures
print(" \n\n\n CONFIIIIIIIIIIG imported")
file_txt = "corpus_abstracts.txt"

def multithread_extraction_ordered(pdf_urls, max_workers=70):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(extract_abstract_from_pdf, pdf_urls))
    return results


def prepare_corpus() :    

    """Prépare le corpus de documents à partir des données chargées et filtrées.
    :return: Tuple contenant la liste des documents et le dictionnaire filtré.
    """
    keywords = None

    year = None


    type = None


    #authors = ["Roland  Badeau", "Pascal  Bianchi", "Philippe  Ciblat",
     #   "Stephan  Clémençon", "Florence  d'Alché-Buc", "Slim  Essid",
      #  "Olivier  Fercoq", "Pavlo  Mozharovskyi", "Geoffroy  Peeters",
      #  "Gaël  Richard", "François  Roueff", "Maria  Boritchev",
      #  "Radu  Dragomir", "Mathieu  Fontaine", "Ekhiñe  Irurozki",
      #  "Yann  Issartel", "Hicham  Janati", "Ons  Jelassi",
      #  "Matthieu  Labeau", "Charlotte  Laclau",
      #  "Laurence  Likforman-Sulem", "Yves  Grenier" ]

    authors = ["Matthieu  Labeau ","Pascal  Bianchi ","Philippe  Ciblat ","Stéphan  Clémençon ","Florence  d’Alché-Buc ","Slim  Essid", "Olivier  Fercoq ","Pavlo  Mozharovskyi ",
               "Geoffroy  Peeters ","Gaël  Richard","François  Roueff ","Maria  Boritchev ","Radu  Dragomir ","Mathieu  Fontaine ","Ekhiñe  Irurozki ",
        "Yann  Issartel ", "Hicham  Janati ", "Ons  Jelassi ",
        "Matthieu  Labeau ", "Charlotte  Laclau ",
        "Laurence  Likforman-Sulem ", "Yves  Grenier " ] # None pour ne pas filtrer par auteurs
    authors = ["Matthieu  Labeau "]
    #authors = ["Philippe  Ciblat "] # 2 docs insuffisants
    authors = ["Stéphan  Clémençon "]
    authors = ["Florence  d’Alché-Buc "]
    authors = ["Pascal  Bianchi"]
    hal_list = []
    documents = []
    Dictionary= load_data()
    rates=set()
    results= filtered_data(Dictionary, keywords, year, type, authors)
    c=0
    for hal_id, e in results.items():
        if e.get("pdf_link"):
            txt = extract_abstract_from_pdf(e["pdf_link"])
            #print(txt)
            documents.append(txt)
            hal_list.append(hal_id)
        else:
            print("No PDF link found for this entry.")
            c+=1
            rates.add(hal_id)
    print(f"Nombre de documents ratés : {c}")
 
    return documents, results, hal_list, authors, 

def prepare_corpus_autonome(Dictionary, Dic_parameters):

    results = filtered_data_dic(Dictionary, Dic_parameters)
    c=0
    rates = set()
    documents = []
    hal_list = []
    authors = Dic_parameters.get("authors")

    for hal_id, e in results.items():
        if e.get("pdf_link"):

            txt = extract_abstract_from_pdf(e["pdf_link"])
            #print(txt)
            documents.append(txt)
            hal_list.append(hal_id)
        else:
            print("No PDF link found for this entry.")
            c+=1
            rates.add(hal_id)
    print(f"Nombre de documents ratés : {c}")
 
    return documents, results, hal_list, authors, rates



def prepare_corpus_multithreading(Dictionary, Dic_parameters):
    results = filtered_data_dic(Dictionary, Dic_parameters)
    c=0
    rates = set()
    hal_list = []
    authors = Dic_parameters.get("authors",None)
    pdf_urls = []
    u=0
    # Rajout de tous les URLS dans la liste pdf_urls

    for hal_id, e in results.items():
        if e.get("pdf_link"):
            pdf_urls.append(e["pdf_link"])
            hal_list.append(hal_id)
            u+=1

        else : 
            print(f"No pdf found for this entry")
            c+=1
            rates.add(hal_id)
    print(f"Nombre de documents ratés : {c}")
    total = u + c
    print(f" Extraction de {u} / {u+c}")
    documents = multithread_extraction_ordered(pdf_urls, max_workers = 50)
 
    return documents, results, hal_list, authors, rates

  
    






 


if __name__ == "__main__":
    # Exemple d'utilisation de la fonction prepare_corpus
    from load_data import load_data
    Dictionary = load_data()
    documents, results, hal_list, authors, rates = prepare_corpus_multithreading(Dictionary, {"keywords" : None, "authors" : None, "year" : None})
    print(f"Nombre de documents préparés : {len(documents)}")

    
    
    #print(documents[0])  # Affiche les 5 premiers documents pour vérification
    #print(documents[1])
    #print(documents[2])
    #print(documents[3])
    #print(documents[4])
    
    # Affichage des titres, authors, keywords, pdf_link et type pour les 5 premiers résultats
    for hal_id,e in results.items():
        if hal_id in hal_list:
            print(f"HAL ID: {hal_id}")
            print(f"Title: {e.get('title')}")
            print(f"Authors: {e.get('authors')}")
            print(f"Keywords: {e.get('keywords')}")
            print(f"PDF Link: {e.get('pdf_link')}")
            print(f"Type: {e.get('type')}")
            print(f"Year: {e.get('year')}\n")

    with open(file_txt, "a", encoding="utf-8") as f:
        for i in range(len(documents)):
            f.write(f"\n\n\n\nhal_id    :    {hal_list[i]}\n\n\n\n")
            f.write(documents[i])
    f.close()
    print(f"Corpus saved to {file_txt}")
    print(f"Nombre de documents préparés : {len(documents)}")
    #print(f"Nombre de documents ratés : {c}")
    