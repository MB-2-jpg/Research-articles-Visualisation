#from load_data import load_data

import unicodedata

#Dictionary = load_data()
def strip_accents(text: str) -> str:
        """Enlève les accents, renvoie en minuscules."""
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return text.lower()


def filtered_data(Dictionary, keywords=None, years=None, type=None, authors=None):
    """
    Filtre les données du dictionnaire en fonction des mots-clés, de l'année, du type et des auteurs.

    :param Dictionary: Dictionnaire contenant les données.
    :param keywords: Liste de mots-clés à filtrer (optionnel).
    :param year: Année à filtrer (optionnel).
    :param type: Type de publication à filtrer (optionnel).
    :param authors: Liste de noms d'auteurs à filtrer (optionnel).
    :return: Dictionnaire filtré.
    """
    filtered_dict = {}

    for hal_id in Dictionary.keys():
        data = Dictionary[hal_id]
        CANONICAL_TOKENS = {
        full: {strip_accents(t) for t in full.replace("-", " ").split()}
        for full in authors
    }
   
        # Filtre par auteurs
        if authors:
            data_authors = data["authors"] 
            Canonical_data={full: {strip_accents(t) for t in full.replace("-", " ").split()}
        for full in data_authors}
            if not (any(CANONICAL_TOKENS[author] in [Canonical_data[da] for da in data_authors] for author in authors)):
                continue

        # Filtre par mots-clés
        data_keywords = data["keywords"] 
   
        
        if keywords and data_keywords:
            Canonical_data_keywords={full: {strip_accents(t) for t in full.replace("-", " ").split()}
                for full in data_keywords}
            CANONICAL_KEYWORDS= {
                full: {strip_accents(t) for t in full.replace("-", " ").split()}
                for full in keywords
            }
            #print(keywords)
            if not any(CANONICAL_KEYWORDS[k] in [Canonical_data_keywords[dk] for dk in data_keywords] for k in keywords):
                continue

        # Filtre par année
        if years and not(int(data["year"]) <=  int(years[1])  and  int(data["year"]) >=  int(years[0])):

            continue


        
        # Filtre par type
        if type and data.get("type") not in type:
            continue

       
        # Si tout passe → on ajoute
        filtered_dict[hal_id] = data

    return filtered_dict

'''
if __name__ == "__main__":
    # Exemple d'utilisation de la fonction filtered_data
    keywords = None

    #keywords = ["deep learning"]
    type = None
    authors = ["Slim  Essid", "Bertrand  David"]
      # None pour ne pas filtrer par auteurs
    #year = "2022"
    year = None  # None pour ne pas filtrer par année  
    print("test")
    filtered_results = filtered_data(Dictionary, keywords, year,  type, authors)
    print(" Done filtering data.")
    print(filtered_results)
    # Affichage des résultats filtrés
    for hal_id, data in filtered_results.items():
        print(f"HAL ID: {hal_id}")
        print(f"Title: {data.get('title')}")
        print(f"Authors: {data.get('authors')}")
        print(f"Keywords: {data.get('keywords')}")
        print(f"Year: {data.get('year')}")
        print(f"Type: {data.get('type')}")
        print(f"PDF Link: {data.get('pdf_link')}\n")
'''