from load_data import load_data
from filter_parameters import filtered_data



def search(Dictionary, hal_id) : 
    print(Dictionary[hal_id])


def scan_for_top_keywords(Dictionary):
    Freq_keywords={}
    for hal_id, e in Dictionary.items():
        key_list = e.get("keywords")
        if key_list:
            for key in key_list:
                Freq_keywords[key] = Freq_keywords.get(key,0) + 1
    sorted_list = sorted(Freq_keywords.items(), key= lambda x :  x[1], reverse=True)
    return sorted_list[:20]

def scan_for_top_authors(Dictionary):
    Freq_authors={}
    for hal_id, e in Dictionary.items():
        authors_list = e.get("authors")
        if authors_list:
            for author in authors_list:
                Freq_authors[author]= Freq_authors.get(author,0) + 1
    sorted_list = sorted(Freq_authors.items(), key = lambda x : x[1], reverse = True)
    return sorted_list

if __name__ == "__main__":
    # Example usage:
    Dictionary = load_data()
    #search(Dictionary, 'hal-02288519')

    for keyword,freq in scan_for_top_keywords(Dictionary):
        print(f"{keyword} : {freq}")
  
    