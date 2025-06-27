
from flask import Flask, redirect, render_template, request, url_for
from lda_visualisation import visualisation, main
from threading import Thread, Lock

#Fonctions de normalisation
def normalize_author_name(name: str) -> str:
    return ' '.join(name.strip().lower().split())

app = Flask(__name__)

keywords = [
    "machine learning",
    "deep learning",
    "audio source separation",
    "anomaly detection",
    "natural language processing",
    "speech enhancement",
    "representation learning",
    "nonnegative matrix factorization",
    "variational inference",
    "artificial intelligence"
]

authors = [ "Roland Badeau", "Pascal Bianchi", "Philippe Ciblat", "Stephan Clémençon", "Florence d'Alché-Buc", "Slim Essid", "Olivier Fercoq", "Pavlo Mozharovskyi", "Geoffroy Peeters","Gaël Richard", "François Roueff", "Maria Boritchev",     "Radu Dragomir", "Mathieu Fontaine", "Ekhiñe Irurozki",     "Yann Issartel", "Hicham Janati", "Ons Jelassi",     "Matthieu Labeau", "Charlotte Laclau",     "Laurence Likforman-Sulem", "Yves Grenier" ]
years = [str(y) for y in range(2025, 2014, -1)]

@app.route("/", methods=["GET", "POST"])
def index():
    selected = {}
    if request.method == "POST":
        selected["keywords"] = request.form.getlist("keywords")
        selected["authors"] = request.form.getlist("authors")
        selected["years"] = request.form.getlist("years")
        print(selected)
        selected["authors"] = [normalize_author_name(author) for author in selected["authors"]]
        if __name__ == "__main__":
            from load_data import load_data
            from config import prepare_corpus_multithreading
            print(f"Loading data ...")
            Dictionary = load_data()
            print(f"Load data completed")
            print(f"Corpus prep ...")
            Documents, results, hal_list, authors_selected, rates = prepare_corpus_multithreading(Dictionary,selected)
            print(f"Corpus prep completed")
            print(f" Starting LDA")
            results = main(6, Documents, results, hal_list)
            print(f"LDA completed")
            print(f"Visualising...")
            visualisation(results)
            print(f"Done visualising")
            
            return render_template("index.html", keywords=keywords, authors=authors, years=years, selected=selected, show_vis = 1)
            

    show_vis = request.args.get('show_vis') == '1'    

    return render_template("index.html", keywords=keywords, authors=authors, years=years, selected=selected, show_vis = show_vis)

if __name__ == "__main__":
    app.run(debug=True)


