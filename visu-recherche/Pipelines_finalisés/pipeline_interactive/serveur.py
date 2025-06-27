from flask import Flask, render_template, request, send_file, redirect, url_for
from flask_cors import CORS
import topic_modeling as tm
import nouvelle_anc as kw_anc
import threading
import graph_topic as gt
from lda_visualisation import visualisation, main
from threading import Thread, Lock
import shutil


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
app = Flask(__name__)
CORS(app)  # Autorise les requêtes cross-origin si besoin

@app.route('/')
def index():
    return render_template('index_main.html')



@app.route('/visualiser1', methods=['POST'])
def visualiser1():
    data = request.get_json()
    print("Reçu :", data)
    # Générez ici votre image (ex: 'result.png') à partir des données
   
    tm.run_topic_modeling(data['keywords'], [data['start_year'],data['end_year']])
    return send_file("result_ac.png", mimetype="image/png")



@app.route("/visualiserANC")
def visualiserANC():
    return render_template('index.html')




@app.route('/lancer_graphe', methods=['POST'])
def lancer_graphe():
    return redirect(url_for('page_graphes'))

@app.route('/graphes')
def page_graphes():
    return render_template('graphes.html')



@app.route("/visualiserGraphe",methods=['POST'])
def visualiserGraphe():
   data = request.get_json()
   print("Reçu :", data)
   gt.generate_graph(keywords=data['keywords'] , years=[data['start_year'],data['end_year']],type=None,authors= ["Mathieu Labeau" , 'François Roueff','Roland  Badeau','Gael  Richard'],type_graph="auteurs")
   return send_file("graph_image.png", mimetype="image/png")


@app.route('/visualiserTopics',methods = ["GET","POST"])
def visualiserTopics():
    selected = {}
    if request.method == "POST":
        selected["keywords"] = request.form.getlist("keywords")
        selected["authors"] = request.form.getlist("authors")
        selected["years"] = request.form.getlist("years")
        print(selected)
        selected["authors"] = [normalize_author_name(author) for author in selected["authors"]]

        #Analyse en correspondance d'abord
    
        data={}
        data["keywords"] = selected["keywords"]
        data["start_year"] = min([int(year) for year in selected["years"]])
        data["end_year"] = max([int(year) for year in selected["years"]])
        # Ton traitement
        kw_anc.run_topic_modeling(data['keywords'], [data['start_year'], data['end_year']])

        # Copie ou déplace l'image générée vers le dossier static
        shutil.copy("result_nouvelle_anc.png", "static/ma_visualisation.jpg")

    


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



@app.route('/visualiser2', methods=['POST'])
def visualiser2():

    data = request.get_json()
    
    print("Reçu :", data)
    # Générez ici votre image (ex: 'result.png') à partir des données

    kw_anc.run_topic_modeling(data['keywords'], [data['start_year'],data['end_year']])
    return send_file("result_nouvelle_anc.png", mimetype="image/png")




if __name__ == '__main__':
    app.run(debug=True , port=5050)
