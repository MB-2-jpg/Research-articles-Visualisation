# Comment utiliser le git OCTIS (fourni par l'encadrant) 

## Objectif
Ce projet permet de comparer deux modèles de topic modeling, en utilisant des métriques de similarité et des visualisations.

## Pré-requis (ce qu'il faut installez)
Installer Octis : pip install octis
Il faut Python 3, et les bibliothèques suivantes (version spécifique indiqué si besoin) :

    - gensim>=4.2.0,<5.0
    - nltk
    - pandas
    - spacy
    - scikit-learn==1.1.0
    - scikit-optimize>=0.8.1
    - matplotlib
    - torch
    - numpy>=1.23.0,<2.0
    - libsvm
    - flask
    - sentence_transformers
    - requests
    - tomotopy
    - scipy<1.13

## Comment utiliser

il faut utiliser la commande : python compare_models.py model_a.json model_b.json --output output_folder/

### Charger les donnée

Il faut que les topics pour chacune des deux méthodes de topic modeling soientt au format Json (model_a.json model_b.json) , donc il faut 
adapter les pipelines (exemple ajouter dans chaque pipeline une fonction qui transforme la sortie en fichier json) pour pouvoir
faire la comparaison.


### Score de diversité

Avec Octis on peut obtenire un score de diversité , qui indique si les différents topic sont uniques .  Il mesure la proportion de mots uniques parmi les top-k mots de chaque topic.
Il faut que l'entrée soit de la forme :
model_output = {
    "topics": [
        ["mot1", "mot2", ..., "motk"],  # top-k mots du topic 1
        ["motA", "motB", ..., "motk"],   # top-k mots du topic 2
        ...
    ]
}
