#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construire un dictionnaire (clé -> [auteurs, keywords, année, titre])
à partir des entrées BibTeX présentes dans article.txt
"""

import re
import json
import unicodedata
from pathlib import Path
from graph_to_csv import write_to_file
from filter_parameters_graph import filtered_data
from load_data_graph import load_data
import networkx as nx
import subprocess
import matplotlib.pyplot as plt

dico = load_data()
#print("dico :", dico)

#dico = filtered_data(dico,None,"2020",None,None) # on prend les articles de 2020
#print(dico)

TARGET_AUTHORS = {
        "Roland Badeau", "Pascal Bianchi", "Philippe Ciblat",
        "Stephan Clémençon", "Florence d'Alché-Buc", "Slim Essid",
        "Olivier Fercoq", "Pavlo Mozharovskyi", "Geoffroy Peeters",
        "Gaël Richard", "François Roueff", "Maria Boritchev",
        "Radu Dragomir", "Mathieu Fontaine", "Ekhiñe Irurozki",
        "Yann Issartel", "Hicham Janati", "Ons Jelassi",
        "Matthieu Labeau", "Charlotte Laclau",
        "Laurence Likforman-Sulem", "Yves Grenier"
    }
def en_commun(liste1,liste2):
    temp1 = [elem.lower() for elem in liste1]
    temp2 = [elem.lower() for elem in liste2]
    return any(elem in temp1 for elem in temp2) #regarde s'il y a des chaine de caractere en commun

def en_commun(liste1,liste2):
    if(liste1 is None ):
        return False
    if(liste2 is None ):
        return False

    temp1 = [elem.lower() for elem in liste1]
    temp2 = [elem.lower() for elem in liste2]
    
    return any(elem in temp1 for elem in temp2) #regarde s'il y a des chaine de caractere en commun
def id_to_title(id):
    #print(database[id][-1][1:])
    return dico[id]['title']

def id_to_authors(id):
    return dico[id]['authors']

def id_to_keywords(id):
    if (dico[id]['keywords'] is None):
        return []
    return dico[id]['keywords']

def article_permanent(noms):
    for permanent in TARGET_AUTHORS:
        if permanent in noms :
            return True
    return False
def generate_graph_authors():
    res = {}
    taille=0
    for id1 in dico.keys():
        res[id1]=[]
        for id2 in dico.keys():
            if id1!=id2:
                liste1 = id_to_authors(id1)
                liste2 = id_to_authors(id2)
                if(en_commun(liste1,liste2)) :
                    taille+=1
                    res[id1].append(id2)
                
            
    return res

def generate_graph_topic():
    res = {}
    taille=0
    for id1 in dico.keys():
        res[id1]=[]
        for id2 in dico.keys():
            if id1!=id2:
                liste1 = id_to_keywords(id1)
                liste2 = id_to_keywords(id2)
                if(en_commun(liste1,liste2)) :
                    taille+=1
                    res[id1].append(id2)
                
            
    return res

def authors_article():
    res = {}
    for id in dico.keys():
        for nom in id_to_authors(id):
            res[nom]= []
    for id in dico.keys():
        for nom in id_to_authors(id):
            res[nom] += [id]
    return res


keywords_dic_authors={}
def generate_graph_authors_cowork():
    res = {}
    dictionnaire = authors_article()
    for nom1 in dictionnaire.keys():
        res[nom1]=[]
        temp = []
        for elt in dictionnaire[nom1]:
            temp+= id_to_keywords(elt) 
        keywords_dic_authors[nom1] = temp
        for nom2 in dictionnaire.keys():
            if nom1!=nom2:
                liste1 = dictionnaire[nom1]
                liste2 = dictionnaire[nom2]
                if(en_commun(liste1,liste2)) :
                    res[nom1].append(nom2)
                          
    return res

def generate_graph_authors_sametheme():
    res = {}
    dictionnaire = authors_article()
    for nom1 in dictionnaire.keys():
        res[nom1]=[]
        temp = []
        for elt in dictionnaire[nom1]:
            temp+= id_to_keywords(elt) 
        keywords_dic_authors[nom1] = temp
    for nom1 in dictionnaire.keys():
        for nom2 in dictionnaire.keys():
            if nom1!=nom2:
                liste1 = keywords_dic_authors[nom1]
                liste2 = keywords_dic_authors[nom2]
                if(en_commun(liste1,liste2)) :
                    res[nom1].append(nom2)              
    return res

def transfo(graph):
    res = {}
    for keys in graph.keys():
        res[id_to_title(keys)]=[id_to_title(element) for element in graph[keys]]
    return graph

def author_dic(dictionnary):
    res ={}
    for id in dictionnary.keys():
        sum = ''
        liste_nom = id_to_authors(id)
        for elt in liste_nom:
            sum+=elt + ';'
        sum = sum[:-1]
        res[id] = sum
    return res

def keywords_dic(dictionnary):
    res ={}
    for id in dictionnary.keys():
        sum = ''
        liste_words = id_to_keywords(id)
        for elt in liste_words:
            sum+=elt + ';'
        sum = sum[:-1]
        res[id] = sum
    return res

def create_gephi_graph(graph: dict, auteur_dict: dict, auteur_cible: str, file_name: str):
    G = nx.Graph()
    #print(auteur_dict)
    # Ajoute toutes les arêtes
    for source, voisins in graph.items():
        for target in voisins:
            G.add_edge(source, target)

    # Ajoute les attributs aux nœuds
    for node in G.nodes:
        auteurs = auteur_dict.get(node, "")
        titre = id_to_title(node)
        cible = "oui" if auteur_cible.lower() in auteurs.lower() else "non"

        G.nodes[node]['auteurs'] = auteurs
        G.nodes[node]['auteur_cible'] = cible
        G.nodes[node]['label'] = titre  # Le titre est l'étiquette visible dans Gephi

        if cible == "oui":
            G.nodes[node]['viz'] = {'color': {'r': 255, 'g': 80, 'b': 80, 'a': 1.0}}
        else:
            G.nodes[node]['viz'] = {'color': {'r': 180, 'g': 180, 'b': 180, 'a': 0.7}}

    nx.write_gexf(G, f"graphes/{file_name}.gexf")


def create_gephi_graph_2(graph: dict, auteur_dict: dict, keywords_dict: dict,auteurs_cible: list, keywords_cible: list, file_name: str):
    G = nx.Graph()
    #print(auteur_dict)
    # Ajoute toutes les arêtes
    for source, voisins in graph.items():
        for target in voisins:
            G.add_edge(source, target)

    # Ajoute les attributs aux nœuds
    for node in G.nodes:
        auteurs = auteur_dict.get(node, "")
        keywords = keywords_dict.get(node, "")
        titre = id_to_title(node)
        cible= "non"
        for noms in auteurs_cible :
            if noms.lower() in auteurs.lower() :
                cible = "oui"
        if(auteurs_cible==[]):
            cible = "oui"
        for words in keywords_cible :
            if cible == "oui" :
                if words.lower() not in keywords.lower() :
                    cible = "non"
        #print(cible)
        G.nodes[node]['auteurs'] = auteurs
        G.nodes[node]['mots-clefs'] = keywords
        G.nodes[node]['cible'] = cible
        G.nodes[node]['label'] = titre  # Le titre est l'étiquette visible dans Gephi
        if article_permanent(auteurs):
            G.nodes[node]['viz'] = {'color': {'r': 255, 'g': 80, 'b': 255, 'a': 1.0}}
        elif cible == "oui":
            G.nodes[node]['viz'] = {'color': {'r': 255, 'g': 80, 'b': 80, 'a': 1.0}}
        else:
            G.nodes[node]['viz'] = {'color': {'r': 180, 'g': 180, 'b': 180, 'a': 0.7}}

    nx.write_gexf(G, f"graphes/{file_name}.gexf")

def create_gephi_graph_authors(graph: dict, auteur_dict: dict, keywords_dict: dict,auteurs_cible: list, keywords_cible: list, file_name: str):
    G = nx.Graph()
    #print(auteur_dict)
    # Ajoute toutes les arêtes
    for source, voisins in graph.items():
        for target in voisins:
            G.add_edge(source, target)

    # Ajoute les attributs aux nœuds
    for node in G.nodes:
        auteurs = auteur_dict.get(node, "")
        keywords = keywords_dict.get(node, "")
        cible= "non"
        
        for noms in auteurs_cible :
            if noms in node :
                cible = "oui"

            else:
                for elt in graph[node]:
                    if noms.lower() in elt.lower():
                        cible = "oui"
        if(auteurs_cible==[]):
            cible = "oui"
        for words in keywords_cible :
            if cible == "oui" :
                if all(words.lower() not in elt.lower() for elt in  keywords_dic_authors[node]):
                    cible = "non"
        G.nodes[node]['auteurs'] = auteurs
        G.nodes[node]['mots-clefs'] = keywords
        G.nodes[node]['cible'] = cible
        G.nodes[node]['label'] = node  # Le titre est l'étiquette visible dans Gephi

        if cible == "oui":
            G.nodes[node]['viz'] = {'color': {'r': 255, 'g': 80, 'b': 80, 'a': 1.0}}
        else:
            G.nodes[node]['viz'] = {'color': {'r': 180, 'g': 180, 'b': 180, 'a': 0.7}}
     

    nx.write_gexf(G, f"graphes/{file_name}.gexf")

#print(transfo((generate_graph_topic())))
#print(len(transfo((generate_graph_topic())).keys()))
fichier_gexf = "fichier_de_test"
create_gephi_graph_authors(((generate_graph_authors_sametheme())),author_dic(dico),keywords_dic(dico),[],["AI"],fichier_gexf)
#create_gephi_graph_2(((generate_graph_topic())),author_dic(dico),keywords_dic(dico),[],[],fichier_gexf)
#create_gephi_graph_2(((generate_graph_authors())),author_dic(dico),keywords_dic(dico),[],[],fichier_gexf)

#write_to_file(transfo (generate_graph_topic()),"topic_common_new") # A l'ancienne
#write_to_file(transfo (generate_graph_authors()),"author_common_new")
def ouvrir_gephi(fichier_gexf):
    chemin_gephi = "/Applications/Gephi.app/Contents/MacOS/Gephi"
    subprocess.run([chemin_gephi, "graphes/"+fichier_gexf +".gexf" ])

#ouvrir_gephi(fichier_gexf)

G = nx.read_gexf("graphes/"+fichier_gexf +".gexf")
# Création d'une disposition (ex: spring layout)
pos = nx.spring_layout(G, seed=42, k=0.5)

# Définir les couleurs à partir de l'attribut 'auteur_cible'
colors = []
for node in G.nodes(data=True):
    if article_permanent(node[1]['auteurs']):
        colors.append('purple')
        if node[1].get('cible') != 'oui': 
            node[1]['label']= ""
    elif node[1].get('cible') == 'oui':     
        colors.append('red')
    else:
        node[1]['label']= ""
        colors.append('lightgray')

edge_colors = []
for u, v in G.edges():
    if (G.nodes[u].get('cible') == 'oui') and (G.nodes[v].get('cible') == 'oui'):
        edge_colors.append('red')  # lien entre deux cibles
    else:
        edge_colors.append('gray')  # lien normal
# Récupérer les labels (par exemple depuis l'attribut "label")
labels = {node: data.get('label', node) for node, data in G.nodes(data=True)}
#print(dico)
def generate_graph(dic=dico ,keywords=['Artificial Intelligence', 'Multidisplinary Approach', 'Explainability', 'interpretability', 'Neural networks', 'Hybrid AI', 'Law', 'Regulation', 'Safety', 'Liability', 'Fairness', 'Accountability', 'Cost-benefit analysis','Analyse des signaux sociaux', 'états émergents', 'cohésion'] , years=[2020,2024],type=None,authors= ["Mathieu Labeau" , 'François Roueff','Roland  Badeau','Gael  Richard'],type_graph="auteurs"):
    print("dic: ",dic)
    result={}
    dico =  filtered_data(dic,None,None,None,authors) 
    print("dico:", dico)
    if type_graph == "auteurs":
        res = {}
        taille=0
        for id1 in dico.keys():
            res[id1]=[]
            for id2 in dico.keys():
                if id1!=id2:
                    liste1 = id_to_authors(id1)
                    liste2 = id_to_authors(id2)
                    if(en_commun(liste1,liste2)) :
                        taille+=1
                        res[id1].append(id2)
                    
        #print("res === :", res)
        result = res

    elif type_graph == "topics":
      
        res = {}
        taille=0
        for id1 in dico.keys():
            res[id1]=[]
            for id2 in dico.keys():
                if id1!=id2:
                    liste1 = id_to_authors(id1)
                    liste2 = id_to_authors(id2)
                    if(en_commun(liste1,liste2)) :
                        taille+=1
                        res[id1].append(id2)
                    
            
        result = res





    elif type_graph =="auteurs_coworking":

            res = {}
            dictionnaire = authors_article()
            for nom1 in dictionnaire.keys():
                res[nom1]=[]
                temp = []
                for elt in dictionnaire[nom1]:
                    temp+= id_to_keywords(elt) 
                keywords_dic_authors[nom1] = temp
                for nom2 in dictionnaire.keys():
                    if nom1!=nom2:
                        liste1 = dictionnaire[nom1]
                        liste2 = dictionnaire[nom2]
                        if(en_commun(liste1,liste2)) :
                            res[nom1].append(nom2)
                                
            result = res

    elif type_graph == "auteurs_meme_theme":
        res = {}
        dictionnaire = authors_article()
        for nom1 in dictionnaire.keys():
            res[nom1]=[]
            temp = []
            for elt in dictionnaire[nom1]:
                temp+= id_to_keywords(elt) 
            keywords_dic_authors[nom1] = temp
        for nom1 in dictionnaire.keys():
            for nom2 in dictionnaire.keys():
                if nom1!=nom2:
                    liste1 = keywords_dic_authors[nom1]
                    liste2 = keywords_dic_authors[nom2]
                    if(en_commun(liste1,liste2)) :
                        res[nom1].append(nom2)              
        result = res


    elif type_graph =="auteurs_articles":
        res = {}
        for id in dico.keys():
            for nom in id_to_authors(id):
                res[nom]= []
        for id in dico.keys():
            for nom in id_to_authors(id):
                res[nom] += [id]
        result =res
    
    #print (result)
    create_gephi_graph_2(result, author_dic(result), keywords_dic(result),author_dic(result), keywords_dic(result),  "graphe1")
    G = nx.read_gexf("graphes/"+"graphe1" +".gexf")
    # Création d'une disposition (ex: spring layout)
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Définir les couleurs à partir de l'attribut 'auteur_cible'
    colors = []
    for node in G.nodes(data=True):
        if article_permanent(node[1]['auteurs']):
            colors.append('purple')
            if node[1].get('cible') != 'oui': 
                node[1]['label']= ""
        elif node[1].get('cible') == 'oui':     
            colors.append('red')
        else:
            node[1]['label']= ""
            colors.append('lightgray')

    edge_colors = []
    for u, v in G.edges():
        if (G.nodes[u].get('cible') == 'oui') and (G.nodes[v].get('cible') == 'oui'):
            edge_colors.append('red')  # lien entre deux cibles
        else:
            edge_colors.append('gray')  # lien normal
    # Récupérer les labels (par exemple depuis l'attribut "label")
    labels = {node: data.get('label', node) for node, data in G.nodes(data=True)}

    # Dessin du graphe
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, font_color='black')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig("graph_image.png", dpi=300)
    #plt.show()




#generate_graph()