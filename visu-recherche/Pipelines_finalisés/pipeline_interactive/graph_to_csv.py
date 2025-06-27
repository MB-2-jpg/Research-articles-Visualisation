# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:39:03 2025

@author: alice
"""

# Ce module permet d'écrire dans un fichier csv un graphe codé 
# par liste d'adjacence avec un dictionnaire


def write_to_file(graph:dict, file_name:str):
    """save the graph in the file 'graphes/csv/file_name.csv'"""
    assert (type(file_name)==str and type(graph)==dict), "type des arguments incorrect"
    file = open("graphes/csv/"+file_name+".csv", "w")
    
    for key in graph.keys():
        str_key = str(key)
        file.write(str_key)
        
        if type(graph[key])!=list:
            file.close()
            assert False, "format incorrect : la valeur associée à la clé {} n'est pas une list".format(key)
        for voisin in graph[key]:
            str_voisin = str(voisin)
            file.write(";")
            file.write(str_voisin)
        file.write("\n")
        
    file.close()

def test():
    
    my_graph = {"paris":["lille", "bordeaux", "lyon"], "bordeaux":["paris", "lyon"], "lyon":["paris", "bordeaux"]}
    write_to_file(my_graph, "toto")
  