# -*- coding: utf-8 -*-
"""
Script pour filtrer les articles par auteurs spécifiques
"""
import re
from typing import List, Set

# Liste des auteurs cibles
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

def load_articles_file(filename: str) -> str:
    """Charge le contenu du fichier d'articles"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{filename}' n'a pas été trouvé.")
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return None

def parse_articles(content: str) -> List[dict]:
    """Parse le contenu et extrait les articles individuels"""
    
    # Pattern pour détecter le début d'un article
    article_pattern = r'========== Article #(\d+): (.+?) =========='
    
    # Trouver tous les marqueurs d'articles
    matches = list(re.finditer(article_pattern, content))
    
    articles = []
    
    for i, match in enumerate(matches):
        article_number = match.group(1)
        article_filename = match.group(2)
        start_pos = match.end()
        
        # Déterminer la fin de l'article (début du suivant ou fin du fichier)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        # Extraire le contenu de l'article
        article_content = content[start_pos:end_pos].strip()
        
        # Créer l'objet article
        article = {
            'number': int(article_number),
            'filename': article_filename,
            'header': match.group(0),  # Le header complet
            'content': article_content,
            'full_text': content[match.start():end_pos]  # Header + contenu
        }
        
        articles.append(article)
    
    return articles

def find_authors_in_text(text: str, target_authors: Set[str]) -> List[str]:
    """Trouve les auteurs cibles présents dans le texte"""
    found_authors = []
    
    # Convertir le texte en minuscules pour la recherche insensible à la casse
    text_lower = text.lower()
    
    for author in target_authors:
        # Recherche insensible à la casse
        if author.lower() in text_lower:
            found_authors.append(author)
    
    return found_authors

def filter_articles_by_authors(articles: List[dict], target_authors: Set[str]) -> List[dict]:
    """Filtre les articles qui contiennent au moins un auteur cible"""
    
    filtered_articles = []
    
    print("=== ANALYSE DES ARTICLES ===")
    print(f"Nombre total d'articles : {len(articles)}")
    print(f"Auteurs recherchés : {len(target_authors)}")
    print()
    
    for article in articles:
        # Chercher les auteurs dans tout le texte de l'article
        found_authors = find_authors_in_text(article['full_text'], target_authors)
        
        if found_authors:
            article['found_authors'] = found_authors
            filtered_articles.append(article)
            
            print(f"✅ Article #{article['number']} ({article['filename']})")
            print(f"   Auteurs trouvés: {', '.join(found_authors)}")
            print()
        else:
            print(f"❌ Article #{article['number']} ({article['filename']}) - Aucun auteur cible")
    
    print(f"\n=== RÉSUMÉ ===")
    print(f"Articles conservés : {len(filtered_articles)} / {len(articles)}")
    print(f"Pourcentage : {(len(filtered_articles)/len(articles)*100):.1f}%")
    
    return filtered_articles

def save_filtered_articles(filtered_articles: List[dict], output_filename: str):
    """Sauvegarde les articles filtrés dans un nouveau fichier"""
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as file:
            for i, article in enumerate(filtered_articles):
                # Écrire le contenu complet de l'article (header + contenu)
                file.write(article['full_text'])
                
                # Ajouter une séparation entre les articles (sauf pour le dernier)
                if i < len(filtered_articles) - 1:
                    file.write('\n')
        
        print(f"✅ Articles filtrés sauvegardés dans : {output_filename}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {e}")

def create_authors_report(filtered_articles: List[dict], target_authors: Set[str]):
    """Crée un rapport détaillé sur les auteurs trouvés"""
    
    # Compter les occurrences de chaque auteur
    author_counts = {}
    total_articles_with_authors = len(filtered_articles)
    
    for article in filtered_articles:
        for author in article['found_authors']:
            author_counts[author] = author_counts.get(author, 0) + 1
    
    print(f"\n=== RAPPORT DES AUTEURS ===")
    print(f"Total d'articles avec auteurs cibles : {total_articles_with_authors}")
    print()
    
    # Trier par nombre d'occurrences
    sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Auteurs trouvés (classés par fréquence) :")
    for author, count in sorted_authors:
        print(f"  • {author}: {count} article(s)")
    
    # Auteurs non trouvés
    found_author_names = set(author_counts.keys())
    missing_authors = target_authors - found_author_names
    
    if missing_authors:
        print(f"\nAuteurs non trouvés ({len(missing_authors)}) :")
        for author in sorted(missing_authors):
            print(f"  • {author}")

# SCRIPT PRINCIPAL
def main():
    # Nom du fichier d'entrée (à modifier selon votre fichier)
    input_filename = "articles_permanents.txt"  # Remplacez par le nom de votre fichier
    output_filename = "articles_permanents_filtered.txt"
    
    print("=== FILTRAGE D'ARTICLES PAR AUTEURS ===")
    print(f"Fichier d'entrée : {input_filename}")
    print(f"Fichier de sortie : {output_filename}")
    print()
    
    # 1. Charger le fichier
    content = load_articles_file(input_filename)
    if content is None:
        return
    
    print(f"Fichier chargé : {len(content)} caractères")
    
    # 2. Parser les articles
    articles = parse_articles(content)
    
    if not articles:
        print("❌ Aucun article trouvé dans le fichier.")
        print("Vérifiez le format du fichier et le pattern de séparation.")
        return
    
    # 3. Filtrer par auteurs
    filtered_articles = filter_articles_by_authors(articles, TARGET_AUTHORS)
    
    if not filtered_articles:
        print("❌ Aucun article ne contient les auteurs recherchés.")
        return
    
    # 4. Sauvegarder le résultat
    save_filtered_articles(filtered_articles, output_filename)
    
    # 5. Créer un rapport détaillé
    create_authors_report(filtered_articles, TARGET_AUTHORS)
    
    print(f"\n✅ Filtrage terminé avec succès !")

if __name__ == "__main__":
    main()