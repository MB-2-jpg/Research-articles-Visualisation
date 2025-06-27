-Mail reçu le 24/01/2025 "	[Ing-1a] Projets Artishow (version longue)":
"vous serez /davantage/ notés sur votre faculté à
vous organiser pour travailler, que sur votre maîtrise finale"
"Un point d’étape noté avec l’encadrant sera réalisé en fin de P3 en
fonction des attendus définis au début de semestre, et une présentation
aura lieu le 27 juin dans le hall de l’école, où chaque projet
présentera une démonstration de son projet, et un poster explicatif, à
des évaluateurs."

Résumés des références bibliographiques:
### KeywordScape: Visual Document Exploration using Contextualized Keyword Embeddings

Le document présente **KeywordScape**, un outil de visualisation interactif conçu pour explorer et comparer des documents en se basant sur des mots-clés contextualisés. Voici un résumé et une explication détaillée, en lien avec le projet Artishow :

---

### **Contexte et Problématique**

- **Motivation :**  
  Les méthodes traditionnelles de visualisation de documents reposent souvent sur des représentations statiques de mots-clés (par exemple, des nuages de mots ou des graphiques basés sur la fréquence). Toutefois, ces approches supposent que chaque mot a une signification fixe, ce qui n’est pas réaliste puisque le sens d’un mot peut varier selon son contexte (par exemple, « training » en machine learning versus psychologie).

- **Objectif :**  
  KeywordScape vise à tirer parti des **embeddings contextualisés** (issus de modèles comme BERT) pour représenter les mots-clés en fonction de leur usage dans le contexte. L’idée est de fournir une vue plus fine et pertinente de la sémantique des documents.

---

### **Architecture Système**

KeywordScape se divise en deux grandes parties :

1. **NLP Pipeline :**  
   - **Prétraitement :** Chaque document est converti en une chaîne de texte, nettoyé (suppression des tokens non pertinents) et découpé en paragraphes (correspondant à la largeur contextuelle de BERT).  
   - **Extraction de Mots-Clés :** À l’aide de méthodes telles que TextRank, RAKE ou TF-IDF, un nombre adapté de mots-clés est extrait pour chaque document.  
   - **Encodage Contextuel :** Les paragraphes sont passés à un modèle BERT qui génère pour chaque mot (ou sous-mot) un vecteur d’embedding. Pour chaque mot-clé, si celui-ci est constitué de sous-unités, on calcule la moyenne de leurs embeddings pour obtenir une représentation unique.
   - **Réduction Dimensionnelle :** Les vecteurs de mots-clés sont ensuite réduits à une dimension plottable (2D ou sur une sphère) via UMAP, permettant de conserver les relations sémantiques.

2. **Visualization Pipeline :**  
   - **Création de Cartes Contextualisées :**  
     Le système projette les embeddings réduits sur une carte (ici, une projection géoéquirectangulaire d’une sphère) pour visualiser les mots-clés sous forme d’îles.  
   - **Quantification et Clustering :**  
     Pour éviter le chevauchement et faciliter l’exploration, la surface est quantifiée en cellules et un algorithme de clustering (comme HDBSCAN) est appliqué afin de regrouper les mots-clés ayant des contextes similaires en « îles thématiques ».
   - **Interactions Utilisateur :**  
     L’interface, basée sur D3.js, permet diverses interactions : zoom, filtrage, survol (hovering), et « brushing » pour explorer les régions d’intérêt et obtenir des détails sur les documents associés.

---

### **Apports pour le Projet Artishow**

- **Exploration des Thèmes :**  
  KeywordScape montre comment exploiter des représentations contextuelles pour aller au-delà d’une simple extraction de mots. Dans le cadre d’Artishow, cela peut inspirer une méthode permettant de transformer le vocabulaire brut en concepts intermédiaires, ce qui est au cœur de votre travail de pipeline.

- **Visualisation Interactive :**  
  Le système offre des idées pour créer des visualisations interactives riches (comme la carte d’îles thématiques) qui facilitent l’exploration et la comparaison des contenus. Cela peut servir de base pour la partie interactive de votre projet.

- **Pipeline Complet :**  
  L’approche détaillée du document (NLP Pipeline + Visualization Pipeline) fournit un exemple concret d’intégration d’une transformation de données pour obtenir des concepts à partir d’un corpus. Cela correspond exactement à la recommandation de votre encadrant de visualiser des concepts intermédiaires plutôt que le simple vocabulaire.

---

Le document KeywordScape présente une solution innovante pour la **visualisation de documents** en exploitant la puissance des embeddings contextuels. Pour le projet Artishow, il offre un excellent point de départ pour :
- Intégrer une transformation avancée dans votre pipeline de traitement (passage du niveau mot au niveau concept).  
- Développer des visualisations interactives qui permettent de naviguer dans un espace sémantique riche et informatif.  

En résumé, KeywordScape démontre que la combinaison de techniques NLP modernes (BERT, UMAP, clustering) avec des outils de visualisation interactifs peut grandement enrichir l’exploration et l’interprétation de grands corpus de documents, ce qui est au cœur de vos objectifs pour Artishow.

### A Survey of Visual Analytics Techniques for Machine Learning

Ce document est une revue exhaustive des techniques de visual analytics appliquées au machine learning. Il offre un état de l’art en organisant les méthodes selon les phases clés du pipeline de machine learning :

1. **Avant la construction du modèle :**  
   - **Amélioration de la qualité des données :** Méthodes interactives pour détecter et corriger les anomalies, combler les valeurs manquantes et améliorer les étiquettes.  
   - **Ingénierie des caractéristiques :** Techniques pour sélectionner et construire des caractéristiques pertinentes, souvent de manière interactive.

2. **Pendant la construction du modèle :**  
   - **Compréhension du modèle :** Visualisation des mécanismes internes d’un modèle (par exemple, l’impact des paramètres ou la structure d’un réseau de neurones) pour mieux comprendre comment il produit ses prédictions.  
   - **Diagnostic du modèle :** Outils interactifs permettant de détecter et d’analyser les erreurs survenant durant l’entraînement, ainsi que de comprendre la dynamique du processus d’apprentissage.  
   - **Model Steering :** Méthodes interactives qui intègrent le retour d’expérience des experts pour affiner et améliorer les modèles.

3. **Après la construction du modèle :**  
   - **Analyse des résultats statiques et dynamiques :** Visualisations destinées à interpréter les sorties du modèle, que ce soit sous forme de structures statiques (extraction de thèmes, clusters) ou d’analyses temporelles montrant l’évolution des résultats.

Le document recense 259 travaux issus de conférences de haut niveau et propose une taxonomie détaillée des approches de visual analytics pour le machine learning. Il identifie également plusieurs défis et pistes de recherche future, tels que l’intégration d’outils interactifs pour l’analyse des données en temps réel, l’amélioration de la compréhension des modèles à l’aide de visualisations interactives, ou encore l’analyse de l’évolution des données (concept drift).



**Pour le projet Artishow :**  
Ce document peut servir de référence pour identifier et s’inspirer de techniques avancées de visualisation interactive qui aideront à transformer un simple vocabulaire extrait d’un corpus en concepts plus riches et interprétables. En intégrant ces approches dans votre pipeline, vous pourrez non seulement améliorer la compréhension des données, mais aussi offrir une interface interactive permettant aux utilisateurs d’explorer et d’affiner les résultats. Cela correspond exactement à l’un des objectifs majeurs de votre projet, qui est de passer d’une visualisation brute des mots à une représentation de concepts intermédiaires obtenus à partir des documents.


###############

Pour reconnaître les types de mots: utiliser spacy, voici un exemple de code:
import spacy

'''Télécharger et charger le modèle français (à faire une seule fois)
python -m spacy download fr_core_news_sm'''

nlp = spacy.load("fr_core_news_sm")

texte = "Le renard brun rapide saute par-dessus le chien paresseux."
doc = nlp(texte)

'''Afficher chaque token et son étiquette grammaticale'''
for token in doc:
    print(token.text, token.pos_, token.tag_)

Il a comme output:
Le DET DET__Definite=Def|PronType=Art
renard NOUN NOUN__Gender=Masc|Number=Sing
brun ADJ ADJ__Gender=Masc|Number=Sing|Degree=Pos
rapide ADJ ADJ__Gender=Masc|Number=Sing|Degree=Pos
saute VERB VERB__Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin
par-dessus ADP ADP
le DET DET__Definite=Def|PronType=Art
chien NOUN NOUN__Gender=Masc|Number=Sing
paresseux ADJ ADJ__Gender=Masc|Number=Sing|Degree=Pos
. PUNCT PUNCT
