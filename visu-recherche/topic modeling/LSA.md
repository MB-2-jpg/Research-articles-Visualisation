LSA (Latent Semantic Analysis) est une technique de topic modeling qui vise à découvrir les relations cachées entre les mots et les documents d'un corpus.

-Etapes:

1. Construction de la matrice document-terme :
On représente le corpus (l'ensemble des documents) sous la forme d'une matrice où chaque ligne correspond à un document et chaque colonne à un terme (mot). Les cellules de la matrice contiennent des valeurs indiquant l'importance d'un mot dans un document (souvent pondérées par TF-IDF (voir https://www.youtube.com/watch?v=C3V2Lf1Y9Qk&ab_channel=KNIMETV)).*

2. Application de la décomposition en valeurs singulières (SVD) :
La matrice document-terme est décomposée en trois matrices :

U (matrice des vecteurs document),

Σ (matrice diagonale des valeurs singulières, qui sont des nombres réels non négatifs, généralement ordonnés de la plus grande à la plus petite et qui indiquent l'importance de chaque dimension latente.),

Vᵀ (matrice des vecteurs terme).
Cette décomposition permet de réduire la dimensionnalité du problème en conservant les dimensions les plus significatives qui capturent la structure latente du corpus.

3. Réduction de dimensionnalité :
En gardant uniquement les plus grandes valeurs singulières (et leurs vecteurs associés), on élimine le bruit et on obtient une représentation condensée des documents et des termes dans un espace latent de moindre dimension. Cela permet de mettre en évidence les associations entre les mots et de regrouper les documents en fonction des thèmes sous-jacents.

4. Interprétation des dimensions latentes :
Les dimensions résultantes ne correspondent pas directement à des mots, mais à des concepts ou thèmes latents. Par exemple, des mots qui apparaissent souvent ensemble (comme "chat", "miauler", "félin") se rapprocheront dans cet espace, révélant une dimension qui pourrait être interprétée comme liée à la thématique "animaux domestiques".

On peut appliquer la LSA en utilisant Python et des outils de scikit-learn (TruncatedSVD et TfidfVectorizer)

Explication en vidéo avec des exemples d'application:
https://www.youtube.com/watch?v=hB51kkus-Rc&list=PLroeQp1c-t3qwyrsq66tBxfR6iX6kSslt&ab_channel=DatabricksAcademy

- Input de la LSA :
Un corpus de documents (pour notre cas c'est un fichier texte avec les documents séparés par un séparateur arbitraire) converti en matrice document-terme (souvent via TF-IDF).

- Output de la LSA :

Les composantes latentes (la projection des documents dans un espace de dimension réduite).

Pour chaque composante, une liste de mots-clés indiquant le thème latent.

- Avantages :
Simplicité et Rapidité :
La LSA est basée sur l’algèbre linéaire et se calcule généralement rapidement même sur des corpus de taille modérée.

Réduction de dimension :
Elle permet de réduire la dimensionnalité et de capturer les associations latentes entre mots et documents.

Pas de paramètres d'initialisation aléatoires :
Contrairement aux modèles probabilistes (ou aux méthodes comme LDA), la LSA ne nécessite pas de paramétrage complexe lié à l'initialisation.

- Inconvénients :
Pas de modèle génératif :
Comme mentionné, la LSA n'est pas un modèle probabiliste. Cela signifie qu'elle ne permet pas de calculer naturellement des métriques comme la perplexity.

Moins intuitive pour la modélisation thématique :
Les composants latents sont des combinaisons linéaires de termes et peuvent être plus difficiles à interpréter que les distributions de probabilité fournies par LDA.

Sensibilité aux informations non pertinentes :
Sans un prétraitement rigoureux, la LSA peut être influencée par le bruit dans les données (mots rares ou termes trop fréquents non filtrés).

Score de cohérence: 0.4 à 0.6 (pour les 17 premières thèses)