# Topic modeling 

## Source

https://www.youtube.com/watch?v=IUAHUEy1V0Q&ab_channel=SummerInstituteinComputationalSocialScience

## Concept 

It's a mixed membership model of hard clustering + matrix representation
of a corpus. It only groups words under a set of topic but does not name the topic.
The user has to search for it.
 
- More on iterative baysian learning / gibbs sampler 

## Input

 - Document term matrix ( documents | vocab | meta)

## Parameters

- Number of topics.

It is advised to try a different range of these to 
get closer to the most optimal value. There could be words grouped under a topic 
that don't seem to be linked. This is caused by a surestimation or underestimation of 
the number of topics.


## Output

- Array of words associated to each topic with high probability
-  Assingnment of each document to a topic

## Structural Topic Modeling

- Adding key variables to consider : time of publishment for example

# LDA : Latent Dirichlet allocation

Modèle probabiliste génératif qui suppose que chaque document est une combinaison 
de plusieurs topics, et que chaque topic est une distribution de mots.

## Library 

- Sous Python : Gensim (LDA model) -> Fournit une API simple pour entraînner le modèle avec des 
corpus textuels 

## Input : 
 - Dictionnaire Gensim (mapping entre mots et identifiants) 
 - un corpus (liste de vecteurs de fréquence de mots)

## Important : 

Il est nécessaire d'utiliser des métriques pour mesurer le degré de cohérence des 
résultats : cohérence thématique et autres. En plus d'une analyse qualitative humaine
de la pertinence des résultats quant aux valeurs d'initialisation, le nombre de topics choisis

## Output :

- Extraction des mots-clés de chaque topic

- Affichage les distributions de topics dans les documents

- Visualisation les résultats avec la bibliothèque pyLDAvis, qui permet une exploration interactive

## Principe mathématique sous jacent

 Le modèle repose sur des distributions de Dirichlet pour générer les proportions de topics par document et les proportions de mots par topic. 




###  Paramètres et notations

- `D` : nombre de documents dans le corpus  
- `K` : nombre de topics (fixé à l'avance)  
- `V` : taille du vocabulaire (nombre de mots uniques)  
- `N_d` : nombre de mots dans le document `d`

###  Variables aléatoires

- `theta_d` : vecteur de taille `K` représentant la distribution des topics pour le document `d`  
- `beta_k` : vecteur de taille `V` représentant la distribution des mots pour le topic `k`  
- `z_{d,n}` : topic assigné au mot `n` dans le document `d`, valeur entière entre `1` et `K`  
- `w_{d,n}` : mot observé à la position `n` dans le document `d`, valeur entière entre `1` et `V`

###  Hyperparamètres

- `alpha` : vecteur de taille `K`, paramètre de la distribution de Dirichlet pour `theta_d`  
- `eta` : vecteur de taille `V`, paramètre de la distribution de Dirichlet pour `beta_k`

---

###  Modèle génératif

Pour chaque topic `k` de `1` à `K` :  
- `beta_k` ~ Dirichlet(`eta`)

Pour chaque document `d` de `1` à `D` :  
- `theta_d` ~ Dirichlet(`alpha`)  
- Pour chaque mot `n` de `1` à `N_d` :  
  - `z_{d,n}` ~ Multinomial(1, `theta_d`)  
  - `w_{d,n}` ~ Multinomial(1, `beta_{z_{d,n}}`)

---

###  Relations de probabilité

- P(z_{d,n} = k | theta_d) = theta_{d,k}  
- P(w_{d,n} = v | z_{d,n} = k, beta) = beta_{k,v}  
- P(w_{d,n} = v | theta_d, beta) = sum_{k=1}^{K} theta_{d,k} * beta_{k,v}

---

###  Objectif d'inférence

On observe les mots `w_{d,n}` du corpus.  
On souhaite approximer la distribution postérieure suivante :

P(theta, z, beta | w, alpha, eta)

où :

- `theta = {theta_d}` : proportions de topics dans chaque document  
- `z = {z_{d,n}}` : assignations de topics aux mots  
- `beta = {beta_k}` : proportions de mots dans chaque topic

---

###  Résultats après entraînement

- `theta_d` : profil thématique estimé du document `d`  
- `beta_k` : liste pondérée de mots caractéristiques du topic `k`  
- `z_{d,n}` : topic dominant estimé pour le mot `n` du document `d`
