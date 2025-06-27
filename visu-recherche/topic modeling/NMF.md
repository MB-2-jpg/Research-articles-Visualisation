# Non-negative Matrix Factorization (NMF) - Documentation  

## Présentation du principe

### Avec des mots

Le NMF (Non-negative Matrix Factorization) est un groupe d'algorithme d'algèbre linéaire où on factorise une Matrice V en deux matrices W et H. on parle de Non-negative Matrix car les trois matrices n'ont aucun élément négative. Cela rend plus facile l'extraction et inspection de caracteristiques dont le topic modeling.

### Avec des Maths

$$ m,n,k \in N , V \in M_{m,n}, W \in M_{m,k} , H \in M_{k,n} $$

$$ V \approx W \times H$$
On remarque que ce n'est pas forcément une égalité strict mais surtout une approximation de V.

### Application sur une extraction des sujets dans un corpus de documents

Soit la matrice d'entrée (la matrice à factoriser) V avec 10000 lignes et 1000 colonnes où les mots sont dans les lignes et les documents dans les colonnes. En d'autres termes, nous disposons de 1000 documents indexés par 10000 mots. Il s'ensuit qu'un vecteur colonne v dans V représente un document.
Supposons que nous demandions à l'algorithme de trouver 10 caractéristiques afin de générer une matrice de caractéristiques W avec 10000 lignes et 10 colonnes et une matrice de coefficients H avec 10 lignes et 1000 colonnes.
Le produit de W et H est une matrice de 10000 lignes et 1000 colonnes, de la même forme que la matrice d'entrée V et, si la factorisation a fonctionné, il s'agit d'une approximation raisonnable de la matrice d'entrée V.
Le traitement de la multiplication matricielle ci-dessus montre que chaque colonne de la matrice produit WH est une combinaison linéaire des 10 vecteurs colonnes de la matrice caractéristiques W avec des coefficients fournis par la matrice coefficients H.



## Algorithme de Factorisation

Le proincimpe d'un tel algroithmle est donc de minimiser l'écart entre l'approximation par le produit de W et H de la matrice d'entré V :
$$min_{W,H} \left\lVert V- WH \right\rVert_{F}^{2}$$

La norme utilisé pour cette approximation est la norme de Frobenius.

### La norme de Frobenius

- Définition :

    $$\left\lVert V \right\rVert_{F} = \sqrt{\sum_{i,j}V_{i,j}^{2}} $$

Pourquoi utiliser cette norme ?

- Elle quantifie l’écart global entre la matrice originale et sa factorisation.

- Elle permet d’assurer une convergence plus stable de l’algorithme.

- Elle est compatible avec les algorithmes d’optimisation couramment utilisés pour NMF.