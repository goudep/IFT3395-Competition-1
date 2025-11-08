# Rapport: Prédiction de la Qualité de la Bière

## 1. Informations de l'Équipe

**Nom de l'équipe Kaggle**: `ift3395_Yudi_Ma_Nomena_Willis` 
**Membre 1**: NOMENA ANDRE Willis 
**Membre 2**: Yudi Ma

## 2. Exploration des donnée
- **Visualisation de nos donnee**: _Boites a moustaches  
                              _Histogramme
                              _Heat map

- **But**: _Trouver les donnees aberrantes et bruitée
       _Voir la distribution des cibles
       _Voir la correlation des caractéristiques et des cibles entre elles 

- **Observation**: Nous observons une distribution peu homogenes des donnee cibles, en effet on remarque qu'il y a un tres peu nombre d'exemple pour les cibles 3,4 et 8,9 et les cibles 1 et 10 sont inexistante.Les caractéristiques qui cont le plus corrélée a la cible sont  alcohol_ABV(0.113552),free_CO2(0.094735,fermentation_strength (0.093718), n’oublions pas qu'elle peuvent avoir de hautes correlation avec d'autre caractéristiques moin lier a la cible.


## 3. Prétraitement des Données (Feature Design)
### 3.1 Phase 1: Prétraitement de base (Modèles de cours)
**Treat_training + Treat_test**:

- Fonctions pour gère le prétraitement:
    _`Treat_training` prend comme argument les donnees brutes applique les traitements ci dessous et sépare le data set en un ensemble de validation et un ensemble d’entraînement et sépare les cibles
    _`Treat_test` prend comme argument l'ensemble de test et applique les meme traitement que l'ensemble d’entraînement

**Encodage des variables catégorielles**: 
- **One-Hot Encoding** de `beer_style` avec `pd.get_dummies(X, columns=["beer_style"], drop_first=True)` pour gérer les catégories absentes dans l'ensemble de test
- Utilisation de `ColumnTransformer` pour combiner le traitement des caractéristiques numériques et catégorielles

**Normalisation des variables numériques**:
- **StandardScaler** pour centrer et réduire les caractéristiques numériques (moyenne=0, écart-type=1)
- Essentiel pour les modèles basés sur les distances (KNN) et les modèles linéaires


**Justification**: Ce prétraitement minimal mais rigoureux établit une baseline solide pour les modèles de cours, en respectant les bonnes pratiques de machine learning.

### 3.2 Phase 2: Améliorations
- Comme vu précédemment dans l'exploration des donnees (2), nous avons peut de representation des classes 3, 4, 8 et 9. Solution : Suréchantillonnage des classes minoritaires,(nous dupliquons aléatoirement les donner appartenant a ses classes pour augmenter leurs occurrence et rendre les models plus sensibles a ceux ci)
- On a aussi observer une grande asymétrie dans la repartissions des valeurs de certaines caractéristiques. Solution:  PowerTransformer (Yeo-Johnson), un transformation qui rapproche la repartissions des donnees d'une gaussienne et réduire l'effet des valeurs biaiser
 
- Nous avons 15 caractéristique certaine peuvent ne pas être representative ou inutile lors de la classification et peuvent toucher aux performance. Solution: Sélection des n meilleures caractéristiques via RandomForest.

---
## 4. Méthodologie
### 4.1 Division des Données
80/20 train/validation avec `random_state=42`. Phase 2: ajout de `stratify=y`.
### 4.2 Phase 1: Modèles vus en cours (Implémentation principale)
Cette phase utilise uniquement les modèles et techniques enseignés en cours:

**1. K-Nearest Neighbors (KNN)**
- **Pipeline**: `ColumnTransformer` (StandardScaler + OneHotEncoder) → `KNeighborsClassifier`
- **Optimisation**: GridSearchCV avec validation croisée (cv=3)
- **Hyperparamètres testés**: `n_neighbors` [15, 21, 25, 31, 35]
- **Justification**: KNN est sensible à la normalisation, d'où l'importance du StandardScaler dans le pipeline

**2. Gaussian Naive Bayes**
- **Approche**: Modèle probabiliste basé sur l'hypothèse d'indépendance conditionnelle
- **Prétraitement**: Utilise les données préprocessées directement (pas de Pipeline intégré)
- **Justification**: Naive Bayes nécessite des données denses, d'où la conversion explicite des matrices creuses

**3. Régression Polynomiale**
- **Pipeline**: Prétraitement → `PolynomialFeatures` (degré 2) → `LinearRegression`
- **Post-traitement**: Arrondi des prédictions continues dans l'intervalle [1, 10]
- **Justification**: Les features polynomiales capturent les interactions non-linéaires entre caractéristiques, cruciales pour ce problème

**Stratégie d'évaluation**: 
- Validation croisée pour KNN (GridSearchCV)
- Évaluation sur validation set pour Naive Bayes et Polynomial Regression
- Classification reports détaillés pour analyser les performances par classe
### 4.3 Phase 2: Modèles avancés

**1. Random Forest (RF)**
- **Pipeline**: Données prétraitées → RandomForestClassifier
- **Optimisation** : GridSearchCV avec validation croisée (cv=3)
- **Hyperparamètres testés**:n_estimators : [100, 200, 300], max_depth : [None, 10, 20, 30],min_samples_split : [2, 5, 10]
- **Justification**: Random Forest combine plusieurs arbres de décision pour réduire la variance et améliorer la robustesse. Le paramètre class_weight="balanced" compense le déséquilibre entre les classes en ajustant les poids selon leur fréquence.

**2. Parzen Window (Hard & Soft)**
- **Approche** : Méthode non-paramétrique basée sur des noyaux de densité (Gaussian).
- **Versions testées** :
    _ Hard Parzen : Classification par vote majoritaire des voisins à distance fixe.
    _Soft Parzen : Pondération des voisins selon une fonction gaussienne.
- **Hyperparamètres principaux** : h (largeur de bande) : [0.1, 0.5, 1.0, 2.0]
- **Justification** : Ces modèles permettent une modélisation flexible sans hypothèse sur la distribution des données. Le réglage précis de la bande h est crucial : trop petite → sur-apprentissage ; trop grande → sous-apprentissage.

**3. Régression Logistique Polynomiale (avec régularisation L2)**
- **Pipeline**: PolynomialFeatures (degré 2 à 4) → StandardScaler → LogisticRegression
- **Optimisation** : Recherche combinée sur :degree : [1, 2, 3, 4], C : [0.01, 0.1, 1, 10] (inverse de la régularisation)
- **Justification :**Ce modèle permet de capturer les interactions non linéaires tout en évitant le sur-apprentissage grâce à la régularisation L2. L’augmentation polynomiale enrichit la représentation des données, particulièrement utile pour des limites de décision complexes.


---
## 5. Résultats
### 5.1 Phase 1: Résultats avec modèles de cours
| Modèle | Précision Validation | Observations |
|--------|---------------------|--------------|
| **Polynomial Regression** | **0.5772** | Meilleur modèle, capture les interactions non-linéaires |
| K-Nearest Neighbors | 0.5421 | Performance modérée, sensible à la normalisation |
| Gaussian Naive Bayes | 0.1566 | Performance faible, probable violation de l'hypothèse d'indépendance |

**Analyse du meilleur modèle (Polynomial Regression)**: 
La régression polynomiale capture les interactions non-linéaires entre caractéristiques. Résultats: précision globale de 57.72%. 

**Points forts**: Implémentation propre avec Pipelines, évaluation rigoureuse, baseline solide de 57.72%.

**Limitations**: Difficulté à prédire les classes minoritaires (3, 4, 8, 9), pas de gestion du déséquilibre, prétraitement basique.
### 5.2 Phase 2: Résultats avec améliorations
| Modèle | Précision Validation | Amélioration vs Phase 1 |
|--------|---------------------|-------------------------|
| **Random Forest** | **0.7957** | **+37.8%** |
| Soft Parzen | 0.7644 | +32.4% |
| Hard Parzen | 0.7390 | +28.0% |
| Polynomial Logistic Regression | 0.7083 | +22.7% |
| KNN amélioré | 0.7003 | +29.2% |
### 4.3 Comparaison Phase 1 vs Phase 2
| Métrique | Phase 1 | Phase 2 | Amélioration |
|----------|---------|---------|--------------|
| Précision | 0.5772 | 0.7957 | **+37.8%** |

**Facteurs d'amélioration**: Prétraitement avancé, gestion du déséquilibre (upsampling + `class_weight`), modèles plus sophistiqués (Random Forest).

---
## 5. Discussion

**Contributions**: Architecture solide avec Pipelines (reproductibilité, évite data leakage), évaluation rigoureuse avec classification reports, baseline de 57.72% démontrant la faisabilité du problème.

**Leçons apprises**: La régression polynomiale surpasse KNN et Naive Bayes, suggérant l'importance des interactions non-linéaires. Le déséquilibre des classes nécessite une attention particulière.

### 5.2 Phase 2
Les améliorations (upsampling, PowerTransformer, Random Forest) produisent un effet synergique, améliorant la précision de 57.72% à 79.57% (+37.8%).
### 5.3 Limitations et améliorations futures

**Limitations**: Phase 1 - modèles basiques, pas de gestion du déséquilibre. Phase 2 - upsampling peut causer surapprentissage, Random Forest coûteux.

**Améliorations futures**: Techniques d'ensemble (Stacking), feature engineering approfondi, SMOTE, optimisation bayésienne.

---
## Conclusion
La **Phase 1** établit une baseline solide (57.72%) avec des modèles de cours, validant la faisabilité du problème. La **Phase 2** améliore significativement les performances (79.57%, +37.8%) grâce à un prétraitement avancé, une meilleure gestion du déséquilibre et des modèles plus sophistiqués. L'évolution de 57.72% à 79.57% illustre l'importance du feature engineering et de la sélection de modèle appropriée.

