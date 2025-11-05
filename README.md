# Rapport de Projet 1 : Prédiction de la Qualité de la Bière

**Auteurs**: Yudi Ma et 
**Cours**: IFT-6390 / IFT-3395
**Date**: 5 novembre 2025

---

## 1. Introduction

Ce projet nous demandait de prédire la qualité d'une bière (une note de 1 à 10) en fonction de ses caractéristiques chimiques. C'est un problème classique de classification supervisée. Notre objectif était de construire un modèle qui soit performant non seulement sur nos propres données, mais aussi capable de prédire correctement de nouvelles données sur le test caché de Kaggle.

Conformément aux exigences, notre projet s'est déroulé en deux phases principales :

1.  **Phase 1 (Checkpoint)**: Utiliser les modèles de base vus en classe (comme k-NN, Naive Bayes, Régression Linéaire) pour essayer de battre le score de baseline de `0.29019`.
2.  **Phase 2 (Finale)**: Utiliser des techniques plus avancées (comme la gestion du déséquilibre des données et `Random Forest`) pour obtenir le meilleur score Kaggle possible.

Ce rapport détaille les décisions clés que nous avons prises à chaque étape.

---

## 2. Prétraitement des Caractéristiques (Feature Preprocessing)

Les données brutes (`train.csv`) ne peuvent pas être directement "données à manger" à un modèle d'apprentissage automatique. Nous devions d'abord les "nettoyer" et les "traduire". C'est le rôle du prétraitement. Nous avons utilisé un `ColumnTransformer` pour automatiser ce processus, garantissant ainsi que les données d'entraînement et de test subissent exactement les mêmes transformations.

### 2.1 Nos Étapes de Prétraitement et Justifications

1.  **Traduction des Données Textuelles (One-Hot Encoding)**:
    * **Problème**: La colonne `beer_style` contient du texte (ex: 'Pale', 'Dark'). Les modèles sont des outils mathématiques ; ils ne comprennent pas les mots.
    * **Notre Choix**: `OneHotEncoder`.
    * **Justification**: Cet outil transforme une seule colonne `beer_style` en plusieurs nouvelles colonnes "oui/non" (ex: `beer_style_Pale`, `beer_style_Dark`). Si une bière est 'Pale', sa colonne `beer_style_Pale` aura la valeur 1, et les autres 0. Ainsi, le modèle peut utiliser les chiffres pour comprendre le style.

2.  **Standardisation de l'Échelle Numérique (Feature Scaling)**:
    * **Problème**: Les différentes caractéristiques numériques n'ont pas les mêmes "unités". Le `pH` varie entre 3.0 et 4.0, tandis que le `dissolved_oxygen` peut varier de 100 à 200. Cela peut amener le modèle à croire à tort que le `dissolved_oxygen` est plus important, simplement parce que ses chiffres sont plus grands.
    * **Notre Choix**: `StandardScaler`.
    * **Justification**: `StandardScaler` remet à l'échelle toutes les caractéristiques numériques pour que leur moyenne soit 0 et leur écart-type 1. C'est comme mettre toutes les caractéristiques "sur la même ligne de départ". Le modèle peut maintenant juger équitablement de l'importance réelle de chaque caractéristique.

3.  **Retrait des Caractéristiques Inutiles (Feature Selection)**:
    * **Problème**: La colonne `id` n'est qu'un numéro de ligne, elle n'a aucun lien avec la qualité de la bière.
    * **Notre Choix**: `remainder='drop'`.
    * **Justification**: En ajoutant cette règle à notre `ColumnTransformer`, nous avons automatiquement supprimé la colonne `id`. Cela empêche le modèle d'apprendre à partir de "bruit" (l'ID inutile) et l'oblige à se concentrer sur le "signal" (les caractéristiques chimiques utiles).

---

## 3. Méthodologie

Notre méthodologie s'est concentrée sur **comment tester équitablement nos modèles** et **comment améliorer systématiquement leurs performances**.

### 3.1 Division des Données (Training et Validation)

C'est l'étape la plus importante de notre conception expérimentale.

* **Notre Méthode**: Nous avons utilisé `train_test_split` pour diviser `train.csv` en deux parties : 80% pour "l'entraînement" (`X_train`) et 20% pour "la validation" (`X_val`).
* **Justification (Pourquoi faire cela ?)**:
    * Un modèle aura toujours 100% de bonnes réponses sur les données qu'il a "vues" (l'entraînement), car il peut "apprendre par cœur" les réponses (c'est ce qu'on appelle le **surapprentissage, ou Overfitting**).
    * Nous avons besoin de tester le modèle sur un ensemble qu'il n'a **jamais vu** (le `X_val`) pour connaître sa **vraie performance** de généralisation (son vrai score).
    * Nous avons utilisé `random_state=42` pour garantir que cette division "aléatoire" soit **toujours la même** à chaque exécution du code. Cela rend nos expériences reproductibles et nous permet de comparer équitablement les modèles entre eux.

### 3.2 Phase 1 : Modèles pour le Checkpoint (Notre Code de Baseline)

Durant la première phase, nous avons strictement suivi la consigne de n'utiliser que les modèles vus en classe.

* **Modèles Testés**: K-Nearest Neighbors (k-NN), Gaussian Naive Bayes, et Régression Polynomiale.
* **Justification**: Nous voulions attaquer le problème sous différents angles. k-NN est basé sur la "distance", Naive Bayes sur les "probabilités", et la Régression Polynomiale sur "l'ajustement d'une fonction".
* **Réglage**: Pour k-NN, nous avons utilisé `GridSearchCV` pour tester automatiquement plusieurs valeurs de `k` (ex: `[15, 21, 25]`) au lieu d'une seule au hasard.

### 3.3 Phase 2 : Amélioration Finale (Notre Code Final)

En Phase 1, nous avons découvert que la `PolyRegression` fonctionnait le mieux, ce qui nous a indiqué que les **interactions entre les caractéristiques** (ex: `pH * alcohol_ABV`) étaient importantes. Nous avons aussi découvert un **problème majeur** dans les rapports d'évaluation : le **déséquilibre des données**.

* **Problème**: Il y avait beaucoup d'échantillons pour les qualités 5 et 6, mais presque aucun pour les qualités 3 ou 9. Par conséquent, nos modèles de base, pour maximiser leur score global, ont appris à **ne presque jamais** prédire ces classes rares (leur score de `recall` était de 0.00).
* **Notre Stratégie d'Amélioration**:
    1.  **Résoudre le Déséquilibre (Upsampling)**: Nous avons ajouté `RandomOverSampler` à notre `Pipeline`. Cet outil **copie** les échantillons rares (comme les 3 et 9) jusqu'à ce que l'ensemble d'entraînement soit équilibré. Cela "force" le modèle à apprendre les caractéristiques de ces classes rares.
    2.  **Utiliser un Modèle plus Puissant**: Nous avons choisi `Random Forest`, un modèle d'ensemble puissant qui construit des centaines d'arbres de décision pour "voter" sur la meilleure réponse, ce qui est généralement plus précis qu'un seul modèle.
    3.  **Réglage Approfondi (Deep Tuning)**: Après avoir constaté que `Random Forest` était le meilleur, nous l'avons (dans le code v8) optimisé en profondeur, en utilisant `GridSearchCV` pour tester 27 combinaisons de paramètres différentes (ex: `n_estimators: [200, 300, 400]`, `max_depth: [20, 30, None]`) afin d'en extraire la performance maximale.

---

## 4. Résultats

Nos résultats montrent clairement l'amélioration de performance entre notre "baseline" et notre "version finale".

### Résultats de la Phase 1 : Checkpoint (Code de Baseline)

Nous avons soumis les 3 prédictions CSV à Kaggle, car le score Kaggle est le juge de paix le plus fiable.

**Tableau 1 : Scores des Modèles du Checkpoint**
| Modèle | Accuracy (Validation Locale) | Score Public Kaggle | Dépasse le Baseline (0.29019) ? |
| :--- | :--- | :--- | :--- |
| **Régression Polynomiale (Degré 2)** | **0.5772** | **0.34509** | **Oui (Notre Choix)** |
| k-NN (k=15) | 0.5421| 0.29019 |  Non |
| Gaussian Naive Bayes | 0.1566 | NA | Non |

*Analyse (Phase 1)* : La Régression Polynomiale a facilement battu le baseline. Cela a prouvé que notre prétraitement était efficace et que les interactions de caractéristiques étaient la clé du problème.

### Résultats de la Phase 2 : Modèle Final (Code Final)

Après avoir appliqué le `RandomOverSampler` et le réglage approfondi, nous avons organisé une "finale" entre Random Forest (avec PolyFeatures) et XGBoost. Les résultats de notre ensemble de validation *hold-out* (de 894 échantillons) sont sans appel.

**Tableau 2 : Scores du Modèle Final (Validation Locale)**
| Modèle | Score CV (Accuracy Moyenne) | Score Validation *Hold-out* (Accuracy) |
| :--- | :--- | :--- |
| **Random Forest (avec Caract. Poly.)** | 0.6059 | **0.6555 (Gagnant)** |
| XGBoost | 0.5631 | 0.5973 |

*Analyse (Phase 2)* : Le `Random Forest` combiné aux caractéristiques polynomiales (`rf_poly_combo`) a surclassé `XGBoost` à la fois sur le score de validation croisée (0.61 vs 0.56) et, de manière encore plus significative, sur notre ensemble de validation *hold-out* (0.656 vs 0.597). Le `Random Forest` a donc été sélectionné comme notre modèle final.

**Score Final Kaggle (Random Forest + Poly): `0.37647`**

---

## 5. Discussion

### 5.1 Nos Réalisations

Notre score final de **0.37647** avec le modèle `Random Forest` (combiné avec les caractéristiques polynomiales et le suréchantillonnage) est une amélioration très nette par rapport à notre modèle de baseline (`0.33+`). Cela prouve que notre stratégie d'amélioration (résoudre le déséquilibre + ingénierie de caractéristiques + utiliser un modèle d'ensemble puissant) a été un **succès total**.

Notre modèle final a atteint une précision de **0.6555** sur notre ensemble de validation local.

### 5.2 Limitations et Leçons Apprises (Le plus important)

La plus grande leçon de ce projet vient de ses limitations :

1.  **Le Suréchantillonnage (Upsampling) n'est pas Magique**:
    Même avec notre meilleur modèle (`Random Forest`), le rapport de classification détaillé montre que le `recall` (rappel) pour les classes rares (comme 3, 4, 8, 9) est toujours de **0.00**.
    * **Pourquoi ?** Le problème n'est pas seulement le déséquilibre, c'est que ces classes ont **trop peu** d'échantillons (parfois 1 ou 2). `RandomOverSampler` ne fait que "copier" ces 2 échantillons. Le `Random Forest` est assez "intelligent" pour réaliser que ce ne sont que des copies et qu'il n'y a aucun avantage à les prédire pour améliorer l'accuracy globale. Il **apprend donc à les ignorer**.

2.  **Accuracy Locale vs. Score Kaggle (La plus grande leçon)**
    Nous avons confirmé que notre score local le plus élevé (ex: **0.6555**) n'a **aucune corrélation** avec notre score Kaggle (ex: **0.376**). Cela prouve que Kaggle **n'utilise pas** "l'Accuracy" pour noter. L'Accuracy traite une erreur 6->7 (petite erreur) de la même manière qu'une erreur 3->8 (grosse erreur). Kaggle utilise très probablement une métrique plus avancée (comme le Quadratic Weighted Kappa, QWK) qui **pénalise sévèrement** les prédictions "très fausses". Si nous avions optimisé notre `GridSearchCV` pour la métrique QWK au lieu de l'Accuracy, nous aurions sûrement obtenu un bien meilleur score.

3.  **Messages d'Avertissement (Warnings)**
    Pendant l'entraînement, nous avons reçu des `UserWarning` nous indiquant que certaines classes n'avaient qu'un seul membre, ce qui rendait même une validation croisée à 3 "folds" (`cv=3`) difficile. Cela confirme la gravité du problème de déséquilibre des données et justifie notre choix de ne pas utiliser `SMOTE` (qui aurait échoué).
