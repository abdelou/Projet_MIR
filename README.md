# MIR Project — Content-Based Image Retrieval on Cars

Projet réalisé dans le cadre du cours de Multimedia Information Retrieval (MIR).
L’objectif de cette partie du projet est de développer un moteur de recherche d’images par le contenu sur une base de voitures. Le moteur repose sur l’extraction de descripteurs visuels, le calcul de distances entre vecteurs de caractéristiques, l’affichage des images les plus similaires et l’évaluation par des métriques de recherche d’information.


## 1. Contenu du projet

L’archive contient le code source de la partie 1, les résultats d’évaluation déjà calculés et un dossier de rapport. Les fichiers lourds, notamment la base d’images et les index `.npz`, ne sont pas inclus afin de limiter la taille du rendu.


## 2. Données attendues

La base d’images n’est pas incluse dans le dépôt. Elle doit être placée manuellement dans : `data/cars/`

Les noms des images doivent respecter le format utilisé par le projet, par exemple :

1_4_Kia_stinger_1990.jpg
3_1_Renault_Twingo_4487.jpg
5_0_Mercedes_ClasseCLS_7059.jpg
7_0_Peugeot_508break_9591.jpg
9_0_Audi_A6_12268.jpg

Le premier nombre du nom de fichier correspond à la classe de la voiture. Le projet actuel est configuré pour le groupe 08, avec les classes impaires :

1 : Kia
3 : Renault
5 : Mercedes
7 : Peugeot
9 : Audi

Les 15 images requêtes utilisées pour l’évaluation officielle sont définies dans `src/config.py`.


## 3. Installation

Création d'un environnement virtuel Python et installation des dépendances (listées dans `requirements.txt`):

`py -3.11 -m venv .venv`
`.\.venv\Scripts\Activate.ps1`
`python -m pip install --upgrade pip`
`pip install -r requirements.txt`


## 4. Description des fichiers

`src/config.py` contient les chemins principaux du projet et la configuration du groupe :

* dossier des données ;
* dossier des artefacts (descripteurs et résultats expérimentaux) ;
* identifiant de groupe ;
* liste des 15 requêtes associées à notre groupe

Les requêtes sont stockées sous la forme :

`("R1", 1, "1_4_Kia_stinger_1990")`

où `R1` est l’identifiant de la requête, `1` est la classe pertinente et le dernier champ correspond au début du nom de l’image requête.

`src/datasets.py` contient les fonctions de gestion de la base "Cars", il est utilisé par les scripts d’indexation et d’évaluation.

`src/descriptors/classical.py` implémente les descripteurs classiques vus dans les TPs (BGR, HSV, HOG, SIFT, ORBB, Hu, LBP, GLCM):

Les descripteurs SIFT et ORB sont utilisés sous forme agrégée : les descripteurs locaux sont résumés en un vecteur global de taille fixe. Cette représentation est différente d’un matching local complet BF/FLANN, mais elle permet de stocker tous les descripteurs dans des matrices homogènes et de les fusionner avec les autres méthodes.

`src/descriptors/cnn.py` implémente le descripteur profond `ResNet50` et `src/descriptors/vit.py` implémente le descripteur profond `ViT-B/16`.

`src/search.py` contient les distances et fonctions de ranking utilisées dans le moteur :

`src/metrics.py` contient les fonctions de métriques de recherche :
Precision@K; Recall@K; Average Precision@K; mAP; R-Precision
Ces métriques sont utilisées pour évaluer les résultats sur les 15 requêtes du groupe.

`src/visualization.py` génère des grilles d’images pour visualiser les résultats Top-20 et Top-50.
Les images pertinentes et non pertinentes peuvent ainsi être inspectées visuellement pour analyser qualitativement les cas réussis et les cas d’échec.

`src/part1_index_cars.py` extrait les descripteurs de toutes les images de la base et sauvegarde les résultats dans des fichiers `.npz`.

Chaque fichier `.npz` contient :

* features      : matrice des vecteurs de caractéristiques
* labels        : labels/classes des images
* image_paths   : chemins des images
* descriptor    : nom du descripteur utilisé
* indexing_time : temps d’indexation


`src/part1_evaluate_cars.py` charge un index `.npz`, exécute les 15 requêtes affiliées à notre groupe, calcule les métriques et sauvegarde :

* un CSV détaillé par requête ;
* un résumé JSON global ;
* des figures Top-20 et Top-50 si un dossier de visualisation est fourni.

La distance peut être choisie manuellement ou automatiquement avec `--metric auto`.
Le mode `auto` assigne les métriques suivantes de la sorte :

* BGR / HSV / LBP                   → chi_square
* HOG / SIFT / ORB / Moments / GLCM  → euclidean
* ResNet50 / ViT-B/16               → cosine

`src/multi_retrieval_engine.py` est le moteur de recherche utilisé par l’interface Flask.
Il permet de charger plusieurs index simultanément et de faire une fusion tardive des distances :

1. extraction de la requête avec chaque descripteur sélectionné ;
2. calcul des distances vers toutes les images de la base ;
3. normalisation des distances entre 0 et 1 ;
4. moyenne des distances normalisées ;
5. tri croissant pour obtenir le classement final.

Le moteur vérifie également que les index chargés sont alignés, c’est-à-dire qu’ils contiennent les mêmes images dans le même ordre.

`src/app.py` est l'application Flask du moteur de recherche.

Elle permet de :

* charger une image requête ;
* sélectionner un ou plusieurs descripteurs ;
* choisir une mesure de similarité;
* choisir l'affichage du Top-20 ou Top-50 ;
* lancer la recherche ;
* afficher les résultats ;
* afficher la courbe Recall/Precision et les métriques de la requête.

`src/templates/index.html` représente le template HTML de l’interface Flask.

`results/` contient les résultats d’évaluation déjà calculés pour chaque descripteur.

Pour chaque descripteur, deux fichiers sont fournis : `cars_<descriptor>_auto.csv` et `cars_<descriptor>_summary.json`

Le CSV contient les résultats détaillés pour les 15 requêtes et le JSON contient un résumé global.


## 5. Reconstruction des index `.npz`

Les fichiers `.npz` ne sont pas inclus dans le dépôt final car ils peuvent être volumineux. Ils doivent être reconstruits localement.

Indexer tous les descripteurs : `python -m src.part1_index_cars --data-dir data/cars --descriptor <descriptor_utilisé> --out artifacts/descriptors/cars_<descriptor_utilisé>.npz`


## 6. Évaluation des descripteurs

Évaluer tous les descripteurs avec la distance automatique :

`python -m src.part1_evaluate_cars --index artifacts/descriptors/cars_<descriptor_utilisé>.npz --metric <métrique_utilisée> --results-csv artifacts/results/cars_<descriptor_utilisé>_<métrique_utilisée>.csv --summary-json artifacts/results/cars_<descriptor_utilisé>_summary.json --visual-dir artifacts/figures`

Les résultats déjà fournis dans le dossier `results/` correspondent à cette évaluation.


## 7. Lancement de l’interface graphique

Avant de lancer l’interface, les fichiers `.npz` doivent exister dans : `artifacts/descriptors/`

Lancer Flask : `python -m src.app`

Ouvrir ensuite : http://127.0.0.1:5000

Utilisation de l’interface :

1. cliquer sur `Load` pour choisir une image requête ;
2. sélectionner un ou plusieurs descripteurs ;
3. cliquer sur `Load descriptors` ;
4. choisir la distance, ou laisser `Auto` ;
5. choisir `Top-20` ou `Top-50` ;
6. cliquer sur `Search (Image)` ;
7. cliquer sur `Compute R/P Curve` pour afficher la courbe et les métriques.


## 8. Descripteurs disponibles dans l’interface

Les descripteurs recommandés pour l’analyse principale sont : ViT-B/16; ResNet50 et HSV.
Ce choix permet de comparer :

* un Vision Transformer ;
* un CNN profond ;
* un descripteur classique couleur.


## 9. Distances disponibles dans l’interface

Les distances disponibles sont toutes celles ayant été implémentées, le mode `Auto` est recommandé pour les tests principaux, car il choisit une distance adaptée à chaque descripteur.


## 10. Remarques méthodologiques

La pertinence est définie au niveau de la marque automobile. Cette définition produit des classes visuellement hétérogènes : deux véhicules d’une même marque peuvent être très différents, tandis que deux voitures de marques différentes peuvent être visuellement proches.

Les valeurs absolues de mAP restent donc modérées. L’objectif principal est de comparer les descripteurs entre eux et d’évaluer leur capacité à concentrer des images pertinentes dans les premiers rangs du classement.

SIFT et ORB sont utilisés sous forme agrégée, et non comme matching local complet. Ce choix permet une indexation homogène et une fusion simple avec les autres descripteurs.
