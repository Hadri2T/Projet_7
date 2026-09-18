# BrainScanAI — Détection de tumeurs cérébrales avec peu d'étiquettes

Projet 7 du parcours Data Scientist OpenClassrooms : *Labellisez et appliquez des approches semi-supervisées en traitement d'images.*

## Le problème

CurelyticsIA, startup e-santé, développe des outils d'IA pour aider les radiologues à analyser des images médicales. Dans le cadre du projet R&D **BrainScanAI**, l'entreprise veut savoir s'il est possible d'automatiser la détection de tumeurs cérébrales sur des IRM.

La difficulté n'est pas le manque d'images, c'est le manque d'**étiquettes**. Faire annoter une image par un radiologue coûte cher et prend du temps : sur 1 500 images, seules 100 ont été labellisées par des partenaires hospitaliers. Entraîner un modèle classique sur 100 exemples mène presque à coup sûr au surapprentissage.

La question du projet est donc : **peut-on exploiter les ~1 400 images non étiquetées pour construire un meilleur modèle qu'avec les 100 étiquettes seules ?** C'est le principe de l'apprentissage semi-supervisé.

## Les données

| Sous-ensemble | Nombre d'images | Étiquettes |
| --- | --- | --- |
| `avec_labels/cancer` | 50 | Tumeur présente (annotation experte) |
| `avec_labels/normal` | 50 | Cerveau sain (annotation experte) |
| `sans_label` | 1 406 | Aucune |

Images JPEG, 512×512 pixels. Le jeu de données n'est pas versionné dans ce dépôt (données médicales).

## La démarche

### 1. Exploration

Vérifier la résolution, les canaux de couleur, la cohérence et la qualité des images. Repérer les éventuels doublons ou images aberrantes avant tout traitement.

### 2. Extraction de caractéristiques par transfer learning

Plutôt que d'apprendre à « voir » à partir de 100 images, on réutilise un réseau convolutif **pré-entraîné sur ImageNet** (ResNet). Ses couches convolutionnelles, gelées, ont déjà appris à détecter des contours, des textures et des formes. On fait passer chaque IRM dans le réseau et on récupère le vecteur de sortie de l'avant-dernière couche : un **embedding** qui résume le contenu visuel de l'image.

### 3. Clustering exploratoire et labellisation « faible »

Sur ces embeddings :

- **Réduction de dimension** (PCA, t-SNE) pour visualiser les données en 2D
- **Clustering** en 2 groupes (K-Means, DBSCAN…) pour voir si les images se séparent naturellement entre cerveaux sains et cerveaux tumoraux
- **Validation avec l'ARI** (*Adjusted Rand Index*) : on compare les clusters obtenus aux 100 étiquettes expertes pour mesurer si le regroupement a un sens médical

Si c'est le cas, chaque image non étiquetée reçoit le label de son cluster. On obtient un **jeu faiblement labellisé** : des étiquettes bon marché mais bruitées. Il reste strictement séparé du **jeu fortement labellisé** (les annotations des radiologues) : les deux ne sont jamais mélangés.

### 4. Apprentissage semi-supervisé

On compare deux stratégies d'entraînement d'un CNN :

| Approche | Entraînement |
| --- | --- |
| **Supervisée** (référence) | Uniquement sur les images labellisées par les experts |
| **Semi-supervisée** | D'abord sur le jeu faiblement labellisé, puis affinage (*fine-tuning*) sur le jeu fortement labellisé |

Les deux modèles sont évalués sur le même jeu de test, composé d'images expertes jamais vues pendant l'entraînement.

## Les métriques

Chaque modèle est évalué avec trois métriques complémentaires :

| Métrique | Ce qu'elle mesure |
| --- | --- |
| **Accuracy** | La proportion de prédictions correctes. Simple à lire, mais elle ne distingue pas les types d'erreurs. |
| **F1-score** | La moyenne harmonique de la précision et du rappel. Elle pénalise un modèle qui rate des cas positifs ou qui multiplie les fausses alertes. |
| **AUC** | L'aire sous la courbe ROC : la capacité du modèle à classer les images par probabilité, indépendamment du seuil de décision (1 = parfait, 0,5 = hasard). |

Les trois sont suivies tout au long du projet. Le contexte médical oriente toutefois l'interprétation : toutes les erreurs n'ont pas le même coût. Un **faux négatif**, c'est-à-dire une tumeur non détectée, est bien plus grave qu'un faux positif, qui entraîne seulement une vérification supplémentaire par un radiologue. L'accuracy ne fait pas cette distinction, alors que le F1-score intègre le rappel. La conclusion sur la métrique la plus pertinente sera tirée à l'issue des expériences.

## Passage à l'échelle

L'étude doit aussi répondre à une question de faisabilité : **peut-on labelliser 4 millions d'images avec un budget de 5 000 € ?** Cela représente environ 0,00125 € par image, soit 160 fois moins que le budget actuel (300 € pour 1 500 images, soit 0,20 € par image). Les résultats du projet serviront à déterminer sous quelles conditions ce passage à l'échelle est réaliste.

## Organisation du dépôt

```
Projet_7/
├── README.md
├── notebooks/
│   ├── 1_features_clustering.ipynb   # preprocessing, embeddings, clustering
│   └── 2_semi_supervise.ipynb        # CNN supervisé vs semi-supervisé
└── presentation/                     # support de soutenance
```

## Outils

Python, PyTorch / torchvision, scikit-learn, NumPy, pandas, matplotlib, seaborn. Entraînement des modèles sur Google Colab (GPU).

## Statut

En cours.
