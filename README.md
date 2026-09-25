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

Les trois sont suivies tout au long du projet. Le contexte médical oriente toutefois l'interprétation : toutes les erreurs n'ont pas le même coût. Un **faux négatif**, c'est-à-dire une tumeur non détectée, est bien plus grave qu'un faux positif, qui entraîne seulement une vérification supplémentaire par un radiologue. L'accuracy ne fait pas cette distinction, alors que le F1-score intègre le rappel : c'est donc lui qui sert de métrique de décision, l'AUC servant à juger la qualité du classement indépendamment du seuil.

## Résultats

Les quatre modèles sont évalués sur les **mêmes 29 images de test**, annotées par des radiologues et jamais utilisées, ni pour le clustering, ni pour l'entraînement, ni pour le réglage des hyperparamètres.

| Modèle | Entraînement | F1 en validation croisée | F1 sur le test | AUC |
| --- | --- | --- | --- | --- |
| **A — supervisé** | 67 images expertes | 0,765 ± 0,062 | 0,765 | 0,810 |
| **B0 — faible seul** | 1 214 labels issus du clustering | — | 0,800 | 0,905 |
| **B — semi-supervisé** | B0, puis affinage sur les 67 images | 0,925 ± 0,068 | 0,857 | 0,905 |
| **C — semi-supervisé filtré** | Les 70 % de labels faibles les plus sûrs, puis affinage | 0,920 ± 0,025 | 0,929 | 0,952 |

**Les enseignements :**

1. **Le semi-supervisé fonctionne.** Le F1 passe de 0,765 à 0,925 en validation croisée, un écart plus de deux fois supérieur aux écarts-types. Les images sans annotation humaine apportent donc une réelle valeur.
2. **Un modèle sans aucun label humain dépasse déjà le modèle supervisé** (0,800 contre 0,765) : 1 214 étiquettes imparfaites valent mieux que 67 étiquettes parfaites.
3. **Filtrer les labels faibles stabilise l'apprentissage** : l'écart-type est divisé par deux. C'est le principe du seuil de confiance de la pseudo-labellisation.
4. **Les erreurs restantes sont des faux négatifs.** Le meilleur modèle manque 2 tumeurs sur 15 sans produire la moindre fausse alerte. En contexte médical, c'est le compromis à inverser : abaisser le seuil de décision permettrait de récupérer ces tumeurs au prix de quelques vérifications supplémentaires.

**Les limites**, assumées et documentées dans les notebooks :

- Le jeu de test ne compte que **29 images** : une erreur de plus ou de moins déplace l'accuracy de 3,5 points.
- Le clustering de l'étape 3 sépare **d'abord les plans de coupe et les séquences d'IRM**, et seulement ensuite les tumeurs. Comme le jeu annoté présente le même déséquilibre, l'ARI (0,573) en est flatté. Son intervalle de confiance à 95 %, obtenu par bootstrap, va de 0,33 à 0,83.
- L'augmentation de données (rotations, retournements, variations de luminosité) atténue ce biais sans le supprimer.

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

Analyses terminées. Support de présentation en cours.
