# BrainScanAI — Labellisation semi-supervisée d'IRM cérébrales

Projet 7 du parcours Data Scientist OpenClassrooms : *Labellisez et appliquez des approches semi-supervisées en traitement d'images.*

## Contexte

CurelyticsIA, startup e-santé, veut automatiser la détection de tumeurs cérébrales sur IRM. La majorité des images n'est pas étiquetée ; seul un petit sous-ensemble a été annoté par des radiologues (normal / cancéreux).

## Démarche

1. **Exploration** du jeu de données (résolution, canaux, structure)
2. **Extraction de features** avec un CNN pré-entraîné (ResNet), couches convolutionnelles gelées
3. **Clustering exploratoire** (PCA / t-SNE, K-Means, DBSCAN), évalué par ARI sur les images labellisées → labellisation « faible »
4. **Approche semi-supervisée** : entraînement d'un CNN sur les labels faibles puis sur les labels forts, comparé à un entraînement supervisé seul

## Livrables

- Notebook 1 : preprocessing, extraction des features, analyse non supervisée, clustering
- Notebook 2 : approche semi-supervisée
- Support de présentation (recommandations pour un passage à l'échelle : 4 millions d'images, budget de 5 000 €)

## Données

Le jeu de données n'est pas versionné (données médicales).
