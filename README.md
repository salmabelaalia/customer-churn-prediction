# 🧠 CCP-Net — Prédiction de l'Attrition Client par Réseaux de Neurones Hybrides
[![Python](https://img.shields.io/badge/Python-3.13%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](PyTorch)
[![status](https://img.shields.io/badge/Status-Master%2520Project-rebeccapurple.svg)](status)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

#### Projet de Master 2 - Sciences des Données et Analytique
Reproduction et évaluation approfondie du modèle hybride CCP-Net pour la prédiction de l'attrition client.

#### Article de référence : Liu et al. (2024) — Customer churn prediction model based on hybrid neural networks
DOI : 10.1038/s41598-024-79603-9

## 📌 Présentation du Projet
Ce projet de Master 2 vise à reproduire, implémenter et évaluer le modèle de Deep Learning CCP-Net (Customer Churn Prediction Network), une architecture hybride innovante pour la prédiction de l'attrition client.

## 🎯 Objectifs du projet

- Reproduire l'architecture CCP-Net proposée par Liu et al. (2024)
- Évaluer les performances sur 4 datasets de secteurs différents
- Analyser l'impact de chaque composant via une étude d'ablation
- Comparer les résultats avec les modèles de référence

## 🏗️ Architecture du Modèle

CCP-Net combine trois architectures de Deep Learning complémentaires :
Modules clés :
- **Multi-Head Self-Attention**: Capture les dépendances globales dont il comprend les relations complexes entre variables.
- **BiLSTM**: Modélise les dépendances temporelles dont il analyse le passé et le futur simultanément.
- **CNN**: Extrait les caractéristiques locales	Identifie les patterns importants à court terme

## 📊 Résultats Obtenus
### Performances du modèle CCP-Net
| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
|📱 Telecom | 80.54% | 62.21% | 86.74% | 71.94% | 
|🏦 Bank | 78.44% | 48.94% | 72.12% | 57.88% |
|🛡️ Insurance | 87.31% | 47.34%	66.22% | 55.02% |
|📰 News | 68.83% | 46.18% | 44.16% | 37.13% |

### 📌 Points clés :

✅ Meilleur Recall : Telecom (86.74%) — excellent pour détecter les clients à risque

✅ Meilleure Accuracy : Insurance (87.31%) — modèle très précis

⚠️ Performance plus faible : News (F1 = 37.13%) — données plus complexes

🎯 Écart type faible : Indique une bonne stabilité du modèle

### 🔬 Étude d'Ablation

L'étude d'ablation sur le dataset Telecom révèle la contribution de chaque module :

| Configuration | Accuracy | F1-Score |
|---------------|----------|----------|
| Attention seul|84.77%	|72.77% |
| BiLSTM seul	|85.43%	|74.00% |
| CNN seul	|84.43% |71.31% |
| Attention + BiLSTM	|85.40%	|73.69% |
| Attention + CNN	|84.40%|71.78% |
| BiLSTM + CNN	|85.93%	|73.45% |
| CCP-Net complet |★	85.20% |72.18% |

#### 🔑 Conclusions de l'ablation :

- Le BiLSTM est le module le plus performant individuellement (F1: 74.00%)
- La combinaison BiLSTM + CNN donne la meilleure Accuracy (85.93%)
- L'Attention améliore la performance quand elle est combinée
- Le modèle complet offre le meilleur compromis global

## 📁 Structure du Projet
```
Projet_CCP-Net/
│
├── data/                          # Les 4 datasets originaux
│   ├── Bank.csv
│   ├── Insurance.csv
│   ├── News.xlsx
│   └── Telecom.csv
│
├── images/                        # Figures générées
│   ├── adasyn_distribution.png    # Distribution après ADASYN
│   ├── confusion_matrices.png     # Matrices de confusion
│   ├── learning_curves.png        # Courbes d'apprentissage
│   ├── ablation_study.png         # Résultats ablation
│   └── heatmap_*.png              # Heatmaps de corrélation
│
├── models/                        # Modèles entraînés (.pth)
│   ├── best_model_bank.pth
│   ├── best_model_insurance.pth
│   ├── best_model_news.pth
│   └── best_model_telecom.pth
│
├── implementation.ipynb           # Code principal (PyTorch)
├── README.md                      # Ce fichier
├── rapport_projet_ML.pdf          # Rapport final détaillé
└── presentation_ML.pptx           # Présentation soutenance
```

## 🚀 Installation et Exécution
#### Prérequis
```bash
Python >= 3.10
PyTorch >= 2.0.0
Installation des dépendances
bash
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn
pip install scikit-learn imbalanced-learn
pip install jupyter notebook
```

#### Lancement
```bash
jupyter notebook implementation.ipynb
```
## 📊 Visualisations Générées

1. Distribution des classes après ADASYN
images/adasyn_distribution.png
Équilibrage des classes pour chaque dataset

2. Matrices de confusion
https://images/confusion_matrices.png
Performance détaillée par dataset

3. Courbes d'apprentissage
https://images/learning_curves.png
Convergence du modèle sur les 4 datasets

4. Étude d'ablation
https://images/ablation_study.png
Contribution de chaque module

## 🔬 Méthodologie

- Prétraitement des données
- Nettoyage : Suppression des doublons, gestion des valeurs manquantes
- Encodage : LabelEncoder + OneHotEncoder selon le type de variable
- Sélection de features : Corrélation > seuil (0.01 - 0.05)
- Normalisation : StandardScaler (mean=0, std=1)
- Équilibrage : ADASYN pour gérer le déséquilibre des classes

Hyperparamètres
Paramètre	Valeur
Optimizer	Adam
Learning Rate	0.001
Batch Size	128
Epochs max	30
Early Stopping	6 epochs
Cross-Validation	5-Fold

## 📈 Comparaison avec l'Article

Dataset	Nos résultats (F1)	Article (F1)	Écart
Telecom	71.94%	91.18%	-19.24%
Bank	57.88%	90.19%	-32.31%
Insurance	55.02%	95.43%	-40.41%
News	37.13%	94.22%	-57.09%
Note : L'écart s'explique par :

- L'absence de GPU (entraînement sur CPU)
- Des hyperparamètres optimisés différemment
- Un nombre d'epochs réduit
- Un sous-échantillonnage des données

## 📚 Références

- Liu, X., Xia, G., Zhang, X., Ma, W., & Yu, C. (2024). Customer churn prediction model based on hybrid neural networks. Scientific Reports, 14, 30707.
- Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.
- He, H. et al. (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. IJCNN.

## 📄 Licence
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.
