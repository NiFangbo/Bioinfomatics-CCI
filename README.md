# Cell-Cell Interaction Prediction

---

***Bioinfomatics***

## 📌 Project Overview
CCI Boost is a machine learning project for predicting cell-cell interactions (CCI) using single-cell RNA sequencing data. The model combines multiple ensemble methods to classify whether pairs of cells interact based on gene expression features and known ligand-receptor pairs.

## 🧬 Dataset Description

The project uses three main data sources:

1. **Single-cell RNA-seq Data**
   - File: `Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5ad`
   - Contains gene expression profiles for individual cells

2. **Cell-Cell Interaction Pairs**
   - Training edges: `train_edges.csv` (12,000 samples)
   - Validation edges: `val_edges.csv` (4,000 samples) 
   - Test edges: `test_edges.csv` (4,000 samples)
   - Each sample contains: `source` cell, `target` cell, and `label` (1=interaction, 0=no interaction)

3. **Ligand-Receptor Pairs**
   - File: `celltalk_human_lr_pair.txt`
   - Contains 3,398 known ligand-receptor pairs
   - Used to extract biologically relevant features

## 🏗️ Model Architecture

### Feature Engineering
For each cell pair, the following features are extracted for every ligand-receptor pair:
- Ligand expression in source cell
- Receptor expression in target cell
- Product of ligand and receptor expressions
- Mean expression value
- Expression ratio (ligand/receptor)
- Expression difference

**Total features**: 20,370 per sample

### Ensemble Voting Classifier
The model combines three powerful ensemble methods:

1. **Random Forest**
   - 500 estimators
   - Random state: 50

2. **XGBoost**
   - 1000 estimators
   - Learning rate: 0.05
   - Random state: 50

3. **LightGBM**
   - 1000 estimators
   - Learning rate: 0.05
   - Random state: 50
   - Force column-wise splitting enabled

**Voting Strategy**: Soft voting with weights [1.0, 1.1, 1.1]

## 📊 Performance Metrics

On the validation set (4,000 samples):
- **Accuracy**: 95.50%
- **Precision**: 95.51%
- **Recall**: 95.50%
- **F1 Score**: 95.50%
- **Weighted Score**: 95.50%

## 🚀 Usage Instructions

### Prerequisites
```bash
conda create -n cci_boost python=3.12.12
conda activate cci_boost
pip install numpy pandas scanpy scikit-learn lightgbm xgboost seaborn matplotlib tqdm
```

### Running the Project
1. Place all data files in the working directory:
   - `Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5ad`
   - `train_edges.csv`, `val_edges.csv`, `test_edges.csv`
   - `celltalk_human_lr_pair.txt`

2. Execute the notebook cells in order:
   ```python
   # Run all cells in CCI_Boost.ipynb
   ```

3. The model will:
   - Load and preprocess data
   - Extract features from cell pairs
   - Train the ensemble voting classifier
   - Evaluate on validation data
   - Generate predictions for test data
   - Save results to `submission.csv`

### Output Files
- `submission.csv`: Predictions for test data with columns: `source`, `target`, `label`
- Model evaluation plots (confusion matrix, performance metrics)

## 🔧 Key Features

- **Biological Relevance**: Uses known ligand-receptor pairs for feature extraction
- **Ensemble Approach**: Combins strengths of three tree-based algorithms
- **Scalable**: Handles high-dimensional feature space
- **Reproducible**: Fixed random seeds ensure consistent results
- **Comprehensive Evaluation**: Multiple visualization

## ⚙️ Technical Details

### Data Processing
- Gene expression matrices converted to dense format when necessary
- Unique gene identifiers ensured
- Mitochondrial gene filtering implemented
- Quality control metrics calculated

### Model Training
- Training on 12,000 samples
- Validation on 4,000 samples
- Test prediction on 4,000 samples
- Feature extraction optimized with progress tracking

### Computational Requirements
- Memory: ~8GB RAM recommended
- Storage: ~2GB for data files
- Time: Feature extraction ~25 minutes, training <5 minutes

## 📋 Future Improvements

1. **Feature Selection**: Implement dimensionality reduction techniques
2. **Cross-Validation**: Add k-fold cross-validation for robust evaluation
3. **Hyperparameter Tuning**: Optimize individual model parameters
4. **Additional Features**: Incorporate spatial information and pathway data
5. **Deep Learning**: Explore graph neural networks for CCI prediction

## 📚 References

- Scanpy: Single-cell analysis in Python
- Scikit-learn: Machine learning in Python
- XGBoost: Optimized gradient boosting library
- LightGBM: Gradient boosting framework
- CellTalkDB: Ligand-receptor pair database
