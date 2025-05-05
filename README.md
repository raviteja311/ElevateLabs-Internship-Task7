# SVM Binary Classification on Breast Cancer Dataset

This Jupyter Notebook (`Task_7_Support_Vector_Machines_(SVM).ipynb`) demonstrates the implementation and evaluation of Support Vector Machines (SVM) for a binary classification task using the Wisconsin Breast Cancer dataset.

## Overview

The notebook covers the following key steps involved in building and evaluating SVM models:

1.  **Data Loading and Preparation:** Loading the breast cancer dataset and preparing it for classification (encoding the target variable, feature scaling).
2.  **Data Exploration:** Basic exploration including checking for missing values, viewing data info, descriptive statistics, target variable distribution, and feature correlations.
3.  **Model Training:** Training SVM classifiers with both linear and Radial Basis Function (RBF) kernels.
4.  **Hyperparameter Tuning:** Using GridSearchCV with 5-fold cross-validation to find the optimal hyperparameters (`C` for linear and `C`, `gamma` for RBF) for the SVM models based on accuracy.
5.  **Model Evaluation:** Evaluating the performance of the optimized linear and RBF SVM models on the test set using standard classification metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC).
6.  **Decision Boundary Visualization:** Visualizing the decision boundary of the optimized RBF kernel SVM on a 2D subset of the scaled test data ('radius_mean' and 'texture_mean' features).

## Dataset

The notebook uses the `breast-cancer.csv` dataset, which is a common dataset for binary classification tasks.
- **Target Variable:** `diagnosis` (M = Malignant, B = Benign). This is mapped to 1 (Malignant) and 0 (Benign) during preprocessing.
- **Features:** Various numerical features describing characteristics of cell nuclei.
- **Preprocessing:** The 'id' column is dropped, the target variable is encoded, and numerical features are scaled using `StandardScaler`.

## Requirements

The notebook relies on the following Python libraries:

-   pandas
-   numpy
-   matplotlib
-   seaborn
-   scikit-learn (sklearn)

You can install these dependencies using pip:

<pre>pip install pandas numpy matplotlib seaborn scikit-learn</pre>



How to Run
Ensure you have Python and the required libraries installed.
Make sure the breast-cancer.csv file is in the same directory as the notebook.
Open the notebook using Jupyter Notebook or JupyterLab.
Run the cells sequentially from top to bottom.



Results

The notebook outputs the following:
Data exploration summaries and plots.
The best hyperparameters found for both linear and RBF SVMs via GridSearchCV.
Detailed evaluation metrics (Accuracy, Precision, Recall, F1, AUC) for the optimized models on the test set.
A visualization of the RBF SVM's decision boundary on two selected features.
