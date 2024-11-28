# Credit Card Fraud Detection

This project involves detecting fraudulent transactions using a highly imbalanced dataset. The dataset consists of credit card transactions made by European cardholders over two days in September 2013. It includes a dependent variable (`Class`) indicating whether a transaction is fraudulent (1) or non-fraudulent (0).

---

## Overview of the Dataset
- **Dataset Characteristics:**
  - Highly imbalanced: ~99% of the transactions are non-fraudulent.
  - Most features are in float format, except for the date-time column.
  - Contains anonymized features (likely obtained using PCA) along with `Time` and `Amount`.

- **Objective:** Build and evaluate machine learning models to detect fraudulent transactions.

---

## Workflow

### 1. **Exploratory Data Analysis (EDA)**

#### Univariate Analysis
- **Fraud and Non-Fraud Distribution:**
  - Used a `countplot` to visualize the distribution of fraudulent and non-fraudulent transactions.
- **Transaction Amount and Time Analysis:**
  - `distplot` revealed a bimodal distribution of the `Time` feature, indicating two transaction peaks during the day.
  - Examined the spread of the transaction amounts, observing that fraudulent transactions rarely exceeded $10,000.
- **Feature Distributions:**
  - Plotted histograms of numerical features to check their distributions.
  - Ensured features align closer to a normal distribution for better model performance.

#### Bivariate Analysis
- **Feature Relationships:**
  - Checked relationships between pairs of variables using scatter plots and heatmaps.
  - Identified highly correlated features and ensured they were normally distributed without significant outliers.
- **Class vs. Amount:**
  - Used boxplots to identify outliers in the transaction amount for both fraud and non-fraud cases.
- **Class vs. Time:**
  - Found that fraudulent transactions were independent of the time of the day.

---

### 2. **Data Preprocessing**

- **Scaling Features:**
  - Scaled the `Amount` column using a **Robust Scaler** (based on median and IQR) to handle outliers. 
  - Did not scale `Time` since it was independent of fraud detection.
  
- **Handling Imbalance:**
  - Performed **undersampling** by selecting an equal number of transactions from both fraud and non-fraud classes to create a balanced dataset.
  - Alternatively, considered **k-fold cross-validation** to ensure robust model evaluation.

- **Correlation Analysis:**
  - Plotted a correlation heatmap for both the original and undersampled datasets to identify relationships between features and the target variable.

- **Outlier Removal:**
  - Removed outliers based on the IQR range for highly correlated features, ensuring better model performance.

- **Dimensionality Reduction:**
  - Applied **PCA (Principal Component Analysis)** and **t-SNE** to visualize high-dimensional data in 2D, checking the distinction between fraud and non-fraud classes.

---

### 3. **Model Training**

- **Train-Test Split:**
  - Split the preprocessed dataset into training and testing sets.

- **Models Used:**
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machines (SVM)
  - Decision Tree Classifier
  - Random Forest Classifier

- **Hyperparameter Tuning:**
  - Applied **GridSearchCV** to optimize model hyperparameters for better accuracy.

---

### 4. **Model Evaluation**

- **Confusion Matrix:**
  - Evaluated performance using metrics such as True Positives, False Positives, True Negatives, and False Negatives.
  
- **ROC-AUC Curve:**
  - Plotted the ROC-AUC curve to compare model performance and select the best-performing model.

---

## Conclusion
This project demonstrates the process of handling imbalanced datasets, performing exploratory data analysis, scaling features, reducing dimensions, and training machine learning models for fraud detection. By carefully preprocessing the data and selecting appropriate models, we aim to maximize accuracy and minimize the impact of false positives and false negatives.

---

### Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

### Future Work
- Explore advanced techniques like **SMOTE** for oversampling the minority class.
- Implement deep learning models for fraud detection.
- Test the models on more diverse datasets to validate generalization.

Feel free to contribute by reporting issues or submitting pull requests!