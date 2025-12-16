# 🏠 California Housing Price Prediction

Machine learning project predicting California house prices using regression models. The repo was created to support Machine Learning training for PowerCoders Switzerland.

Based on a case study from https://github.com/ageron/handson-ml3 

---

## 📋 Project Overview

**Goal:** Predict median house values in California districts.

**Dataset:** California Housing Dataset (1990 census data).

**Methods:** This project employs several regression models including Linear Regression, Decision Tree Regressor, and Random Forest Regressor, along with comprehensive data preprocessing techniques such as imputation, outlier handling, categorical encoding, and feature scaling.

---

## 📁 Project Structure

california-housing-prediction/

├── notebooks/ # Jupyter notebooks

├── dataset/ # Data files

├── plots/ # Visualizations

└── README.md

---

## 🔍 Analysis Steps

- [x] 1. Data Loading & Exploration
    *   The project loads data from a `.tgz` file, inspects its structure using `.head()`, `.info()`, and `.describe()`, and visualizes distributions with histograms.
- [x] 2. Data Cleaning & Preprocessing
    *   This step involves handling missing values using `SimpleImputer` with a median strategy, identifying potential outliers with `IsolationForest`, encoding categorical features using `OneHotEncoder`, and applying feature scaling with `MinMaxScaler` and `StandardScaler`. Log transformation for skewed distributions and discretization of `median_income` are also performed.
- [x] 3. Feature Engineering
    *   New, more predictive features are created, including "rooms_per_house", "bedrooms_ratio", and "people_per_house", to capture meaningful patterns from the raw data.
- [x] 4. Model Training
    *   The data is split into 80% training and 20% testing sets. Three regression models—Linear Regression, Decision Tree Regressor, and Random Forest Regressor—are trained using a preprocessing pipeline.
- [x] 5. Model Evaluation
    *   Models are evaluated using Root Mean Squared Error (RMSE) on both the training and test sets to assess performance and generalization.
- [x] 6. Results & Insights

---

## 📊 Results

The models were evaluated based on their Root Mean Squared Error (RMSE) on the test set:

*   **Linear Regression:**
    *   Training RMSE: 67,884.31
    *   Test RMSE: 68,649.38
    *   The Linear Regression model provided a reasonable baseline with good generalization, as the test RMSE was close to the training RMSE. However, the absolute error was still quite high.
*   **Decision Tree Regressor:**
    *   Training RMSE: 0.0
    *   The Decision Tree perfectly fit the training data, resulting in an RMSE of 0.0, which is a classic sign of overfitting. This model is highly likely to perform poorly on unseen data.
*   **Random Forest Regressor:**
    *   Training RMSE: 18,327.91
    *   Test RMSE: 18,663.56
    *   The Random Forest model demonstrated superior performance. Its test RMSE (18,663.56) was very close to its training RMSE, indicating excellent generalization. This RMSE is significantly lower than that of the Linear Regression model, confirming Random Forest as a powerful model for this dataset.

**Conclusion:** The Random Forest Regressor emerged as the best-performing model, offering a strong balance between predictive accuracy and generalization capabilities.

---

## 🛠️ Technologies Used

-   **Python 3.7+**
-   **Pandas**, **NumPy**: Libraries for data analysis and numerical operations.
-   **Scikit-learn 1.0.1+**: Machine learning library for model training, preprocessing, and evaluation.
-   **Matplotlib**: Library for data visualization.
-   **Google Colab**: Cloud-based environment for development and execution.
-   **`pathlib`**, **`tarfile`**, **`urllib.request`**: For file system operations and data downloading.
-   **`packaging`**: For version parsing.

---

## 👤 Author

**Author's name**

* Author's title*

---

## 📅 Project Timeline

-   Start Date: December 15, 2024
-   Status: In Progress 🚧
