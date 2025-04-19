**🚗 AML Car Price Prediction**

Autotrader Car Price Prediction Model
This project aims to predict the prices of used cars using a real-world dataset from Autotrader, one of our university's industry partners. With over 402,000 records, the project demonstrates a complete machine learning workflow—from data preprocessing and feature engineering to model training and evaluation.

📊 1. Data Processing for Machine Learning
The dataset contains 12 key attributes about car listings, including:

Mileage

Year of registration

Registration code

Make and model

Vehicle condition

Body type

Fuel type

Standard colour

Public reference

🧹 Handling Missing Values
year_of_registration and reg_code had the most missing values.

New vehicles were imputed with the year 2021, based on UK registration logic.

Corresponding registration codes (e.g., 21, 71) were assigned using domain-specific knowledge.

Used SimpleImputer and KNNImputer via pipeline for robust imputation.

🚫 Outlier Removal & Cleaning
Outliers in mileage handled using IQR.

Inconsistent values (e.g., alphanumeric reg_codes after 2001) were removed.

🧪 Data Splitting
Data was stratified on standard_model and standard_make to ensure balanced representation and split into:

Train: 80%

Validation: 10%

Test: 10%

🛠️ 2. Feature Engineering
Created vehicle age from year of registration.

Categorized mileage into levels for interpretability.

Used 2nd-order polynomial features and interaction terms to capture non-linear relationships.

Final feature matrix contained 77 features.

🧠 3. Feature Selection & Dimensionality Reduction
Removed irrelevant or weakly correlated features (e.g., public_reference, crossover_car_and_van).

Techniques used:

Univariate analysis

Correlation heatmaps

SelectKBest

Recursive Feature Elimination (RFE) – selected 15 best features

Sequential Feature Selector – selected 3 best features

Applied PCA to reduce dimensionality while preserving 98% variance.

🤖 4. Model Building
4.1 Linear Models
Compared models trained on original vs transformed features.

Polynomial + RFE features improved generalization and reduced MAE.

Demonstrated that raw features led to overfitting and underperformance.

📈 Performance Overview
Various models were benchmarked using:

MAE (Mean Absolute Error)

RMSE (Root Mean Square Error)

R² (Coefficient of Determination)

📊 Visualizations and performance comparisons are available in the project notebooks.

📁 Repository Structure
bash
Copy
Edit
.
├── data/                      # Raw and processed datasets
├── notebooks/                # Jupyter notebooks for each ML stage
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_building.ipynb
│   └── ...
├── models/                   # Saved models and outputs
├── visuals/                  # Charts and graphs
├── README.md
└── requirements.txt
✅ Key Takeaways
Thoughtful preprocessing and feature engineering significantly impact model performance.

Combining manual and automated feature selection improves model accuracy and efficiency.

Domain expertise (e.g., UK car registration system) added value to the imputation strategy.

📌 Future Improvements
Experiment with advanced models like XGBoost, LightGBM, and CatBoost.

Use SHAP values for model interpretability.

Deploy the model via a web interface for public testing.

