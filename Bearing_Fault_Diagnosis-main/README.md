# Bearing Fault Diagnosis

A machine learning project for identifying bearing faults using statistical features extracted from bearing signals. The project includes data ingestion, preprocessing, model training, evaluation, and model artifact storage.

## Project Overview

Bearings are essential components in industrial machinery. Faults in bearings can lead to equipment failure, downtime, and increased maintenance costs.

This project uses machine learning techniques to analyze bearing-related features and predict the corresponding fault category.

### Objectives

- Process and prepare bearing-condition data for machine learning.
- Handle missing values and standardize numerical features.
- Apply Principal Component Analysis (PCA) for dimensionality reduction.
- Train and compare multiple machine learning models.
- Evaluate model performance and save trained artifacts.

## Dataset

The project uses `notebook/data/raw.csv`, containing **2,300 rows and 10 columns**.

| Feature | Description |
|---|---|
| `max` | Maximum signal value |
| `min` | Minimum signal value |
| `mean` | Mean signal value |
| `sd` | Standard deviation |
| `rms` | Root mean square |
| `skewness` | Measure of signal asymmetry |
| `kurtosis` | Measure of signal-tailedness |
| `crest` | Crest factor |
| `form` | Form factor |
| `fault` | Target bearing-condition label |

The dataset contains nine numerical features and ten fault categories, including normal bearing operation, ball faults, inner-race faults, and outer-race faults.

## Project Structure

```text
Bearing_Fault_Diagnosis/
├── .github/
│   └── workflows/
├── .dvc/
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── label_encoder.pkl
│   └── model_scores.png
├── catboost_info/
├── notebook/
│   └── data/
│       └── raw.csv
├── src/
│   └── Bearing_Fault_Diagnosis/
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       ├── pipelines/
│       │   ├── prediction_pipeline.py
│       │   └── training_pipeline.py
│       ├── exception.py
│       ├── logger.py
│       └── utils.py
├── app.py
├── main.py
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

## Methodology

### 1. Data Ingestion

The raw dataset is loaded from a CSV file. The ingestion component prepares the data and splits it into training and testing datasets.

- Training set: 80%
- Testing set: 20%
- Random state: 42

### 2. Data Preprocessing

The preprocessing pipeline includes:

- **Missing-value imputation:** Replaces missing numerical values using the mean.
- **Feature scaling:** Uses StandardScaler to standardize numerical features.
- **Dimensionality reduction:** Uses PCA to retain 95% of the variance in the transformed features.
- **Label encoding:** Converts categorical fault labels into numerical representations.

The preprocessing steps are fitted on the training data and then applied to the test data.

### 3. Model Training

The project compares the following machine learning algorithms:

- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Linear Regression
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

GridSearchCV is used for hyperparameter tuning with three-fold cross-validation.

### 4. Model Evaluation

The current implementation evaluates models using the coefficient of determination (R² score) and compares training and testing performance.

A model comparison plot is generated to visualize the results.

**Note:** Since bearing fault labels represent categories, classification models and metrics such as accuracy, precision, recall, F1-score, and a confusion matrix would be more appropriate for the final fault-diagnosis task.

### 5. Model Artifact Storage

The project saves the following artifacts:

- `model.pkl`: Selected trained model.
- `preprocessor.pkl`: Fitted preprocessing pipeline.
- `label_encoder.pkl`: Label encoder for fault categories.
- `model_scores.png`: Model performance comparison plot.

These artifacts can be reused for prediction once a working inference pipeline is implemented.

## Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, CatBoost
- **Dimensionality Reduction:** PCA
- **Model Selection:** GridSearchCV
- **Visualization:** Matplotlib and Seaborn, where used
- **Data Versioning:** DVC
- **Containerization:** Docker configuration included
- **Version Control:** Git and GitHub

See `requirements.txt` for the project's dependencies.

