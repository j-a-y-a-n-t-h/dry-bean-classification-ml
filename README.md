# Dry Bean Multiclass Classification

## a. Problem Statement

The objective is to classify a dry bean sample into one of seven bean varieties using 16 numerical geometric and shape-related measurements extracted from bean images. This is a seven-class multiclass classification problem.

## b. Dataset Description

- **Dataset:** UCI Dry Bean Dataset
- **Source:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Instances:** 13,611
- **Input features:** 16
- **Target column:** `Class`
- **Number of classes:** 7
- **Classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA
- **Missing values:** None in the supplied dataset
- **Train-test split:** Stratified 80:20 split
- **Random state:** 42

The 16 input features are numerical. The target column is `Class`. A separate StandardScaler is fitted within each model pipeline using training data only to prevent data leakage.

## c. GitHub Repository Link

**GitHub Repository:**  
https://github.com/j-a-y-a-n-t-h/dry-bean-classification-ml

## d. Models Used

The following five classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM) with RBF kernel

### Evaluation Metrics

The models are evaluated on the same held-out test dataset using:

- Accuracy
- ROC-AUC (Macro One-vs-Rest)
- Precision (Macro)
- Recall (Macro)
- F1-score (Macro)
- Matthews Correlation Coefficient (MCC)

### Experimental Results

| ML Model | Accuracy | AUC | Precision (Macro) | Recall (Macro) | F1 (Macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.8920 | 0.9450 | 0.9075 | 0.9090 | 0.9081 | 0.8696 |
| KNN | 0.9203 | 0.9835 | 0.9354 | 0.9302 | 0.9325 | 0.9037 |
| Gaussian Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine | **0.9229** | **0.9948** | **0.9360** | **0.9340** | **0.9350** | **0.9067** |

### Observations on Model Performance

| Model | Observation |
|---|---|
| Logistic Regression | Achieved 92.14% accuracy and 93.35% macro F1-score. It demonstrated strong classification performance and a high ROC-AUC of 0.9948. |
| Decision Tree | Achieved 89.20% accuracy. Its performance was lower than the other models, indicating that a single decision tree may not capture the classification patterns as effectively. |
| KNN | Achieved 92.03% accuracy and 93.25% macro F1-score. It performed competitively with Logistic Regression, demonstrating that distance-based classification works well on the scaled numerical features. |
| Gaussian Naive Bayes | Achieved 89.79% accuracy. Despite its relatively lower accuracy, it obtained a high ROC-AUC of 0.9916. |
| Support Vector Machine | Achieved the highest accuracy of 92.29%, macro F1-score of 93.50%, and MCC of 0.9067. It also achieved the highest ROC-AUC of 0.9948 (approximately). |
| Overall Winner | Support Vector Machine performed best overall across the reported metrics, with the highest accuracy, macro precision, macro recall, macro F1-score and MCC. |

The Support Vector Machine achieved the strongest overall results, although Logistic Regression and KNN also performed competitively. The Decision Tree and Gaussian Naive Bayes models achieved comparatively lower accuracy.

The macro-averaged precision, recall and F1-score provide a view of performance across all seven classes rather than allowing larger classes to dominate the evaluation. These results indicate strong overall multiclass classification performance.

## e. Streamlit Application

The project includes an interactive Streamlit application for evaluating the saved classification models.

**Live Application:**  
https://dry-bean-classification-ml-mpmge6fwuvqhbtcwqwxkzq.streamlit.app/

### Application Features

- Upload labelled test data in CSV format.
- Select one of the five trained classification models.
- Display Accuracy, AUC, Precision, Recall, F1-score and MCC.
- View the confusion matrix.
- View the classification report.
- Preview uploaded test data.
- Compare model performance using the generated metrics.

The application loads previously trained models from disk. It does not retrain models using uploaded test data.

## f. How to Run Locally

1. Clone the GitHub repository:

   `git clone https://github.com/j-a-y-a-n-t-h/dry-bean-classification-ml.git`

2. Navigate to the project directory.

3. Install the required dependencies:

   `pip install -r requirements.txt`

4. Ensure that `Dry_Bean_Dataset.xlsx` is present in the project directory.

5. Train and evaluate all five models:

   `python train_models.py --data Dry_Bean_Dataset.xlsx`

6. Start the Streamlit application:

   `streamlit run app.py`

7. Upload the generated `test_data.csv` file through the application and select a classifier to view its evaluation results.

## g. Repository Contents

- `train_models.py` – Training, evaluation and model artifact generation.
- `app.py` – Streamlit application for model evaluation.
- `requirements.txt` – Required Python dependencies.
- `Dry_Bean_Dataset.xlsx` – Original dataset.
- `test_data.csv` – Generated held-out test dataset.
- `metrics.csv` – Model comparison results.
- `model/` – Saved trained model pipelines and metadata.
