# Dry Bean Multiclass Classification

## a. Problem statement
The objective is to classify a dry bean sample into one of seven bean varieties using 16 numeric geometric and shape-related measurements extracted from bean images. The task is a seven-class classification problem.

## b. Dataset description
- Dataset: UCI Dry Bean Dataset
- Source: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- Instances: 13,611
- Input features: 16
- Target: `Class`
- Classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA
- Missing values: none in the supplied dataset
- Split: stratified 80:20 train/test split, random state 42

The 16 input columns are numeric. The target column is `Class`. The model training script fits a separate StandardScaler inside each model pipeline using training data only.

## c. GitHub Repository Link
Replace this line with your own public GitHub repository URL:
`[PASTE YOUR GITHUB REPOSITORY URL HERE]`

## d. Models used

The following five classifiers are implemented:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel)

### Evaluation metrics
The training script calculates Accuracy, macro one-vs-rest ROC-AUC, macro Precision, macro Recall, macro F1 and Matthews Correlation Coefficient (MCC) on the same held-out test set.

After running `train_models.py`, copy the actual values from `metrics.csv` into this table. Do not use estimated or published values as your experiment results.

| ML Model | Accuracy | AUC | Precision (macro) | Recall (macro) | F1 (macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Fill from metrics.csv | Fill | Fill | Fill | Fill | Fill |
| Decision Tree | Fill from metrics.csv | Fill | Fill | Fill | Fill | Fill |
| KNN | Fill from metrics.csv | Fill | Fill | Fill | Fill | Fill |
| Gaussian Naive Bayes | Fill from metrics.csv | Fill | Fill | Fill | Fill | Fill |
| Support Vector Machine | Fill from metrics.csv | Fill | Fill | Fill | Fill | Fill |

### Observations on model performance
Write observations after running the experiment. Compare the models using the measured metrics, including whether a model performs consistently across the seven classes. Explain any trade-offs between accuracy and macro-averaged metrics.

| Model | Observation |
|---|---|
| Logistic Regression | Complete this using your actual test results. |
| Decision Tree | Complete this using your actual test results. |
| KNN | Complete this using your actual test results. |
| Gaussian Naive Bayes | Complete this using your actual test results. |
| Support Vector Machine | Complete this using your actual test results. |
| Overall winner | Identify the model using the metrics and explain why. |

## Streamlit application
The application supports:
- CSV upload for labelled test data
- Model selection dropdown
- Display of all six evaluation metrics
- Confusion matrix
- Classification report
- Test data preview

The app loads saved models and does not retrain on uploaded test data.

## How to run locally

1. Install dependencies:
   `pip install -r requirements.txt`
2. Place `Dry_Bean_Dataset.xlsx` in the project folder.
3. Train and evaluate:
   `python train_models.py --data Dry_Bean_Dataset.xlsx`
4. Start the app:
   `streamlit run app.py`
5. Upload the generated `test_data.csv` in the app.

## Repository contents
- `train_models.py`: training, evaluation and artifact generation
- `app.py`: Streamlit evaluation interface
- `requirements.txt`: Python dependencies
- `test_data.csv`: generated held-out test set (created by training script)
- `metrics.csv`: actual model comparison results (created by training script)
- `model/`: saved trained model pipelines and metadata (created by training script)

## Academic and reproducibility note
Run the experiment yourself in BITS Virtual Lab. Review and understand the implementation, record your own outputs, customize the interface and observations, and include the required screenshot of execution in BITS Virtual Lab.
