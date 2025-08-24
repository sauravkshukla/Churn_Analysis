# Churn_Analysis
End_to_End_Churn_Analysis_Power_Bi Dashboard

# Churn Analysis ML Project

## Overview
This repository contains a machine learning project focused on customer churn prediction and analysis. The project utilizes multiple Jupyter notebooks for data analysis, model training, and evaluation, along with a Power BI report for visualization.

## Project Structure
```
Churn Analysis ML Files/
├── Churn Analysis Report.pbix          # Power BI report for visualization
├── Churn Prediction.ipynb              # Main churn prediction notebook
├── Churn Prediction 2.ipynb            # Additional churn prediction notebook
└── Churn Prediction 3.ipynb            # Extended churn prediction notebook
```

## Features

### Data Analysis and Visualization
- Comprehensive data exploration and preprocessing
- Feature engineering and selection
- Advanced visualization using Matplotlib and Seaborn
- Power BI integration for interactive reporting

### Machine Learning Models
- Random Forest classifier implementation
- Model evaluation metrics including:
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - F1-Score
  - ROC AUC
  - Average Precision
  - Log Loss

### Model Evaluation
- Calibration plots
- Confusion matrices
- Feature importance analysis
- Prediction distribution analysis
- Tree depth visualization

## Technical Requirements

### Python Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- openpyxl
- mlflow
- Other MLFlow dependencies

### Additional Tools
- Jupyter Notebook
- Power BI Desktop

## Getting Started

1. Install the required Python packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl mlflow
```

2. Open the Jupyter notebooks in sequence:
   - Start with `Churn Prediction.ipynb`
   - Follow with `Churn Prediction 2.ipynb`
   - Continue with `Churn Prediction 3.ipynb`

3. For visualization, open the Power BI report (`Churn Analysis Report.pbix`) using Power BI Desktop.

## Project Flow
The project follows a structured approach:
1. Data preprocessing and feature engineering
2. Model training and validation
3. Model evaluation and visualization
4. Result interpretation and reporting

## Model Tracking and Experiment Management
The project leverages MLFlow for comprehensive experiment tracking and model management. MLFlow is used because it provides:
- Model versioning and lifecycle management
- Experiment tracking with metrics and parameters
- Artifact management for storing model files and visualizations
- Model deployment capabilities
- Centralized tracking of multiple model iterations
- Easy comparison between different model versions and experiments

## Machine Learning Models
The project implements and evaluates multiple machine learning models across different notebooks:

### Primary Models
- Random Forest Classifier (Churn Prediction.ipynb)
  - Used for churn prediction
  - Features calibration and probability estimation
  - Includes feature importance analysis
  - Configurable hyperparameters for optimization
  - Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC AUC

- Logistic Regression (Churn Prediction 2.ipynb)
  - Baseline classification model
  - Performance metrics:
    - Accuracy: 0.7795
    - Precision: 0.6805
    - Recall: 0.5014
    - F1-Score: 0.5774
    - ROC AUC: 0.8096
  - Includes coefficient analysis and calibration plots

- XGBoost Classifier (Churn Prediction 3.ipynb)
  - Gradient boosting model
  - Performance metrics:
    - Accuracy: 0.8236
    - Precision: 0.7172
    - Recall: 0.6814
    - F1-Score: 0.6989
    - ROC AUC: 0.8769
  - Includes feature importance and calibration analysis

- LightGBM Classifier (Churn Prediction 3.ipynb)
  - Gradient boosting model optimized for speed
  - Performance metrics:
    - Accuracy: 0.8236
    - Precision: 0.7172
    - Recall: 0.6814
    - F1-Score: 0.6989
    - ROC AUC: 0.8769
  - Includes feature importance and calibration analysis

### Model Comparison
- All models are tracked using MLFlow for performance comparison
- Comprehensive evaluation metrics including:
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - F1-Score
  - ROC AUC
  - Average Precision
  - Log Loss
  - Matthews Correlation
  - Balanced Accuracy
  - Specificity
  - False Positive Rate
  - False Negative Rate
  - Negative Predictive Value

### Model Selection
- Models are evaluated using cross-validation
- Performance metrics are logged in MLFlow for comparison
- Final model selection based on:
  - Prediction accuracy
  - Model interpretability
  - Computational efficiency
  - Business requirements

## Visualization and Analysis
The project includes comprehensive visualization outputs for model analysis and interpretation:
- Calibration plots for probability estimates
- Feature importance charts showing key predictors
- Prediction distribution plots for model performance
- Tree depth visualizations for Random Forest model
- Confusion matrices for classification performance
- Additional analysis plots in development notebooks

## Contributing
To contribute to this project:
1. Clone the repository
2. Set up the required dependencies
3. Run the notebooks in sequence
4. Submit improvements or bug fixes

## License
This project is for educational and research purposes only. Please ensure proper attribution if using the code or models in other projects.

---
Last Updated: 2025-07-15


