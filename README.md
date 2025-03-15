# Breast Cancer Prediction using Machine Learning and GCP

## Overview
This project utilizes machine learning techniques to predict breast cancer based on clinical data. The models used include Logistic Regression, Decision Tree, and Random Forest. The dataset used is the Wisconsin Diagnostic Breast Cancer dataset (1995). The project is hosted on Google Cloud Platform (GCP) to take advantage of its cloud computing services.

## Project Structure
- `GCP_Breast_Cancer.py`: Python script implementing machine learning models.
- `data.csv`: Dataset used for training and testing.
- `requirements.txt`: List of dependencies.

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-link>
   cd <repo-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model:
   ```bash
   python GCP_Breast_Cancer.py
   ```

## Dependencies
- NumPy
- Pandas
- Scikit-learn

## Model Performance
| Model             | Accuracy (%) |
|------------------|-------------|
| Logistic Regression | 88.44        |
| Decision Tree      | 100.00       |
| Random Forest      | 94.72        |

## Cloud Deployment
The model can be deployed using Google Cloud services such as:
- Google Cloud Storage (for dataset and model storage)
- Google Cloud Compute Engine (for model execution)
- Cloud SQL (for structured data storage)

## References
- [Breast Cancer Dataset](https://www.kaggle.com/buddhiniw/breast-cancer-prediction)
- [Scikit-learn Documentation](https://scikit-learn.org/)
