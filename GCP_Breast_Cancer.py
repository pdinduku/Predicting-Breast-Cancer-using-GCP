import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load dataset
df = pd.read_csv("data.csv")
df.drop(columns=['id', 'Unnamed: 32'], errors='ignore', inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Define features and target
features = list(df.columns[1:])  # Use all available features
target = 'diagnosis'

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

def classification_model(model, train_df, features, target):
    model.fit(train_df[features], train_df[target])
    predictions = model.predict(train_df[features])
    accuracy = metrics.accuracy_score(predictions, train_df[target])
    print(f"Model Accuracy: {accuracy:.3%}")
    
    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    errors = []
    for train_idx, test_idx in kf.split(train_df):
        model.fit(train_df.iloc[train_idx][features], train_df.iloc[train_idx][target])
        errors.append(model.score(train_df.iloc[test_idx][features], train_df.iloc[test_idx][target]))
    
    print(f"Cross-Validation Score: {np.mean(errors):.3%}")

# Logistic Regression
print("Logistic Regression")
logistic_model = LogisticRegression(max_iter=1000)
classification_model(logistic_model, train_df, features, target)

# Decision Tree
print("Decision Tree")
decision_tree_model = DecisionTreeClassifier(max_depth=5)
classification_model(decision_tree_model, train_df, features, target)

# Random Forest
print("Random Forest")
random_forest_model = RandomForestClassifier(n_estimators=200, min_samples_split=10, max_depth=10, max_features='sqrt')
classification_model(random_forest_model, train_df, features, target)

# Feature Importance for Random Forest
feat_imp = pd.Series(random_forest_model.feature_importances_, index=features).sort_values(ascending=False)
print(feat_imp.head(10))  # Show top 10 important features
