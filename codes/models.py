import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Example dataset (replace with your actual feature matrix and labels)
# X: Feature matrix (e.g., extracted features from preprocessed images)
# y: Labels (e.g., 0 for non-jaundiced and 1 for jaundiced cases)
# X = np.load('feature_matrix.npy')  # Replace with your feature matrix file
# y = np.load('labels.npy')          # Replace with your labels file

# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define evaluation function
def evaluate_model(model, X_test, y_test):
    """Evaluates the model on test data using various metrics."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Matthews Correlation Coefficient": mcc
    }

# 1. Support Vector Machine (SVM)
def svm_model(X_train, y_train):
    """Trains an SVM model."""
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    return svm



# 2. Random Forest Classifier
def random_forest_model(X_train, y_train):
    """Trains a Random Forest model."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf



# 3. k-Nearest Neighbors (k-NN)
def knn_model(X_train, y_train):
    """Trains a k-NN model."""
    knn = KNeighborsClassifier(n_neighbors=5)  # You can tune the number of neighbors
    knn.fit(X_train, y_train)
    return knn



# 4. XGBoost Classifier
def xgboost_model(X_train, y_train):
    """Trains an XGBoost model."""
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    return xgb





