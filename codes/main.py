import pandas as pd
import numpy as np
from models import svm_model, random_forest_model, knn_model, xgboost_model, evaluate_model
from data import X_train, X_test, y_train, y_test


svm = svm_model(X_train, y_train)
svm_results = evaluate_model(svm, X_test, y_test)

rf = random_forest_model(X_train, y_train)
rf_results = evaluate_model(rf, X_test, y_test)

knn = knn_model(X_train, y_train)
knn_results = evaluate_model(knn, X_test, y_test)

xgb = xgboost_model(X_train, y_train)
xgb_results = evaluate_model(xgb, X_test, y_test)

# Print Results for Each Model
print("SVM Results:", svm_results)
print("Random Forest Results:", rf_results)
print("k-NN Results:", knn_results)
print("XGBoost Results:", xgb_results)