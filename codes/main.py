import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import svm_model, random_forest_model, knn_model, xgboost_model, build_cnn_model, evaluate_model, evaluate_dl_model
from data import X_train, X_test, y_train, y_test
from data import X_train_img, X_test_img, y_train_img, y_test_img

from tensorflow.keras.utils import to_categorical


svm = svm_model(X_train, y_train)
svm_results = evaluate_model(svm, X_test, y_test)

rf = random_forest_model(X_train, y_train)
rf_results = evaluate_model(rf, X_test, y_test)

knn = knn_model(X_train, y_train)
knn_results = evaluate_model(knn, X_test, y_test)

xgb = xgboost_model(X_train, y_train)
xgb_results = evaluate_model(xgb, X_test, y_test)

y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)
cnn = build_cnn_model(input_shape=(224,224,3), num_classes=2)
cnn.fit(X_train_img, y_train_cat, epochs=20, batch_size=16, validation_split=0.1)

cnn_results = evaluate_dl_model(cnn, X_test_img, y_test_cat)


# Print Results for Each Model
# print("SVM Results:", svm_results)
# print("Random Forest Results:", rf_results)
# print("k-NN Results:", knn_results)
# print("XGBoost Results:", xgb_results)
print("CNN Results:", cnn_results)

#%%
# --- Graphical Representation ---
# Ensure the results are dictionaries with keys: 'accuracy', 'precision', 'recall', 'f1-score'
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Matthews Correlation Coefficient']

svm_values = [svm_results[m] for m in metrics]
rf_values = [rf_results[m] for m in metrics]
knn_values = [knn_results[m] for m in metrics]
xgb_values = [xgb_results[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.2

plt.figure(figsize=(12, 7))
bars1 = plt.bar(x - 1.5*width, svm_values, width=width, label='SVM', color='#17becf')
bars2 = plt.bar(x - 0.5*width, rf_values, width=width, label='Random Forest', color='#ffbb78')
bars3 = plt.bar(x + 0.5*width, knn_values, width=width, label='k-NN', color='#c0493d')
bars4 = plt.bar(x + 1.5*width, xgb_values, width=width, label='XGBoost', color='#f5f5dc')

# Add value labels on top of each bar
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            yval + 0.01, 
            f'{yval:.2f}', 
            ha='center', va='bottom', fontsize=9
        )

plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Comparison of Model Performance Metrics')
plt.legend()
plt.tight_layout()
plt.show()
