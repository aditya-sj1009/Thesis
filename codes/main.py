#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import svm_model, random_forest_model, knn_model, xgboost_model,evaluate_model
from models import build_cnn_model, build_resnet50_model, evaluate_dl_model
from models import build_cnn_feature_extractor, train_cnn_svm_model, predict_cnn_svm
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

#CNN Model

y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)
# cnn = build_cnn_model(input_shape=(224,224,3), num_classes=2)
# cnn.fit(X_train_img, y_train_cat, epochs=20, batch_size=16, validation_split=0.1)

# cnn_results = evaluate_dl_model(cnn, X_test_img, y_test_cat)

# ResNet50 Model

# resnet = build_resnet50_model(input_shape=(224,224,3), num_classes=2)
# resnet.fit(X_train_img, y_train_cat, epochs=10, batch_size=16, validation_split=0.1)

# resnet_results = evaluate_dl_model(resnet, X_test_img, y_test_cat)

# Hybrid CNN + SVM Model

# Step 1: Build feature extractor (e.g., ResNet50 or custom CNN, without top layer)
feature_extractor = build_cnn_feature_extractor(input_shape=(224,224,3))

# Step 2: Train SVM on CNN features
scaler, svm = train_cnn_svm_model(feature_extractor, X_train_img, y_train)

# Step 3: Predict on test set
y_pred = predict_cnn_svm(feature_extractor, scaler, svm, X_test_img)

# Step 4: Evaluate
cnn_svm_results = evaluate_model(svm, y_test, y_pred) 

# Print Results for Each Model
print("SVM Results:", svm_results)
print("Random Forest Results:", rf_results)
print("k-NN Results:", knn_results)
print("XGBoost Results:", xgb_results)
# print("CNN Results:", cnn_results)
# print("ResNet50 Results:", resnet_results)
print("Hybrid CNN + SVM Results:", cnn_svm_results)
#%%
# --- Graphical Representation ---
# Ensure the results are dictionaries with keys: 'accuracy', 'precision', 'recall', 'f1-score'
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Matthews Correlation Coefficient']

svm_values = [svm_results[m] for m in metrics]
rf_values = [rf_results[m] for m in metrics]
knn_values = [knn_results[m] for m in metrics]
xgb_values = [xgb_results[m] for m in metrics]
# cnn_values = [cnn_results[m] for m in metrics]
# resnet_values = [resnet_results[m] for m in metrics]
cnn_svm_values = [cnn_svm_results[m] for m in metrics]


x = np.arange(len(metrics))
width = 0.2

plt.figure(figsize=(12, 7))
bars1 = plt.bar(x - 1.5*width, svm_values, width=width, label='SVM', color='#17becf')
bars2 = plt.bar(x - 0.5*width, rf_values, width=width, label='Random Forest', color='#ffbb78')
bars3 = plt.bar(x + 0.5*width, knn_values, width=width, label='k-NN', color='#c0493d')
bars4 = plt.bar(x + 1.5*width, xgb_values, width=width, label='XGBoost', color='#f5f5dc')
bars5 = plt.bar(x + 2.5*width, cnn_values, width=width, label='CNN', color='#9467bd')
bars6 = plt.bar(x + 3.5*width, resnet_values, width=width, label='ResNet50', color='#8c564b')
bars7 = plt.bar(x + 4.5*width, cnn_svm_values, width=width, label='Hybrid CNN + SVM', color='#2ca02c')
# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add horizontal line at y=0.5 for reference
plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)


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
