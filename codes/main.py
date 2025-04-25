#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import svm_model, random_forest_model, knn_model, xgboost_model,evaluate_model
from models import build_cnn_model, build_resnet50_model, evaluate_dl_model
from models import build_cnn_feature_extractor, train_cnn_svm_model, predict_cnn_svm
from models import build_tf_vit_model, train_tf_vit_model, predict_tf_vit_model
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
cnn = build_cnn_model(input_shape=(224,224,3), num_classes=2)
cnn.fit(X_train_img, y_train_cat, epochs=20, batch_size=16, validation_split=0.1)

cnn_results = evaluate_dl_model(cnn, X_test_img, y_test_cat)

# ResNet50 Model

resnet = build_resnet50_model(input_shape=(224,224,3), num_classes=2)
resnet.fit(X_train_img, y_train_cat, epochs=10, batch_size=16, validation_split=0.1)

resnet_results = evaluate_dl_model(resnet, X_test_img, y_test_cat)

# Hybrid CNN + SVM Model

# Step 1: Build feature extractor (e.g., ResNet50 or custom CNN, without top layer)
feature_extractor = build_cnn_feature_extractor(input_shape=(224,224,3))

# Step 2: Train SVM on CNN features
scaler, svm = train_cnn_svm_model(feature_extractor, X_train_img, y_train)

# Step 3: Predict on test set
y_pred = predict_cnn_svm(feature_extractor, scaler, svm, X_test_img)

# Step 4: Evaluate
cnn_svm_results = evaluate_model(svm, y_test, y_pred) 

#ViT model

vit_model, vit_processor = build_tf_vit_model(num_classes=2)
vit_model = train_tf_vit_model(vit_model, vit_processor, X_train_img, y_train, epochs=5, batch_size=16)
y_pred_vit = predict_tf_vit_model(vit_model, vit_processor, X_test_img)
vit_results = evaluate_predictions(y_test, y_pred_vit)

# Print Results for Each Model
print("SVM Results:", svm_results)
print("Random Forest Results:", rf_results)
print("k-NN Results:", knn_results)
print("XGBoost Results:", xgb_results)
print("CNN Results:", cnn_results)
print("ResNet50 Results:", resnet_results)
print("Hybrid CNN + SVM Results:", cnn_svm_results)
print("ViT Results:", vit_results)
#%%
# --- Graphical Representation ---
# Ensure the results are dictionaries with keys: 'accuracy', 'precision', 'recall', 'f1-score'
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Matthews Correlation Coefficient']

svm_values = [svm_results[m] for m in metrics]
rf_values = [rf_results[m] for m in metrics]
knn_values = [knn_results[m] for m in metrics]
xgb_values = [xgb_results[m] for m in metrics]
cnn_values = [cnn_results[m] for m in metrics]
resnet_values = [resnet_results[m] for m in metrics]
cnn_svm_values = [cnn_svm_results[m] for m in metrics]
vit_values = [vit_results[m] for m in metrics]

models = ['SVM', 'Random Forest', 'k-NN', 'XGBoost', 'CNN', 'ResNet50', 'Hybrid CNN + SVM']
values = [svm_values, rf_values, knn_values, xgb_values, cnn_values, resnet_values, cnn_svm_values]
colors = ['#17becf', '#ffbb78', '#c0493d', '#f5f5dc', '#9467bd', '#8c564b', '#2ca02c']

x = np.arange(len(metrics))  # label locations
width = 0.11 
fig, ax = plt.subplots(figsize=(16, 8))

for i, (model, vals, color) in enumerate(zip(models, values, colors)):
    bars = ax.bar(x + (i - 3)*width, vals, width=width, label=model, color=color)
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            yval + 0.01,
            f'{yval:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )


# Add grid and baseline
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)

# Ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Comparison of Model Performance Metrics')
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))

plt.tight_layout()
plt.show()
