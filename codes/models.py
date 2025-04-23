import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

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

def evaluate_dl_model(model, X_test, y_test):
    """
    Evaluates a Keras/TensorFlow deep learning model on test data.
    Assumes y_test is categorical (one-hot) or integer labels.
    """
    # Predict class probabilities
    y_pred_probs = model.predict(X_test)
    # If output is probabilities, take argmax for class labels
    if y_pred_probs.shape[-1] > 1:
        y_pred = np.argmax(y_pred_probs, axis=1)
        # If y_test is one-hot, convert to class labels
        if len(y_test.shape) > 1 and y_test.shape[-1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
    else:
        # Binary output
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = y_test.flatten()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
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
    svm = SVC(kernel='linear',class_weight='balanced', random_state=42)
    svm.fit(X_train, y_train)
    return svm



# 2. Random Forest Classifier
def random_forest_model(X_train, y_train):
    """Trains a Random Forest model."""
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',random_state=42)
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
    xgb = XGBClassifier(scale_pos_weight= (sum(y_train==0)/sum(y_train==1)),use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    return xgb

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_resnet50_model(input_shape, num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False  # Freeze base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

