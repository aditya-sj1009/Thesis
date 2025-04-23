import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_image
from feature_extractor import combine_features
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Define the paths to the folders
data_dir = 'NJN'  # Replace with the path to your dataset folder
jaundiced_dir = os.path.join(data_dir, 'jaundice')
normal_dir = os.path.join(data_dir, 'normal')



# Initialize lists to store data and labels

def load_features(image_dir, label):
    """
    Load features from images in the specified directory and assign the given label.
    """
    features_data = []
    labels = []
    count=0
    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)
        preprocessed_img = preprocess_image(file_path)
        if preprocessed_img is not None:
            
            features = combine_features(preprocessed_img)
            features_data.append(features)
            labels.append(label)
            
    return features_data, labels

jaundiced_features, j_labels = load_features(jaundiced_dir, 1)
normal_features, n_labels = load_features(normal_dir, 0)

X= np.concatenate([jaundiced_features, normal_features])
y = np.concatenate([j_labels, n_labels])


# Split the dataset into training and testing sets
X_train_, X_test_, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit scaler, PCA, LDA only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_)
X_test_scaled = scaler.transform(X_test_)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train_pca, y_train)
X_test_lda = lda.transform(X_test_pca)

X_test=X_test_lda
X_train=X_train_lda

X_train = X_train_pca
X_test = X_test_pca

print("Train class distribution:", np.bincount(y_train))
print("Test class distribution:", np.bincount(y_test))

def load_img_data(image_dir, label):
    """
    Load images from the specified directory and assign the given label.
    """
    data = []
    labels = []
    count=0
    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)
        preprocessed_img = preprocess_image(file_path)
        if preprocessed_img is not None:
            data.append(preprocessed_img)
            labels.append(label)
            
    return data, labels

jaundice_img, j_img_label = load_img_data(jaundiced_dir, 1)
normal_img, n_img_label = load_img_data(normal_dir, 0)
X_img = np.concatenate([jaundice_img, normal_img])
y_img = np.concatenate([j_img_label, n_img_label])

X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(X_img, y_img, test_size=0.2, random_state=42)


# # Print dataset shapes
# print(f"Total images: {len(data)}")
# print(f"Training set shape: {X_train.shape}, Training labels shape: {y_train.shape}")
# print(f"Testing set shape: {X_test.shape}, Testing labels shape: {y_test.shape}")
# print(y[0:10])
# print(f"Training set shape: {X.shape})")
# print(f"Training set shape: {X_scaled.shape})")
# print(f"Training set shape: {X_pca.shape})")
# print(f"Training set shape: {X_lda.shape})")
# print(f"Jaudice set shape: {jaundiced_features.shape})")
# print(f"Normal set shape: {normal_features.shape})")

# Save the preprocessed data for future use (optional)
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)