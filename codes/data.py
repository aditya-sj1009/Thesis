import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_image
from feature_extractor import combine_features

# Define the paths to the folders
data_dir = 'NJN'  # Replace with the path to your dataset folder
jaundiced_dir = os.path.join(data_dir, 'jaundice')
normal_dir = os.path.join(data_dir, 'normal')



# Initialize lists to store data and labels
data = []
labels = []

# Load images from the 'jaundiced' folder
count=0
for file_name in os.listdir(jaundiced_dir):
    file_path = os.path.join(jaundiced_dir, file_name)
    preprocessed_img = preprocess_image(file_path)
    if preprocessed_img is not None:
        features = combine_features(preprocessed_img)
        data.append(features)
        labels.append(1)  # Label for jaundiced cases

# Load images from the 'normal' folder
for file_name in os.listdir(normal_dir):
    file_path = os.path.join(normal_dir, file_name)
    preprocessed_img = preprocess_image(file_path)
    if preprocessed_img is not None:
        features = combine_features(preprocessed_img)
        data.append(features)
        labels.append(0)  # Label for normal cases

# Convert data and labels to NumPy arrays
# print(data)

# for i in range(190, 201):
#     for j in range(0, len(data[i])):
#         print (i,j,f'=',data[i][j].shape)
            
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# # Print dataset shapes
# print(f"Total images: {len(data)}")
# print(f"Training set shape: {X_train.shape}, Training labels shape: {y_train.shape}")
# print(f"Testing set shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

# Save the preprocessed data for future use (optional)
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)