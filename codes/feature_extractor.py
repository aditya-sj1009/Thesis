import cv2
import numpy as np
from preprocessing import skin_detection
from skimage.feature import graycomatrix, graycoprops

# 1. Color Histogram Features
def extract_color_histogram_features(img):
    """Extracts color histogram features from the image."""
    hist_features = []
    for channel in range(img.shape[2]):  # Iterate over BGR channels
        hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
        hist_features.extend(hist.flatten())  # Flatten and append to feature list
    return np.array(hist_features)


# 2. Texture Features (Gray-Level Co-occurrence Matrix - GLCM)
def extract_texture_features(img):
    """Extracts texture features using GLCM."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Compute GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(gray_img, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    
    # Extract texture properties
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return np.array([contrast, correlation, energy, homogeneity])


# 3. Color Space Transformation Features (YCbCr Mean and Std Dev)
def extract_ycbcr_features(img):
    """Extracts mean and standard deviation of YCbCr channels."""
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    y_mean = np.mean(img_ycbcr[:, :, 0])
    cb_mean = np.mean(img_ycbcr[:, :, 1])
    cr_mean = np.mean(img_ycbcr[:, :, 2])
    
    y_std = np.std(img_ycbcr[:, :, 0])
    cb_std = np.std(img_ycbcr[:, :, 1])
    cr_std = np.std(img_ycbcr[:, :, 2])
    
    return np.array([y_mean, cb_mean, cr_mean, y_std, cb_std, cr_std])



# Combine All Features into a Single Feature Vector
def combine_features(img):
    """Combines all extracted features into a single feature vector."""
    skin_roi, _ , _ , _ , _ = skin_detection(img)
    color_histogram_features = extract_color_histogram_features(skin_roi)
    texture_features = extract_texture_features(skin_roi)
    ycbcr_features = extract_ycbcr_features(skin_roi)
    combined_features = np.concatenate((color_histogram_features[0:50],
                                        texture_features,
                                        ycbcr_features))
    return combined_features

# img = cv2.imread('NJN/normal/normal (1).jpg')
# final_feature_vector = combine_features(img)

# # # Print Feature Vector Shape and Example Values
# print(f"Final Feature Vector Shape: {final_feature_vector.shape}")
# print(f"Example Feature Values: {final_feature_vector[:10]}")

# # Extract Color Histogram Features
# histogram_feature= extract_color_histogram_features(img)
# print(f"histogram features: {histogram_feature.shape}")
# i=0
# for item in enumerate(histogram_feature):
#     i+=1
#     print(f"Shape of item {i}: {np.array(item).shape}")
    


# Extract Texture Features
# texture_feature = extract_texture_features(img)
# print(f"texture features: {texture_feature.shape}")
# i=0
# for item in enumerate(texture_feature):
#     i+=1
#     print(f"Shape of item {i}: {np.array(item).shape}")

# print (texture_feature)


# # Extract YCbCr Features

# ycbcr_feature = extract_ycbcr_features(img)
# i=0
# for item in enumerate(ycbcr_feature):
#     i+=1
#     print(f"Shape of item {i}: {np.array(item).shape}")

# print(f"ycbcr features: {ycbcr_feature.shape}")
# print (ycbcr_feature)

# # Save Feature Vector for ML Model Input
# np.save('feature_vector.npy', final_feature_vector)  # Save as .npy file
