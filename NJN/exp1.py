import cv2

import numpy as np
from skimage import color, feature
from sklearn.preprocessing import StandardScaler

def preprocess_image(image_path):
    # Step 1: Load the image
    img = cv2.imread(image_path)
    
    # Step 2: White balance correction
    img_wb = white_balance_correction(img)
    
    # Step 3: Skin detection and ROI extraction
    skin_mask = detect_skin(img_wb)
    roi = apply_mask(img_wb, skin_mask)
    
    # Step 4: Color space transformations
    b_channel = roi[:,:,0]  # Blue channel from RGB
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    cb_channel = ycrcb[:,:,2]  # Cb channel from YCrCb
    
    # Step 5: Feature extraction
    features = extract_features(roi, b_channel, cb_channel)
    
    # Step 6: Normalization
    normalized_features = normalize_features(features)
    
    return normalized_features

def white_balance_correction(img):
    # Simple white balance using gray world assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def detect_skin(img):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create binary mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def extract_features(roi, b_channel, cb_channel):
    # Color features
    color_features = [np.mean(b_channel), np.mean(cb_channel)]
    
    # Texture features using GLCM
    glcm = feature.graycomatrix(b_channel, [1], [0], symmetric=True, normed=True)
    contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
    energy = feature.graycoprops(glcm, 'energy')[0, 0]
    correlation = feature.graycoprops(glcm, 'correlation')[0, 0]
    
    texture_features = [contrast, dissimilarity, homogeneity, energy, correlation]
    
    # Combine all features
    features = color_features + texture_features
    
    return features

def normalize_features(features):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform([features])
    return normalized_features[0]

# Example usage
image_path = 'NJN\jaundice\jaundice (1).jpg'
preprocessed_features = preprocess_image(image_path)
print(f"Preprocessed features: {preprocessed_features}")
