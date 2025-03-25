import cv2
import numpy as np

# Define image size for resizing (e.g., 224x224 for deep learning models)
IMAGE_SIZE = (224, 224)

# 1. White Balance Adjustment (Simple Gray World Assumption)
def white_balance(img):
    """Applies a simple white balance using the gray world assumption."""
    avg_color_per_row = np.average(img, axis=0)
    avg_colors = np.average(avg_color_per_row, axis=0)
    avg_b, avg_g, avg_r = avg_colors

    balance_b = 128 / avg_b
    balance_g = 128 / avg_g
    balance_r = 128 / avg_r

    balanced_img = img.copy()
    balanced_img[:, :, 0] = np.clip(balanced_img[:, :, 0] * balance_b, 0, 255)
    balanced_img[:, :, 1] = np.clip(balanced_img[:, :, 1] * balance_g, 0, 255)
    balanced_img[:, :, 2] = np.clip(balanced_img[:, :, 2] * balance_r, 0, 255)

    return balanced_img.astype(np.uint8)

# 2. Skin Detection for ROI Selection
def skin_detection(img):
    """Detects skin regions based on color thresholds in the YCrCb color space."""
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Define skin color range in YCrCb
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    # Create a mask for skin regions
    mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)

    # Apply morphological operations to refine the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find the largest contour (skin region)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        skin_roi = img[y:y+h, x:x+w]  # Extract the skin ROI
        return skin_roi, x, y, w, h
    else:
        return None, None, None, None, None


# 3. Contrast Enhancement (CLAHE)
def clahe_enhancement(img):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img





def preprocess_image(img_path):
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return None
    # Resize the image
    img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0
    
    img_wb = white_balance(img_normalized)
    enhanced_img = clahe_enhancement(img_wb)
    
    # 4. Color Space Transformation (YCbCr)
    # img_ycbcr = cv2.cvtColor(enhanced_skin_roi, cv2.COLOR_BGR2YCrCb)
    
    return enhanced_img
    


# # Display the images
# cv2.imshow('Original Image', img)
# cv2.imshow('White Balanced Image', img_wb)
# if x is not None:
#     cv2.rectangle(img_wb, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     cv2.imshow('Skin ROI', skin_roi)
# cv2.imshow('CLAHE Enhanced Skin ROI', enhanced_skin_roi)
# cv2.imshow('YCbCr Image', img_ycbcr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
