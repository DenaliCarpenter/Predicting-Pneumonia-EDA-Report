# Predicting Pneumonia in Chest X-Ray Images

Jupyter Notebook: https://github.com/DenaliCarpenter/predicting-pneumonia-eda-report/blob/main/EDA%20and%20Initial%20Model.ipynb

## Overview
This project focuses on detecting pneumonia in chest X-ray images using machine learning. The goal is to explore various modeling techniques to distinguish between normal and pneumonia-affected lungs.

## Dataset
The dataset consists of chest X-ray images categorized into **NORMAL** and **PNEUMONIA** classes. It includes:
- **Training set**: 5,216 images (74% pneumonia, 26% normal)
- **Test set**: 624 images (63% pneumonia, 37% normal)
- **Validation set**: 16 images (50% pneumonia, 50% normal)

## Exploratory Data Analysis (EDA)
### Data Cleaning
- **Removed duplicate images**: 26 from training, 6 from test.
- **Checked for corrupt files**: None found.
- **Converted all images to grayscale** for uniformity.

### Image Quality and Characteristics
- **Brightness & Contrast**: Normal images tend to be brighter, while pneumonia cases exhibit more contrast variation.
- **Texture Analysis** (GLCM): Pneumonia cases, especially bacterial, show lower energy and higher homogeneity.
- **Edge Strength (Sobel Filter)**: Normal images have sharper edges compared to pneumonia cases.
- **Fourier Transform Energy**: Normal cases contain more high-frequency details, while pneumonia cases exhibit smoother textures.
- **Image Blurriness (Laplacian Variance)**: Pneumonia cases tend to have slightly lower sharpness.

## Feature Engineering
- Extracted pixel intensity distributions.
- Computed GLCM features (contrast, correlation, energy, homogeneity).
- Measured edge strength and Fourier Transform energy.
- Applied **Principal Component Analysis (PCA)** to reduce dimensions while preserving 95% variance (657 components).

## Baseline and Initial Models
- **Dummy Classifier** (majority class baseline):
  - **Accuracy**: 62.6%
  - **Precision/Recall**: Biased toward predicting pneumonia due to dataset imbalance.
  - **Confusion Matrix**: Heavy misclassification of normal cases.
- **Random Forest Classifier** (Initial ML Model):
  - **Accuracy**: 66.7%
  - **Better classification of pneumonia cases** but still misclassifies many normal cases.
  - **Reveals importance of contrast, edge strength, and texture features**.

## Key Findings on Feature Importance
- **GLCM Contrast and Energy**: Pneumonia cases exhibit greater contrast variation and lower energy, indicating more irregular textures.
- **Fourier Transform Energy**: Pneumonia cases have lower high-frequency content, aligning with the presence of smoother opacities.
- **Edge Strength**: Pneumonia-affected lungs show weaker edges, likely due to increased fluid opacity.
- **Pixel Intensity Distribution**: Pneumonia cases tend to have a broader intensity range, reflecting lung opacities caused by infection.
- **Blurriness and Homogeneity**: Higher homogeneity in pneumonia cases suggests more uniform opacities across infected lungs.

## Next Steps
- Train a more advanced classification model (e.g., CNN or transfer learning).
- Address class imbalance using weighting or augmentation.
- Further optimize feature selection and engineering.