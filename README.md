# Predicting Pneumonia in Chest X-Ray Images

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

## Baseline Model
- **Dummy Classifier** (majority class baseline):
  - **Accuracy**: 62.6%
  - **Precision/Recall**: Biased toward predicting pneumonia due to dataset imbalance.
  - **Confusion Matrix**: Heavy misclassification of normal cases.
- This highlights the need for more sophisticated models.

## Key Features Driving the Differences
Through EDA, we identified the following features as key discriminators between normal and pneumonia-affected lungs:
- **Texture Homogeneity**: Pneumonia cases show more uniform textures, particularly bacterial pneumonia.
- **Fourier Transform Energy**: Normal cases have higher frequency content, while pneumonia cases exhibit smoother textures.
- **Edge Strength**: Pneumonia-infected lungs display lower edge intensity, possibly due to fluid-filled regions causing blurring.
- **GLCM Contrast and Energy**: Normal cases have higher GLCM energy, whereas pneumonia cases show greater contrast variability.
- **Brightness and Pixel Intensity Distributions**: Normal images tend to be brighter, while pneumonia cases exhibit a wider range of pixel intensities.

## Next Steps
- Train a more advanced classification model (e.g., CNN or transfer learning).
- Address class imbalance using weighting or augmentation.
- Further optimize feature selection and engineering.