# Skin Cancer Classification Project

## Overview
This project aims to classify skin lesions as cancerous or non-cancerous using the ISIC 2024 - Skin Cancer Detection with 3D-TBP dataset. The project implements machine learning and deep learning models to assist in early detection, improving diagnostic accuracy and potentially saving lives.

## Features
- Preprocessing of the 3D-TBP dataset for effective model training.
- Advanced deep learning model for improved accuracy.
- Comprehensive evaluation metrics: Accuracy, Precision, Recall, F1 Score.
- Analysis of feature importance using Information Gain, Gain Ratio, and Gini Index.

## Dataset
The project uses the **ISIC 2024 - Skin Cancer Detection** dataset, which contains detailed data on various types of skin lesions. The dataset was preprocessed to ensure high-quality inputs for the models.

- **Dataset source:** [Kaggle - ISIC 2024](https://www.kaggle.com/)
- **Features included:**
  - Lesion images and corresponding labels.
  - Clinical features: age, diagnosis, lesion location, etc.

## Workflow
1. **Data Preprocessing**
   - Handling missing values and anomalies.
   - Normalization and augmentation of image data.
2. **Exploratory Data Analysis (EDA)**
   - Understanding data distributions.
   - Visualizing patterns and correlations.
![Screenshot from 2024-11-23 11-46-08](https://github.com/user-attachments/assets/3403ab34-74d8-4c32-8dc5-8ea0c1ab2380) ![Screenshot from 2024-11-23 11-46-24](https://github.com/user-attachments/assets/06947364-5c31-4fd2-9053-5c4f0f2abcae)
![Screenshot from 2024-11-23 11-46-36](https://github.com/user-attachments/assets/4daecdb0-8e12-4c0a-a797-e24e26b7169e) ![Screenshot from 2024-11-23 11-46-49](https://github.com/user-attachments/assets/47b7839d-8826-4bfd-a3d1-67bad9b23cc4)
![Screenshot from 2024-11-23 11-47-28](https://github.com/user-attachments/assets/3f604d41-4187-4873-b2ac-871315f98e57)

3. **Model Training**
   - Initial evaluation with Naive Bayes and Decision Tree classifiers.
   - Advanced classification using a deep learning architecture (CNN).
4. **Evaluation**
   - Performance metrics (Accuracy, Precision, Recall, F1 Score).
   - Feature analysis using Information Gain, Gain Ratio, and Gini Index.
![Screenshot from 2024-11-23 11-48-23](https://github.com/user-attachments/assets/5ff84e0a-8be5-4467-a057-0d6beea859a8)
![Screenshot from 2024-11-23 11-48-05](https://github.com/user-attachments/assets/c88cc9dc-90b2-4231-8089-7bf7c5dd4183)
![Screenshot from 2024-11-23 11-48-41](https://github.com/user-attachments/assets/d5b20d84-a17a-4d8c-a5f5-7896309cd00b)

## Results
- Naive Bayes and Decision Tree classifiers: Baseline metrics for classification.
- Deep learning model: Achieved a significantly higher accuracy with detailed insights i.e. 77%.

![Screenshot from 2024-11-23 11-48-56](https://github.com/user-attachments/assets/30c373bf-3501-4971-a9a2-66c77d151d38)

## Tools & Technologies
- **Programming Language**: Python
- **Libraries**:
  - Pandas, NumPy, Matplotlib, Seaborn (for data analysis and visualization)
  - Scikit-learn (for Naive Bayes and Decision Tree)
  - TensorFlow/Keras (for deep learning)
- **Development Environment**: Jupyter Notebook

## Future Work
- Extend the project to include real-time lesion detection using a webcam or mobile camera.
- Incorporate advanced architectures like transformers or ensemble learning for improved performance.
- Explore explainability methods like Grad-CAM for better understanding of model predictions.

## Acknowledgments
- ISIC Archive for providing the dataset.
- Open-source libraries and frameworks used in this project.
