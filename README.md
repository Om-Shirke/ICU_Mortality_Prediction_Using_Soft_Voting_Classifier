# ICU_Mortality_Prediction_Using_Soft_Voting_Classifier

## 📌 Project Description

This project focuses on developing an ICU Mortality Prediction System for patients with heart failure, as part of a dissertation for the MS in Data Science at Liverpool John Moores University, UK. An extensive literature review was conducted to analyze existing mortality prediction models, revealing a research gap where most studies prioritized accuracy or ROC score over recall. Since missing high-risk cases in an ICU setting can have severe consequences, this study aimed to bridge this gap by optimizing for recall while maintaining a balance across other evaluation metrics. The dataset, sourced from the MIMIC-III database, includes key features such as vital signs, lab results, and demographic details. This project involves extensive data preprocessing, feature selection, handling class imbalance with ADASYN, and developing a custom-weighted soft voting ensemble using SVM, KNN, Random Forest, and XGBoost. The final model achieves a recall of 99% and an ROC score of 98%, demonstrating its potential for real-world ICU deployment to improve patient outcomes.

## 🔧 Tech Stack  

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Seaborn, Matplotlib, Statsmodels, Imbalanced-learn  
- **Dataset:** MIMIC-III ICU patient data (curated subset for heart failure cases)  
- **Data Processing Techniques:** Mean, median, and iterative imputation for missing values, feature scaling, and class imbalance handling using ADASYN  
- **Feature Selection:** Statistical tests (Chi-Square, t-test, Mann-Whitney U test), exploratory data analysis (EDA)
- **Model Used:** Custom-weighted soft voting ensemble (SVM, KNN, Random Forest, XGBoost)  
- **Evaluation Metrics:** Optimized for recall (99%), with additional analysis of precision, ROC score (98%), specificity, and F1-score  

## 🔍 Exploratory Data Analysis (EDA)  

- **Data Cleaning & Preprocessing:** Handled missing values using mean, median, and iterative imputation techniques. Addressed outliers using IQR-based capping. Applied feature scaling to standardize numerical variables.  
- **Feature Engineering:** Created derived features from physiological data to enhance model performance and improve interpretability.  
- **Handling Class Imbalance:** Implemented ADASYN to generate synthetic samples for the minority class, ensuring balanced model training.  
- **Feature Selection:** Conducted statistical tests (Chi-Square, t-test, Mann-Whitney U test) and exploratory data analysis (EDA) to identify the most relevant predictors.  
- **Visualization:** Used Matplotlib and Seaborn for trend analysis, feature distributions, and correlation heatmaps to explore relationships between patient variables and ICU mortality.  

## 🚀 Model Training & Evaluation  

- **Train-Test Split:** 70% training, 30% testing.  
- **Data Balancing:** Applied ADASYN to address class imbalance and improve model performance.  
- **Model Used:** Custom-weighted soft voting ensemble consisting of:  
  - Support Vector Machine (SVM)  
  - k-Nearest Neighbors (KNN)  
  - Random Forest  
  - XGBoost  
- **Hyperparameter Tuning:** Used GridSearchCV with 3-fold cross-validation, optimizing for recall.  
- **Threshold Optimization:** Set a custom classification threshold (0.4) to enhance recall while maintaining balance across other metrics.  
- **Evaluation Metrics:**  
  - **Recall** (Optimized for ICU mortality prediction)  
  - **Precision, F1-score** (Balanced to avoid excessive false positives)  
  - **ROC-AUC Score** (Assessing model discrimination power)  
  - **Confusion Matrix** (Visualizing classification performance)  
  - **Specificity & Sensitivity** (Analyzing trade-offs in medical predictions)  

## ⚡ Key Findings & Insights  

- **Optimizing Recall for ICU Mortality Prediction:**  
  The study successfully developed a custom-weighted soft voting ensemble classifier that prioritizes recall (99.86%) to minimize false negatives in mortality prediction. This ensures high-risk ICU patients are correctly identified, aligning with the critical need for proactive interventions.  

- **Balancing Other Performance Metrics:**  
  While recall was the primary focus, the model maintains a strong ROC-AUC score (98.29%), demonstrating high discrimination power. Despite a trade-off in specificity (41.37%) and precision (62.73%), these metrics are still competitive, ensuring the model remains practical for real-world ICU deployment.  

- **Robust Data Preprocessing & Feature Selection:**  
  Handling missing values through mean, median, and iterative imputation, along with IQR-based outlier treatment, significantly improved data quality. Advanced feature selection using statistical tests (Chi-Square, t-test, Mann-Whitney U) ensured only the most predictive variables were used, enhancing model interpretability.  

- **Addressing Class Imbalance Effectively:**  
  The dataset exhibited a severe mortality class imbalance, which was mitigated using ADASYN. This technique improved model learning, enabling better mortality prediction compared to traditional resampling methods.  

- **Impact of Ensemble Learning in Medical AI:**  
  Through extensive experimentation (50-60 model variations), the study demonstrated that a carefully designed soft voting classifier—leveraging SVM, KNN, Random Forest, and XGBoost—outperforms individual models. The ensemble approach captures complex patterns that standalone models often miss.  

- **Potential for Real-World ICU Deployment:**  
  The model’s high recall ensures that almost all mortality cases are identified in advance, allowing hospitals to allocate resources effectively and intervene earlier. While prioritizing recall, the model also maintains practical utility by balancing other performance metrics, making it suitable for clinical integration.  


