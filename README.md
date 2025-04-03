# DIABETES RISK PREDICTION
## Leveraging Advance Supervised Machine Learning Models to Empower Early Detection For Diabetes Risk.
![image](https://github.com/user-attachments/assets/f1d3d305-92a2-4a2e-8382-ced840c21ea3)

***Disclaimer⚠️:** All datasets and reports do not contain real proprietary, confidential, or sensitive information from any company, institution, or individual. All info are dummy and design to demonstrate my capabilities in advance predictive ML modeling.*
### INTRODUCTION
**Stark Health Clinic**, a leader in technology-driven healthcare, aims to enhance patient outcomes and optimize resource allocation through predictive modeling. This project leverages machine learning to identify individuals at risk of developing diabetes, enabling early and targeted interventions. By analyzing patient data, it seeks to develop a robust and accurate model to predict diabetes onset. The initiative includes exploratory data analysis, feature engineering, and the training of multiple supervised learning models to identify critical patterns. This proactive approach will reduce healthcare costs, improve patient care, and strengthen the clinic's role in combating diabetes.
![image](https://github.com/user-attachments/assets/464b2f41-8b12-42e1-bac3-720b4a98dd67)

### PROBLEM STATEMENT
- Diabetes poses significant health risks to patients and creates financial challenges for healthcare providers.
- Current early detection methods at Stark Health Clinic lack precision, resulting in missed opportunities for timely interventions.
- Inaccurate predictions lead to delayed care, increased complications, and higher treatment costs.
- The clinic requires a reliable and robust predictive model to identify high-risk individuals effectively.

### AIM OF THE PROJECT
- Develop a robust machine learning model to predict the likelihood of diabetes onset in individuals.
- Enable early identification of high-risk patients to facilitate timely and targeted preventive interventions.
- Improve patient outcomes by reducing diabetes-related complications through proactive care.
- Optimize healthcare resource allocation by prioritizing at-risk individuals.
- Lower long-term healthcare costs associated with diabetes management and treatment.
- Strengthen Stark Health Clinic's role as a leader in technology-driven and patient-focused care.

### METHODOLOGY
- **STEP 1: Data Cleaning:**  
  - Handle missing values using appropriate imputation techniques.  
  - Remove duplicate records and irrelevant columns that do not contribute to prediction.  
  - Identify and correct anomalies in the dataset to ensure data quality.  

- **STEP 2: Exploratory Data Analysis (EDA):**  
  - Visualize feature distributions, relationships, and correlations using plots like histograms and heatmaps.  
  - Identify patterns, trends, and anomalies that may influence diabetes risk.  
  - Formulate hypotheses to guide feature engineering and model selection.  

- **STEP 3: Data Preprocessing:**  
  - Scale or normalize numerical features and encode categorical variables for compatibility with machine learning models.  
  - Split the data into training, validation, and test sets to ensure robust evaluation.  

- **STEP 4: Model Training:**  
  - Select and train machine learning models such as Logistic Regression, Random Forest, or Gradient Boosting.  
  - Conduct hyperparameter tuning and k-fold cross-validation for model improvement.  
  - Experiment with multiple algorithms and compare their performance.  

- **STEP 5: Model Evaluation:**  
  - Assess model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.  
  - Analyze performance across subsets (e.g., borrower income levels) and perform error analysis.  
  - Compare results to a baseline model to measure improvements.  

- **STEP 6: Model Optimization:**  
  - Fine-tune hyperparameters using techniques like Grid Search or Random Search.  
  - Apply regularization or ensemble methods to address overfitting and enhance performance.  
  - Refine feature selection and ensure the model generalizes well to unseen data.
    
### LIBRARIES
- **Data Manipulation and Analysis**
  - **Pandas:** Used for data cleaning, manipulation, and exploration, including handling missing values and creating data frames.
  - **NumPy:** Provides support for numerical operations, such as array manipulations and mathematical computations.
- **Data Visualization**
  - **Matplotlib:** A plotting library for creating static, interactive, and animated visualizations (e.g., histograms, scatter plots).
  - **Seaborn:** Built on Matplotlib, it simplifies the creation of aesthetically pleasing and informative statistical graphics (e.g., heatmaps, boxplots).
- **Machine Learning**
  - **Scikit-Learn:** The primary library for machine learning tasks such as model training, evaluation, hyperparameter tuning, and preprocessing (e.g., scaling, encoding).
- **Model Optimization**
  - **XGBoost:** An advanced machine learning library for Gradient Boosting, known for its high performance and scalability.
  - **LightGBM:** A lightweight Gradient Boosting library optimized for speed and accuracy in large datasets.
- **Environment and Workflow**
  - **Jupyter Notebook:** An interactive development environment for running and documenting Python code in a notebook format.
  - **Anaconda:** A distribution that simplifies Python package management and deployment, including pre-installed libraries for data science.

### Explorative Data Analysis
#### Numerica Data
Before performing the exploratory data analysis (EDA), the dataset was examined for missing values and anomalies. The results showed no missing values, and the only detected anomaly was the date column being in an object format, which was converted to an integer for improved data accuracy. The EDA revealed that the distribution of **Age** is standard with no outliers, while the distribution of **BMI (Body Mass Index)** is right-skewed but also free of outliers. insights provide a solid foundation for feature selection and model building.
![image](https://github.com/user-attachments/assets/b83dfa80-9193-420d-a27d-70ad5adcb25f)

A correlation analysis of the numerical variables indicated no negative correlations among the features. The strongest correlation was observed between **Blood Glucose Level** and **Diabetes** at 0.42, followed by **HbA1c Level** and **Diabetes** at 0.40, and **BMI** and **Age** at 0.34. The weakest correlation was between **HbA1c Level** and **Age**, with a value of 0.10. These 
![image](https://github.com/user-attachments/assets/c708b698-2353-4b42-b9e7-d179ed24a959)

#### Categorical Data
The analysis of categorical data revealed that the **Smoking History** variable is heavily right-skewed. A univariate and bivariate examination of the categorical data showed that a higher proportion of males have diabetes compared to females. Further analysis of smoking history in relation to diabetes status indicated that former smokers have the highest prevalence of diabetes, followed by those categorized as "ever" and "never" smokers. When comparing smoking history with age, patients over 50 years old were predominantly former smokers. Additionally, examining smoking history against diabetes status showed that patients with no smoking information had the highest proportion of negative diabetes cases (0), while those who never smoked exhibited the highest proportion of positive diabetes cases (1) compared to other categories. These insights highlight key patterns that can inform feature engineering and predictive modeling efforts.
![image](https://github.com/user-attachments/assets/3fdb22f1-ac93-4ea2-93d0-c5a8e218f1ed)

#### Data Preprocessing
During the data preprocessing stage, categorical variables were encoded into numerical formats to enhance compatibility and accuracy during model training. Additionally, the dataset was split into an 80% training set and a 20% testing set to ensure a robust evaluation of the model's performance. Feature scaling was also applied to normalize numerical variables, ensuring that all features contributed proportionately during model training and preventing dominance by variables with larger scales. These preprocessing steps established a solid foundation for accurate and reliable predictive modeling.
![image](https://github.com/user-attachments/assets/65536b81-94da-4d8a-af8f-426eb32ca4dd)

#### Model Training
Following data preprocessing, model training commenced with the **Logistic Regression** model to establish a baseline performance. Subsequently, other supervised learning models, including **Decision Tree**, **Stochastic Gradient Descent (SGD)**, and **Random Forest**, were explored to identify the model delivering the best performance. This iterative approach ensures the selection of a robust and accurate predictive model for diabetes onset.
![image](https://github.com/user-attachments/assets/5b2723af-a110-46ec-9cc5-d7f7395bfef3)

#### Model Evaluation
The trained models were evaluated to assess their performance across key metrics, including **accuracy**, **precision**, **recall**, and **F1 score**, yielding an average score of 96% across all models. Among the four trained models, the **Random Forest** classifier demonstrated the best performance, achieving a **100% recall**, **97% precision**, **98% F1 score**, and **97% accuracy**. These results highlight Random Forest's superior ability to balance predictive accuracy and sensitivity, making it the most effective model for diabetes risk prediction in this project.
![image](https://github.com/user-attachments/assets/597ff044-0b13-4429-ad6c-250047e2e5c2)

### KEY INSIGHTS
- **Data Quality**: No missing values were identified, and anomalies such as date format issues were resolved to ensure data accuracy.  
- **Feature Analysis**:  
  - Age distribution was standard, and BMI displayed a right-skewed distribution with no outliers.  
  - Strongest correlations included **Blood Glucose Level vs. Diabetes** (0.42) and **HbA1c Level vs. Diabetes** (0.40).  
  - Weakest correlation observed was between **HbA1c Level and Age** (0.10).  
- **Categorical Insights**:  
  - Smoking history was heavily right-skewed, with former smokers showing the highest diabetes prevalence.  
  - Patients above 50 years old were predominantly former smokers, and those who never smoked had the highest positive diabetes cases.  
- **Modeling Results**:  
  - Logistic Regression provided a baseline, while Decision Tree, SGD, and Random Forest models improved predictive performance.  
  - Random Forest outperformed all other models, achieving **100% recall**, **97% precision**, **98% F1 score**, and **97% accuracy**.  
- **Impactful Features**: Blood Glucose Level and HbA1c Level emerged as key predictors of diabetes onset.  
- **Project Outcome**: The Random Forest model's exceptional performance ensures robust diabetes risk prediction, enabling timely and targeted interventions for improved patient outcomes.  

### RECOMENDATION 
- **Focus on Key Predictors**: Prioritize monitoring **Blood Glucose Level** and **HbA1c Level** during routine checkups, as they are the most significant indicators of diabetes risk.  

- **Targeted Interventions for Smokers**: Develop specific preventive programs for former and current smokers, especially those over 50 years old, as they show a higher prevalence of diabetes.  

- **Enhanced Screening Protocols**: Implement Random Forest-based predictive models in regular screenings to identify high-risk patients with greater accuracy and sensitivity.  

- **Personalized Care Plans**: Use the model’s predictions to design personalized care plans for at-risk individuals, focusing on lifestyle modifications, regular monitoring, and early interventions.  

- **Data-driven Resource Allocation**: Allocate healthcare resources more efficiently by focusing on high-risk groups, ensuring timely interventions and reducing long-term costs.  


### CONCLUSION
This project successfully developed a robust diabetes prediction model using advanced machine learning techniques, with the Random Forest classifier demonstrating exceptional performance in identifying at-risk individuals. Through comprehensive data preprocessing, exploratory analysis, and rigorous model evaluation, key predictors such as Blood Glucose Level and HbA1c Level were identified as critical to diabetes risk. The insights derived provide Stark Health Clinic with actionable recommendations to enhance patient care, optimize resource allocation, and implement targeted preventive measures. By integrating this predictive model into its operations, Stark Health can take a proactive approach to combating diabetes, improving patient outcomes, and reducing long-term healthcare costs.

### THANK YOU
For more information, you can contact me
![image](https://github.com/user-attachments/assets/f1fa9d80-667e-421c-b5fa-6d796010c029)

