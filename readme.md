# Hanover Diabetes Education and Engagement Program: Predictions for Prevention and Intervention
<<<<<<< HEAD
![alt text](image-9.png)
=======
![image](https://github.com/user-attachments/assets/495e5406-9cdc-488e-9697-022c503e54f5)
>>>>>>> d4d443ff0c2f096c35cc70bd7965fa8556de6f09

Final Project Submission

Author: Deon durrant
# Business and Data Understanding

## Problem Statement
In the United States, 1 in 3 adults are prediabetic, and an alarming 8 out of 10 individuals remain unaware of their condition. Without timely intervention, most prediabetic individuals are at high risk of developing type 2 diabetes, a chronic condition within 5 years. Diabetes is associated with severe health complications and financial burdens. According to the CDC 1 in 4 health care dollars is spent on people with diagnosed diabetes. In 2022 the USA spent $413 billion for diabetic care.  

The town of Hanover is taking a proactive, holistic approach to address this crucial health challenge by implementing a Diabetes Education and Engagement Program (DEEP). The initiative leverages data analysis to highlight the most at-risk patients so that they may be targeted for earlier healthcare intervention slowing the onset and progression of the disease. Additionally, the project seeks to improve early detection, provide targeted interventions, and empower individuals through education and engagement to prevent the progression of diabetes.

## Data Understanding 
The Diabetes Health Indicators Dataset data was sourced from Kaggle  https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset and contains healthcare statistics and lifestyle information of 253,680 survey responses to the CDC's BRFSS2015. The target variable Diabetes_012 has 3 classes. 
- 0 : no diabetes or only during pregnancy.
- 1 : prediabetes.
- 2 : diabetes.

This dataset has 21 feature variables. Features are described clearly with variable explanation available at https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators and https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_calculated_variables_version4.pdf

**Classifiers**

The following predictive modeling techniques will be evaluated and compared to determine the most effectitve approach for addressing the problem. 
- Logis c Regression 
- Random Forest 
- XGBoost 

 **Evaluation Metrics**
 
 To assess model performance objectively, the following evaluation metrics will be considered:
* Precision
* Recall
* Accuracy
* F1 score
* AUC Score

# Summary 

A diabetes status classification system was developed using the CDC BRFSS2015 dataset, which contains 253,680 entries across 22 columns. Initial data exploration highlighted a significant class imbalance in the Diabetes_012 variable. To address this imbalance, resampling methods like SMOTE (Synthetic Minority Over-sampling Technique) were applied.
To tackle the problem of risk assessment and prediction of at-risk and diabetic groups, a logistic regression baseline model was built and compared to ensemble models, including Random Forest and XGBoost . Hyperparameter tuning was conducted on the selected XGBoost using ParamGridBuilder to optimize model performance. The final model's precision score improved to 59%, compared to a baseline of 56%.

The SHAP (SHapley Additive exPlanations) framework was employed to enhance the model's interpretability. SHAP fidelity was evaluated by comparing the model's predictions to SHAP predictions across the full dataset and a subset of 100 samples. The mean absolute error for SHAP fidelity was close to 0, demonstrating that the SHAP explanations were highly faithful to the model's predictions. Additionally, SHAP was used to analyze feature importance and explore feature interactions through dependence plots, providing further insights into the relationships between key predictors.

## Exploratory Data Analysis

- Examine and sanitize the data
- Handle missing data appropriately
- Investigate data attributes
- Rectify typos, inconsistencies in capitalization, and naming conventions
- Analyze the distribution of variables

## Explore and examine the data 
Examine the data structure.

**Summary of data structure**
* 253680 entries, total 22 columns
* dtype: int64
* No missing data

* **Columns**

    * Diabetes_012*            
    * HighBP                  
    * HighChol                
    * CholCheck               
    * BMI                     
    * Smoker                  
    * Stroke                  
    * HeartDiseaseorAttack    
    * PhysActivity            
    * Fruits                  
    * Veggies                 
    * HvyAlcoholConsump       
    * AnyHealthcare           
    * NoDocbcCost             
    * GenHlth                 
    * MentHlth                
    * PhysHlth                
    * DiffWalk                
    * Sex                     
    * Age                     
    * Education               
    * Income                  


# To ensure a robust response to the business query and deliver practical recommendations:

- Prioritize analysis of patient behavior and their impact on disease 
- Delve into the data features for predictive patterns that can inform preventataive intervention and disease treatment management .
- Use a holistic  approach to establish the correlations between  predictors and onset of diabetes.

# Variables Distribution Analysis

Descriptive Statistics
- Target variable Diabetes_012 has a mean of 29% indicating possible class imbalance
- Health indicators such as HighBP(42%) and HighChol(42%) suggest prevalence of high risk factors
- Lifestyle variables such as Smoker (44%) indicating increase risk for blood glucose condition.  

**Demographic distribution**
    
<<<<<<< HEAD
=======
![image](https://github.com/user-attachments/assets/13047abe-727e-4c37-a445-99147759d58b)
>>>>>>> d4d443ff0c2f096c35cc70bd7965fa8556de6f09

## Demographic distribution
- Age: Age ranges from 18-80 years with he highest frequency appears around the central bins range 8 - 10, highest peak reflecting a strong concentration of individuals 55-69 years old.
- Gender: There is a  slight gender imbalance in datasets with women outnumbering men 56% to 44%
- Income: The majority of the data  is concentrated between 6 and 8 on the income scale above $35000 showing that most individuals fall into higher income brackets.

**Health indicators  distribution**

# BMI Distribution
![image](https://github.com/user-attachments/assets/56e3d636-2194-4e64-aa5c-a23b0892d7e4)


## Health indicators distribution
- BMI:The variable's right-skewed nature suggests potential outliers, most BMI falls bewteen 20-40. 
- Smoker:   there are more Non-Smokers than Smokers in the dataset
- Physical Activities: 
    - A majority of individuals are "Active" 
    - females( 0.0)  are more physically active overall

## Diabetes  distribution**

<<<<<<< HEAD
![alt text](image.png)
    
=======
![image](https://github.com/user-attachments/assets/0937f903-1189-4e68-bc24-7d8e2feea80e)

>>>>>>> d4d443ff0c2f096c35cc70bd7965fa8556de6f09
**Diabetes distribution analysis**
The dataset reveals a significant class imbalance in the Diabetes_012 variable:
- **213703 instanced labeled "0"** (non-diabetic)  dominate the data
- **4631 instances labeled as "1"** (pre-diabetes) which is significantly lower
- **35346 instances labeled as "2"** ( diabetes) forms a meaningful part of the data with diabtes rate 0f 13.93% while prediabetes rate is 1.82%  

**Health  Implication**
- Significant number of patients are actively managing the disease
- There’s an opportunity for preventive measures that could slow or halt progression to diabetes for the at-risk group (prediabetes)
- Mitigation efforts such as education, health care monitoring and lifestyle interventions to reduce the number of patients likely to develop diabetes in the future.

**Modeling Implication**
- **Class imbalance impact**: Daibetes varaible  imbalance can adversely affetct  the performance of machine learning models
specifically  classifiers which are are sensitive to class distribution.

- **Solution**: Due to the importance of risk assessment and prediction of at-risk and diabetic groups, to address the imbalance issues resampling method such as SMOTE (Synthetic Minority Over-sampling Technique) will be considered.


# Education and Diabetes Status Distribution
This table showcases the distribution of diabetes status (`Diabetes_012`) across various education levels. Percentages represent the proportion of individuals in each category.

## Table: Education vs. Diabetes Status
| **Education Level** | **Diabetes Status** | **Percentage (%)** |
|----------------------|---------------------|--------------------:|
| 1.0                 | 0.0                 | 71.84              |
|                      | 1.0                 | 1.15               |
|                      | 2.0                 | 27.01              |
| 2.0                 | 0.0                 | 66.76              |
|                      | 1.0                 | 3.98               |
|                      | 2.0                 | 29.26              |
| 3.0                 | 0.0                 | 72.46              |
|                      | 1.0                 | 3.31               |
|                      | 2.0                 | 24.22              |
| 4.0                 | 0.0                 | 80.21              |
|                      | 1.0                 | 2.15               |
|                      | 2.0                 | 17.64              |
| 5.0                 | 0.0                 | 83.28              |
|                      | 1.0                 | 1.91               |
|                      | 2.0                 | 14.81              |
| 6.0                 | 0.0                 | 88.94              |
|                      | 1.0                 | 1.37               |
|                      | 2.0                 | 9.69               |

## Key
- **Education Level**:
  - 1.0: Did not graduate high school
  - 2.0: High school graduate
  - 3.0: Some college
  - 4.0: College graduate
  - 5.0: Postgraduate education
  - 6.0: Advanced degree
- **Diabetes Status (`Diabetes_012`)**:
  - 0.0: Non-Diabetic
  - 1.0: Pre-Diabetic
  - 2.0: Diabetic

### **Patient Education Levels Diabetes Distribution Analysis**
- Non-diabetic patients are the majority across all education levels.
- At education level 1.0, 71.8% are Non-Diabetic, rising to 88.9% at education level 6.0.
- Prediabetes cases are low but consistent across education levels
- The proportion of Diabetics decreases significantly as education increases.
- At education level 1.0, 27.0% are Diabetic, dropping to 9.6% at level 6.0.. 

# Features correlation analysis
![image](https://github.com/user-attachments/assets/24940863-f2b7-4658-98eb-c8484cacc6bb)

## **Features Correlation Analysis**
- Notable Positive Correlations: highBp & HighCchol, age 
- Notable Negative Correlations:genHlth income education
- The correlation values for diabetes seems to fall below 0.3 for all features.

# Preprocessing
 ## Prepare data for modeling
- Feature engineer composite scores
- Create training and testing sets
- Define the predictor and target variables
- Perform a standard train-test split.
- Assign 20% to the test set
- Set random_state ensuring that the split is reproducible

**Feature engineer composite scores**

**Healthy eating score**
The healthy eating score provides insights of the dietary behavior of individuals. 

**Lifestyle score = (HvyAlcoholConsump + Smoker + 'PhysActivity )**
The Lifestyle Score  feature will help to capture holistic view of an individual’s lifestyle  and the relationship to the  health outcome. The lower score indicates a healther lifestyle

**Comorbiditity rating= HighBP HighChol	BMI(Obese) HeartDietseaseorAttack Stroke Age**
The Comorbidity Rating will provide a  comprehensive method  to quantify the impact of multiple health conditions on  diabestes risk assesment and prediction . The **lower score** indicates a lower risk of diabetes and prediabetes

**Healthcare Access and Engagement Score**
Healthcare Access and Engagement Score composite variable will provide a broader perspective on an individual’s access to and ability to afford healthcare as well as utilization. Higher score means more access and greater engagement. 

**OverallHealth_Score** 'GenHlth', 'MenHlth''PhysHlth'DiffWalk

Combine four individual health indictors and provide an aggegate of the individuals health. The score provides a comprehensive summary of an individual’s overall health, categorized on a scale of 1 to 4, where **higher scores** indicate **poorer health** and lower scores represent better health.

### ***Target variable Description***: 

- Both Diabetes and Pre-Diabetes are grouped into one category (Diabetes_status) indicating blood glucose conditions, with Non-Diabetes as the second category.
- Early Detection Focus: This grouping  useful for  flagging  anyone at risk of diabetes, including those in the pre-diabetic stage.
- Preventive Action: This approach supports early intervention because it treats both diabetic and pre-diabetic as high-risk groups, prompting similar preventive measures.

# Modeling
* Divide the data using random split into train_data 80%  and test_data 20%  
* Build the classification models using Logistic Regression, Random Forest and XGBoost algorithms and  default params on the training data
* Make predictions with  baseline  model 

## Hyperparameter tuning
* Determine the best parameters for the business task by applying a range of parameters. 
* Employ ParamGridBuilder  with CrossValidator to optimize model performance

# Analysis of models perfomances and final model selection

**Model  Comparison**

| Model         | Precision        |Recall  |  F1-Score  | Accuracy | AUC Score|
|-----------------|---------------------|---------|---------|---------|------|
| **Logistic Regression** |  0.50  | 0.17 |  0.25 | 0.84    |      **0.79**    |
| **Random Forest**| **0.62**          |0.02  | 0.04   | 0.84     |     0.51 |
|  **XGBoost**     | 0.56          | **0.20** |**0.30**     |  0.85   |     0.58   | 
| 
                     
- **Precision**: *Random Forest* has the highest precision (0.62), indicating that it has the lowest rate of false positives among the models.(useful when false positives are costly)
- **Recall**: *XGBoost* has the highest recall (0.20), indicating that it has the highest rate of true positives among the models i.e how well the model identifies actual positives (minimizing false negatives).
- **F1 Score**: **XGBoost*   has the highest F1 score (0.30), which balances precision and recall, indicating the overall performance.
- **Accuracy**: XGBoost has largest accuracy (0.85), indicating the overall correctness of the predictions.
- **AUC Score**: *Logistic Regression* has the highest AUC score (0.78), indicating its ability to distinguish between positive and negative classes effectively based on the ROC curve.

 **Confusion Matrices Model Performance** 

| Model           | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|-----------------|---------------------|----------------------|----------------------|---------------------|
| Logistic Regression | 41444           | **1351**                 | 6603               | 1338                |
| Random Forest   |  **42697**            |7771                  | **98**             |170
| XGBoost         | 41584                | 6339               | 1248                |**1602**                  |

 
# Model Selection and Tuning 

**XGBoost** was chosen for  several reasons based on an analysis of model performances:

- It achieved a high recall, F1 , and accuracy scores among the models evaluated.
- More balanced trade-off between precision and recall, which is important in minimizing  false negatives and maintaining true positives rates.
- Provides the best overall performance for the diabetes dataset striking a balance across all metrics yet achieving higher accuracy.
*  Easy to evaluate variable importance or feature contributions to model predictions 
* Reduce risk of overfitting, enhancing the model's generalizability

**Model hyperparameter tuning**

**Key Metric comparison **

| Metric       | XG   | final_XG    |
|--------------|-------|-------|
| **Precision** | 56% | 59% |
| **Recall**    | 20% | 19% |
| **F1-Score**  | 30% | 29% |
|AUC            |  59  |58   |

**Confusion Matrix Comparison**

|Model   | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|---------|----------------------|----------------------|----------------------|----------------------|
| **XG** | 41584          | 6339              |  1248            | 1602           |
| **Final_XG**  | 41712 | 6,414             | 1083              | 1527             |


- True Negatives (TN):Final_XG correctly identifies more non-diabetic cases
- False Positives(FP): Final_XG incorrectly identifies more non-diabetics as blood sugar conditions   (increasing unnecessary interventions) 
- False Negatives(FN):Final_XG has fewer false negatives(1083 vs. 1248) thus better at identifying blood sugar conditions cases. 
- True Positives (TP): XG has a slight decrease in identifing  blood sugar conditions
- Final_XG is slightly better,and has fewer false negatives.

# SHAP (SHapley Additive exPlanations) 
To address the business problem SHAP technique will be used to provide:
- Quantitative measure of feature importance making it easier to 
    - Identify the most influential features for predictions.
    - Detect irrelevant or redundant features.
- Rich visualization providing intuitive plots to interpret model 

**SHAP Fidelity**
- measures the alignment between the SHAP-based reconstruction of predictions and the actual model predictions
- evaluate the max descrepancy on full data set 
- evalute the max descepancy on subset

**Results** 

| **Dataset**              | **Max Discrepancy**     |
|--------------------------|-------------------------|
| Full Test Set            | .0.0   |
| Subset (100 Samples)     | 0.0      |

### Summary
The observed maximum discrepancies are extremely small, confirming high fidelity. The SHAP values successfully reconstruct the model’s predictions to within floating-point precision limits.

** Fidelity Mean Absolute Error (MAE)**
- The Mean Absolute Error (MAE) =0.0.
- High Fidelity: The SHAP explanations are highly faithful to the model's predictions and confirms that the SHAP values correctly explain the contribution of each feature to the prediction.


# Display feature importance 
    
![image](https://github.com/user-attachments/assets/74c1b425-e161-46d5-b7bd-72a5464a86ce)

    
Beeswarm plot  to display the information-dense summary of how the top features in the  dataset impact the model’s output

![image](https://github.com/user-attachments/assets/e8d2b42b-6ce2-4beb-95c2-917cdb6c3c31)

  
## Analysis
Feature Importance Order: Comorbid_rating, OverallHealth_Score, and BMI are the top contributors to the model's predictions

Feature Contributions:

The horizontal position (SHAP value) indicates whether the feature increases or decreases the model prediction .
Positive SHAP values push predictions higher risk of diabetes staus, while negative SHAP values push predictions lower . The colors indicate the magnitude of the prediction red indicates a high feature value while blue a low feature value. 

**High importance features**
- ***Comorbid_rating***:High values (red) tend to increase the prediction (positive SHAP values) while low comorbid_rating (blue) tend to decrease the prediction (negative SHAP values).
- **OverallHealth_Score***:High health scores (red) are associated with higher predictions of diabetes status.
Low health scores (blue) reduce predictions.
- **BMI**: High BMI (red) generally increases predictions, while low BMI (blue) reduces predictions.

**Moderate importance Features**
HighBP and Income has mixed SHAP distributions showing that these features have variable effects on predictions depending on their values. For example at lower incomes (red)increases predictions.

**Lower Importance Features**:
Lifestyle_score, Health_AccessEngagement_Score, and HeartDiseaseorAttack have smaller average SHAP values, indicating less overall contribution to the model predictions.


## Examine Feature Interactions

 SHAP dependence plots to explore interactions between high importance,moderate importance  and low importance features. 

![image](https://github.com/user-attachments/assets/490b06ab-c2f9-4f7b-afb2-25ca464d0d44)

![image](https://github.com/user-attachments/assets/960b4d11-cd7e-4935-9199-e9ff9245e8ae)

![image](https://github.com/user-attachments/assets/2dca7dac-22c1-4d35-8e90-3b9bb379e994)

#  Key Findings from SHAP Analysis

## **Feature Importance Order**
- High Importance: Comorbid_rating, OverallHealth_Score, and BMI are the top contributors to predictions.
- Moderate Importance: HighBP and Income have variable effects based on feature values.
- Lower Importance: Lifestyle_score, Health_AccessEngagement_Score, and HeartDiseaseorAttack contribute less.

## **High Importance Features**

Comorbid_rating
- High values increase risk (positive SHAP values), while low values reduce risk (negative SHAP values).
- Interactions: High Comorbid_rating with high-risk features amplifies risk (pink). Low-risk features (blue) offset severity.

OverallHealth_Score
- High scores indicate worse health and increase risk, while low scores reduce risk.
- Interactions: High comorbid values (pink) increase risk. Low comorbid values (blue) mitigate risk.

BMI
- High BMI increases predictions, while low BMI reduces risk.
- Interactions: Non-linear; risk increases up to a threshold and declines. High BMI with high glucose amplifies risk (pink), while low BMI with low glucose reduces risk (blue).

## **Moderate Importance Features**

HighBP
- Presence increases risk, absence reduces it.
- Interactions: Risk is moderated by comorbid features.

Income
- Higher income reduces risk, while lower income increases it.
- Interactions: Low income amplifies risk when combined with poor health.

HighChol
- Presence increases risk; absence reduces it.

### **Lower Importance Features**

Sex
- Females have reduced risk (negative SHAP values). Males have increased risk (positive SHAP values).

MentHlth
- Poor mental health reduces risk (negative SHAP values).Interactions: Higher mental health issues amplify effects.

Education
- Lower education increases risk: higher education reduces it. 
- Interactions: Low education with high-risk features amplifies risk .

Healthy_Eating_Score
- Poor eating habits increase risk; healthy eating reduces risk.
- Interactions: Poor eating habits amplify risk, while healthy habits are protective .

Lifestyle_score
- Poor lifestyles increase risk; better lifestyles reduce it.
- Interactions: Poor lifestyle combined with high-risk features  amplifies risk.

Health_AccessEngagement_Score
- Poor healthcare access increases risk: better access has minimal impact.
- Interactions: Poor access with high-risk features (pink) amplifies risk.

## Recommended Interventions

Based on the above analysis the following predictive recommendations are proposed:

1.	Target High-Risk Groups: Focus on individuals with high Comorbid_rating, poor health, low income, and unhealthy lifestyles.
2.	Leverage Protective Factors: Promote healthy eating, higher education, better healthcare access, and income growth.
3.	Address Interactions: Target scenarios where multiple high-risk features compound effects to implement tailored interventions.

# Next Steps 

1.	Validate Findings:
- Conduct robustness checks by applying SHAP analysis to new datasets to ensure the consistency of insights.
- Compare the SHAP feature importance rankings with other feature importance metrics like permutation importance
- Perform subgroup analysis to examine feature contributions across different demographic groups.

2.	Feature Engineering:
- Explore additional composite variables to capture nuanced relationships.
- Investigate temporal patterns in the CDC health data.

3.	Enhance Model Interpretability:
- To confirm key drivers, compare SHAP results across different models (e.g., Random Forest, Gradient Boosting) 
- Simplify models by pruning low-importance features identified in SHAP analysis.

4.	Cluster-Based Analysis:
- To identify distinct risk profiles group individuals based on feature similarity (e.g., clustering by OverallHealth_Score and Lifestyle_score) . 
- Tailor interventions and predictive models for each cluster.

For More Information:

See the full analysis in the Jupyter Notebook or review the presentation.


Project_directory

**data**

**images**

**.gitignore**

**MVP.ipynb**

**presentation.pdf**

**readme.md**
 

