# Tech-Moms ML Acceptance Predictor

ML project predicting Tech-Moms program acceptance using data from 1,700+ applications. Found computer access, education level, and employment status strongly influence who gets in! Used Gaussian NB model (41% F1 score) and spotted some fairness concerns (DI of 0.82). Includes cool visualizations and practical recommendations to make the program.

## Project Title

The Tech-Moms program aims to help women transition into technology careers, but program administrators face challenges in making the selection process consistent, fair, and equitable while efficiently allocating limited resources. Current manual evaluation processes may introduce unintended biases and miss important patterns in successful applications.

The objective of this project is to develop a machine learning model to predict Tech-Moms program acceptance by December 2024, identify key factors influencing selection decisions, and provide recommendations to improve program accessibility for diverse applicants.

## Data Dictionary

| Feature | Type | Dataset | Description |
|---------|------|---------|-------------|
| `has_computer` | Binary | Tech-Moms Applications | Whether applicant owns a computer (1 = Yes, 0 = No) |
| `computer_availability` | Categorical | Tech-Moms Applications | Type of computer owned (Laptop PC, Laptop Mac, None) |
| `education_category` | Categorical | Tech-Moms Applications | Highest education level (High School, Bachelor's, Advanced Degree) |
| `employment_simplified` | Categorical | Tech-Moms Applications | Employment status (Full-time, Part-time, Not Employed) |
| `ethnicity_simplified` | Categorical | Tech-Moms Applications | Simplified ethnicity categories |
| `children_category` | Categorical | Tech-Moms Applications | Number of children (None, 1, 2-3, 4+) |
| `application_quarter` | Categorical | Tech-Moms Applications | Quarter when application was submitted (Q1-Q4) |
| `application_month` | Categorical | Tech-Moms Applications | Month when application was submitted |
| `is_weekend` | Binary | Tech-Moms Applications | Whether application was submitted on weekend (1 = Yes, 0 = No) |
| `income_level` | Categorical | Tech-Moms Applications | Income category of applicant |
| `cohort` | Categorical | Tech-Moms Applications | Program cohort applied to |
| `accepted` | Binary | Tech-Moms Applications | Target variable - whether applicant was accepted (1 = Yes, 0 = No) |

Class distribution:
- Accepted: 32.7% (567 applications)
- Not accepted: 67.3% (1,165 applications)

## Executive Summary

### Problem Statement
The Tech-Moms program faces challenges in making consistent and fair selection decisions while maximizing the impact of limited program resources. This project analyzed 1,732 program applications to understand acceptance patterns and develop a predictive model to assist in the selection process.

### Methodology
We implemented a comprehensive data science approach including:
1. **Data Preparation**: Cleaning application data, handling missing values, and engineering temporal features
2. **Exploratory Analysis**: Examining demographic patterns, acceptance rates, and correlations
3. **Model Development**: Testing multiple algorithms with special attention to the class imbalance challenge
4. **Optimization**: Applying dimensionality reduction, threshold tuning, and fairness assessment

### Key Findings
Our analysis revealed several critical insights:

Computer access emerged as the strongest predictor of program acceptance, with owners showing a 34.0% acceptance rate versus 28.7% for non-owners. Educational background significantly influenced selection outcomes, with Bachelor's degree holders achieving a 37.4% acceptance rate compared to 26.5% for high school graduates. Employment status also played a major role, with full-time employed applicants showing higher acceptance rates (37.2%) than unemployed applicants (27.8%).

The optimized Gaussian Naive Bayes model achieved a 41.2% F1 score and identified potential fairness concerns, including a Disparate Impact score of 0.82 for applicants without computer access.

### Conclusion
This project provides Tech-Moms administrators with a data-driven tool to assist in application screening while highlighting important equity considerations. The identified acceptance patterns suggest several potential barriers to program access that administrators should address to ensure the program reaches its intended audience effectively.

### Next Steps
1. **Technology Access Initiative**: Implement a laptop lending program to reduce barriers for applicants without computers
2. **Educational Outreach**: Target recruitment to applicants with high school-only education
3. **Selection Criteria Review**: Ensure selection processes don't unintentionally favor certain demographics
4. **Continuous Improvement**: Expand the model to incorporate program completion and career transition data

## Project Components

This repository contains:

1. **Data Processing Pipeline**: Code for cleaning and preprocessing applications
2. **Exploratory Analysis**: Detailed analysis of application patterns and acceptance rates
3. **Predictive Models**: Implementation of multiple ML algorithms with Gaussian Naive Bayes + PCA emerging as the best performer
4. **Fairness Assessment**: Evaluation of potential biases using standard metrics
5. **Visualization Components**: Charts and graphs illustrating key findings
6. **Deployment System**: Production-ready prediction function for scoring new applicants

## Model Performance

We tested multiple algorithms and optimization strategies:

| Model Approach | F1 Score | ROC AUC |
|----------------|----------|---------|
| Logistic Regression | 35.8% | 0.584 |
| Random Forest | 38.7% | 0.593 |
| Gradient Boosting | 34.9% | 0.581 |
| Gaussian NB | 50.4% | 0.595 |
| Complement NB | 52.0% | 0.601 |
| **PCA + Threshold Optimization** | **41.2%** | **0.594** |

Our final model combines PCA dimensionality reduction with threshold optimization to achieve the best balance of precision and recall.

## Key Feature Importance

Permutation importance analysis identified the most influential features:

1. Computer access (score = 0.0055)
2. Education level - High School (score = 0.0046)
3. Ethnicity factors - American Indian/Alaska Native (score = 0.0035)
4. Computer availability - No laptop (score = 0.0029)
5. Computer availability - Mac laptop (score = 0.0023)

## Repository Structure

```
├── data/
├── code/
└── README.md
```

