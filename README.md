Car Dekho - Used Car Price Prediction
Overview :
The Car Dekho - Used Car Price Prediction project aims to enhance the customer experience and streamline the pricing process for used cars. This machine learning model predicts the prices of used cars based on various features, such as make, model, year, fuel type, transmission, and other relevant attributes. The model is integrated into a Streamlit-based web application, allowing users (both customers and sales representatives) to input car details and receive instant price predictions.

Project Highlights :
1. Data Cleaning & Preprocessing: Handling missing values, standardizing data formats, encoding categorical variables, normalizing numerical features, and removing outliers.

2. Exploratory Data Analysis (EDA): Analyzing and visualizing data to uncover patterns, correlations, and insights.

3. Machine Learning Model Development: Training multiple models including Linear Regression, Decision Trees, Random Forest, and Gradient Boosting to predict car prices.

4. Model Evaluation & Optimization: Evaluating model performance using metrics like MAE, MSE, R-squared, and optimizing the model with techniques such as cross-validation, Grid Search, and Random Search.

5. Streamlit Deployment: Deploying the model as an interactive web application that allows real-time price predictions based on user input.

6. Feature Engineering & Regularization: Creating new features and applying regularization to prevent overfitting.

Domain :
Automotive Industry
Data Science
Machine Learning

Problem Statement :
As a data scientist at Car Dekho, your goal is to enhance the pricing process by creating an accurate and user-friendly tool that predicts the prices of used cars. The model will be deployed in an interactive web application, enabling customers and sales representatives to receive price estimates based on various car features.

Approach :
1. Data Processing:
Import unstructured datasets from multiple cities, convert them into a structured format, and add a city column.
Handle missing values using imputation techniques (mean, median, mode).
Standardize data formats and perform encoding for categorical variables.
Normalize numerical features using Min-Max scaling or standard scaling.
Identify and remove outliers using IQR or Z-score methods.

3. Exploratory Data Analysis (EDA):
Calculate summary statistics to understand data distribution.
Create visualizations (scatter plots, box plots, heatmaps) to uncover patterns and correlations.
Perform feature selection using techniques like correlation analysis and model-based feature importance.

3. Model Development:
Split data into training and testing sets (70-30 or 80-20).
Train models like Linear Regression, Decision Trees, Random Forest, and Gradient Boosting.
Optimize models using hyperparameter tuning techniques (Grid Search, Random Search).

4. Model Evaluation & Optimization:
Evaluate models using metrics such as MAE, MSE, and R-squared.
Compare models and select the best performer.
Apply feature engineering and regularization (Lasso, Ridge) to improve model performance.

5. Deployment:
Deploy the trained model using Streamlit.
Create an interactive user interface where users can input car features and receive real-time price predictions.

Results :
A functional and accurate machine learning model for predicting used car prices.
Comprehensive EDA with visualizations and insights into the dataset.
Detailed documentation covering the methodology, model evaluation, and results.
An interactive Streamlit web application for easy, real-time price predictions.

Requirements :
Python 3.x
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
