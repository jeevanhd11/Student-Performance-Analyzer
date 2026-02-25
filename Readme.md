# Student Performance Analyzer

## Phase 1: Dataset Understanding

### Dataset Overview
- Total Rows: 395
- Total Columns: 33
- Target Variable: G3 (Final Grade)

### Data Types
- Integer Columns: 16
- Categorical (String) Columns: 17

### Key Observations
- No missing values detected.
- Dataset contains both academic and personal attributes.
- Final grade (G3) will be used for prediction.

## Phase 3: Data Visualization

- Visualized grade distribution
- Analyzed study time impact on final grades
- Generated correlation heatmap to identify key predictors

## Phase 4: Machine Learning Model

- Encoded categorical features
- Split dataset into training and testing sets
- Trained Linear Regression model
- Evaluated using MSE and R² Score

### Project Goal
To analyze student performance and build a machine learning model to predict final grades based on various factors.


### Models Implemented

#### Linear Regression
Used as a baseline regression model.

Performance:
- Mean Squared Error: **5.03**
- R² Score: **0.75**

---

#### Random Forest Regressor
Used to capture nonlinear relationships between features.

Performance:
- Mean Squared Error: **3.48**
- R² Score: **0.83**

Random Forest showed improved prediction accuracy compared to Linear Regression.

---

## Key Insights
Model analysis revealed that the following factors strongly influence student performance:

- Previous academic scores
- School support
- Family relationship quality
- Study behavior

---

## Technologies Used

Programming Language:
- Python

Libraries:
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Tools:
- Git
- GitHub
- VS Code

---

## Project Structure


---

## Future Improvements
- Student risk prediction using classification models.
- Deployment as a web application.
- Real-time academic performance dashboard.

---

## Author
Jeevan H. D.

GitHub:
https://github.com/jeevanhd11