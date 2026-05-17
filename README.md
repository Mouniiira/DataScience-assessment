# Climate Analysis & Air Quality Predictive Modeling Framework

This repository contains a comprehensive data mining, exploratory data analysis (EDA), and machine learning pipeline developed as part of a **Technical Assessment Test for a Data Analyst Internship**.

The primary objective of this project is to model complex environmental interactions and build an optimal regression framework to forecast ambient temperature profiles (`temperature_celsius`) while ensuring strict chronological validation and transparency.

---

## Project Architecture & Methodology

The pipeline follows a structured, production-grade machine learning workflow designed to prevent common data science pitfalls like data leakage and overfitting:

### 1. Preprocessing & Chronological Alignment
* **Time-Series Integrity:** The dataset is explicitly sorted by the `last_updated` timestamp to preserve real-world sequential order.
* **Chronological Data Split:** To eliminate future-data leakage, data is separated using a strict sequential split (**80% Training, 20% Testing**) instead of a randomized shuffle split.
* **Sensor Data Imputation:** Corrupted data placeholders (`-9999.0`) within the air quality indicators are dynamically masked and imputed using median-based baseline profiles.

### 2. Feature Engineering
* **Temporal Disaggregation:** Extracted high-resolution cyclical drivers (`hour`, `day`, `month`, `day_of_week`) from raw datetime values.
* **Micro-Climate Interactions:** Engineered human-centric comfort indexes (`temp_comfort_diff`) mapping the delta between apparent and raw temperatures, as well as additive pollutant indexes (`total_particulates`).

### 3. Multivariate Anomaly Detection
* Deployed an unsupervised **Isolation Forest** anomaly detection routine across core climate vectors (`wind_kph`, `humidity`, `uv_index`, etc.) to map out operational spikes and isolate sensor equipment noise.

### 4. Advanced Modeling & Ensembling
Evaluated five distinct forecasting architectures grouped into a clear performance hierarchy:
* **Baseline:** Regularized Linear Model (`Ridge Regression`).
* **Individual Tree-Based Ensembles:** `RandomForestRegressor` and `GradientBoostingRegressor`.
* **Advanced Ensembles:** An unweighted `VotingRegressor` and a meta-learned `StackingRegressor` (utilizing a `LinearRegression` final estimator over a 5-fold cross-validated training architecture).

---

## Model Performance Leaderboard

The predictive models were evaluated on the unseen testing partition using Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), and the Coefficient of Determination (**R² Variance Explained**):

| Model Architecture | MAE | RMSE | R² (Variance Explained) |
| :--- | :---: | :---: | :---: |
| 1 **Ensemble: Stacking Regressor** | **0.024752** | **0.035966** | **86.97%** |
| 2 **Individual: Random Forest** | 0.025804 | 0.037665 | 85.70% |
| 3 **Individual: Gradient Boosting** | 0.027842 | 0.040144 | 83.76% |
| 4 **Ensemble: Voting Regressor** | 0.032824 | 0.045952 | 78.72% |
| 5 **Individual: Ridge Regression (Baseline)** | 0.058093 | 0.075361 | 42.77% |

### Key Modeling Insights
* **The Stacking Advantage:** The Stacking Regressor emerged as the champion framework, capturing **86.97%** of the true target variance. By training a linear meta-learner, it intelligently balanced the strengths of the tree architectures and neutralized the weak linear baseline.
* **Voting Performance Drag:** The simple Voting Regressor underperformed relative to standalone trees. Because it uses an unweighted average, the poor performance of the Ridge baseline artificially dragged down the predictive power of the tree estimators.

---

## Core Insights & Diagnostic Results

### 1. Environmental Impact: Weather & Air Quality
* **Photochemical Ozone Synthesis:** Correlation matrix analysis (`sns.heatmap`) confirmed a strong positive relationship between `temperature_celsius` and ground-level `air_quality_Ozone (O3)`, proving that solar/thermal spikes act as catalysts for chemical pollutants.
* **The Dispersion Ventilation Effect:** Bivariate joint regression tracking (`sns.jointplot`) isolated a distinct negative relationship between wind speed (`wind_kph`) and fine particulate matter (`air_quality_PM2.5`). Higher wind speeds generate mechanical turbulence that breaks up stagnant boundary layers, actively venting away localized pollution concentrations.

### 2. Feature Importance & Model Explainability
* **Model-Specific (Gradient Boosting MDI):** Mean Decrease in Impurity audits show that the model prioritizes raw thermodynamic properties and engineered indices (like `temp_comfort_diff`) to optimize decision tree cuts quickly.
* **Model-Agnostic (Permutation Importance):** Running a 10-repeat permutation test on the Stacking Regressor identified **Geographical Coordinates (Latitude/Longitude)** and **Barometric Pressure** as the most stable, structural anchors of the pipeline. Shuffling these features caused the steepest drops in test accuracy, proving the model depends on physical weather mechanics rather than random noise.

### 3. Spatial Patterns & Thermal Gradients
* **Sampling Distribution:** Spatial mapping revealed distinct horizontal track clustering, matching the physical arrangement of target environmental collection stations.
* **Climate Gradient:** Continuous vector mapping exposed a predictable thermal gradient transitioning smoothly from cold blues to warm reds across coordinate lines, explaining why geographic features were heavily favored during explainability testing.

---

## How to Run the Notebook

### Prerequisites
Ensure you have Python 3.10+ installed along with the following packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Execution
1. Clone this repository:
   ```bash
   git clone https://github.com/Mouniiira/DataScience-assessment.git
   ```
2. Download the dataset file
https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code
3. Place your raw dataset file in the project directory.
4. Launch Jupyter Notebook and run all cells in the project notebook.

---
*This project was completed as an internship assessment task to demonstrate end-to-end data analyst competencies, including rigorous statistical modeling, data cleaning, pipeline engineering, and technical communication.*
