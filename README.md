## 🫀 Heart Disease Prediction (Logistic Regression)

## 📌 Project Description

This project is a machine learning model that predicts the likelihood of heart disease using 15 medical attributes.
The model applies **Logistic Regression** trained on real-world patient data, producing a binary outcome: **0 (Low Risk)** or **1 (High Risk)** of heart disease.

---

## 🧠 What I Learned

### Logistic Regression
- Logistic Regression is a linear classifier that estimates the probability of a binary outcome using a sigmoid function
- The trained model (`Logistic_regression.pkl`) uses **L2 regularization** (Ridge penalty) with the **lbfgs** solver and `C=1.0`
- The model takes **15 input features** and learns a coefficient for each one; larger absolute coefficients indicate stronger influence on the prediction
- Features such as `FastingBS`, `ChestPainType_ATA`, and `ST_Slope_Up` received the highest coefficients, meaning they are the most predictive of heart disease risk
- Binary classification outputs: **0 = Low Risk**, **1 = High Risk**

### Feature Engineering & Preprocessing
- Categorical columns (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) were converted to numerical form using **one-hot encoding**
- Drop-first encoding was applied so one category per feature becomes the implicit reference (e.g., `Sex_F` is the baseline; only `Sex_M` is kept)
- The final feature set stored in `columns.pkl` contains exactly 15 columns:
  `Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`,
  `Sex_M`, `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA`,
  `RestingECG_Normal`, `RestingECG_ST`, `ExerciseAngina_Y`, `ST_Slope_Flat`, `ST_Slope_Up`

### Feature Scaling
- **StandardScaler** (`scaler.pkl`) standardizes all 15 features so each has mean ≈ 0 and standard deviation ≈ 1 before feeding into the model; this is important for Logistic Regression since it is sensitive to the scale of input features
- A second scaler (`heart_scaler.pkl`) was fitted on only the 5 continuous numeric features (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) with learned means ~53.5, ~132.5, ~244.5, ~136.8, ~0.72 respectively
- Scaling must be applied with the **same fitted scaler** used during training; fitting a new scaler on test/live data would introduce data leakage

### Model Persistence
- After training, the model and preprocessing objects were serialized to `.pkl` files using **joblib**
- Four artifacts are saved: `Logistic_regression.pkl` (model), `scaler.pkl` (full scaler), `heart_scaler.pkl` (numeric-only scaler), and `columns.pkl` (expected feature order)
- At prediction time, `joblib.load()` restores each artifact so the exact same preprocessing pipeline is applied to new inputs

### End-to-End Prediction Pipeline
- Raw user input → create a dict with the right feature keys → build a `DataFrame` → fill missing one-hot columns with 0 → reorder columns to match `columns.pkl` → `scaler.transform()` → `model.predict()` → display result
- Ensuring the column order matches what the model was trained on is critical; mismatched columns silently produce wrong predictions

---

## 🔬 Model Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Solver | lbfgs |
| Penalty | L2 (Ridge) |
| Regularization C | 1.0 |
| Number of Features | 15 |
| Output Classes | 0 (Low Risk), 1 (High Risk) |
| Scaler | StandardScaler |

---

## 📊 Input Features

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Patient age in years |
| RestingBP | Numeric | Resting blood pressure (mm Hg) |
| Cholesterol | Numeric | Serum cholesterol (mg/dL) |
| FastingBS | Binary | Fasting blood sugar > 120 mg/dL (1 = Yes) |
| MaxHR | Numeric | Maximum heart rate achieved |
| Oldpeak | Numeric | ST depression induced by exercise |
| Sex_M | Binary (one-hot) | 1 = Male, 0 = Female |
| ChestPainType_ATA | Binary (one-hot) | Atypical Angina |
| ChestPainType_NAP | Binary (one-hot) | Non-Anginal Pain |
| ChestPainType_TA | Binary (one-hot) | Typical Angina |
| RestingECG_Normal | Binary (one-hot) | Normal resting ECG |
| RestingECG_ST | Binary (one-hot) | ST-T wave abnormality |
| ExerciseAngina_Y | Binary (one-hot) | Exercise-induced angina (Yes) |
| ST_Slope_Flat | Binary (one-hot) | Flat ST slope |
| ST_Slope_Up | Binary (one-hot) | Upsloping ST slope |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Google Colab

