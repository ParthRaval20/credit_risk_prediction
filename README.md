## Credit Risk Prediction - End-to-End ML Project

## 📌 Problem Statement
Predict whether a loan applicant is **low risk** (likely to repay) or **high risk** (likely to default), so lenders can take **data-driven credit decisions** and reduce financial loss.

## 🧠 Business Impact
- Better risk estimation helps **reduce default rates** without rejecting too many good customers.
- Even a small improvement in risk prediction on large loan portfolios can significantly impact **profitability** and **capital efficiency**.

## 📊 Dataset
- Custom credit risk dataset: `data/credit_data.csv`
- Includes a target column: `credit_risk`
  - (for example, 1 = good credit, 0 = bad credit; exact encoding depends on your notebook)
- Features include:
  - `age`
  - `amount`
  - `duration`
  - `savings`
  - `employment`
  - (and potentially other socio‑economic / financial indicators)

## ⚙️ Tech Stack
- Python
- Scikit-learn (or compatible ML library for the saved model)
- Pandas
- Joblib
- Jupyter Notebook

## 🚀 ML Pipeline (Conceptual)
1. Data loading and basic cleaning (handled in the notebook).
2. Exploratory Data Analysis (EDA) on credit risk drivers.
3. Feature engineering / selection as needed.
4. Model training and evaluation inside the notebook.
5. Export of the best model to `credit_risk_model.pkl`.
6. **Inference script** (`src/predict.py`) for CLI predictions using the saved model.

> The detailed training steps live in `notebook/credit_risk_prediction.ipynb`, while `src/predict.py` focuses on **robust inference**.

## 📈 Model Usage (Prediction Logic)
- The script:
  - Loads the reference data from `data/credit_data.csv`.
  - Drops the target column `credit_risk` to obtain only features.
  - Takes one row as a **template customer** and overrides key features:
    - `age`, `amount`, `duration`, `savings`, `employment`
  - Passes this single-row `DataFrame` to the trained model’s `predict_proba`.
  - Interprets the probability of the **safe/low-risk** class.
- **Decision rule** (default):
  - If \( P(\text{safe}) \ge 0.7 \) → **Loan Approved**
  - Else → **Loan Rejected (High Risk)**

You can adjust this **threshold** in code to suit different business risk appetites.

## 🔍 Key Insights (Conceptual)
- Longer **duration** and higher **amount** can be associated with increased risk, depending on income and savings.
- Applicants with **lower savings** or more unstable **employment** are typically higher risk.
- A carefully chosen threshold balances **approval rate** vs **default rate**; this can be tuned based on historical performance.

> For full, data-backed insights, open and explore `notebook/credit_risk_prediction.ipynb`.

## 📦 Project Structure

```text
.
├── data
│   └── credit_data.csv
├── notebook
│   └── credit_risk_prediction.ipynb
├── src
│   └── predict.py
├── credit_risk_model.pkl        # (expected model artifact)
└── README.md
```

## 🔧 How to Run (GitHub Ready)

1. **Clone the repository**
   - `git clone <your-repo-url>`
   - `cd credit_risk_prediction`

2. **Create and activate a virtual environment (recommended)**
   - `python -m venv .venv`
   - Windows: `.\.venv\Scripts\activate`
   - macOS / Linux: `source .venv/bin/activate`

3. **Install dependencies**
   - `pip install pandas joblib scikit-learn`
   - Or, if you have `requirements.txt`: `pip install -r requirements.txt`

4. **Ensure required files exist**
   - `data/credit_data.csv` with:
     - Target column: `credit_risk`
     - Feature columns at least: `age`, `amount`, `duration`, `savings`, `employment`
   - `credit_risk_model.pkl` saved from your training notebook and compatible with `predict_proba`.

5. **Run the prediction script**
   - `python -m src.predict`
   - or `python src/predict.py`

This will:
- Load the data and model.
- Build a synthetic customer from the first row of the dataset.
- Predict the repayment probability.
- Print a final decision: **Loan Approved** or **Loan Rejected (High Risk)**.

If any critical error occurs (missing file, wrong column names, model issues), the script:
- Prints a clear error message to **stderr**.
- Exits with a **non-zero status code** (suitable for CI/CD checks).

## 📁 Outputs for Stakeholders

- **Repayment probability** printed to the console, e.g.:

  ```text
  Repayment Probability: 0.8234
  Loan Approved
  ```

- **Trained model artifact**:
  - `credit_risk_model.pkl` (can be loaded into an API or batch scoring pipeline).

- **Notebook insights**:
  - `notebook/credit_risk_prediction.ipynb` for EDA, modeling experiments, and visualizations that can be shared with data science stakeholders.

## 🤝 Extensibility

- Add a dedicated `train.py` script for reproducible training and model versioning.
- Wrap the prediction logic in a REST API (e.g., FastAPI) to serve credit decisions in real time.
- Plug in monitoring for **drift**, **performance**, and **fairness** if deployed in production.

