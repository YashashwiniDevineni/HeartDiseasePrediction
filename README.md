````markdown
# Heart Disease Prediction using Decision Tree and Random Forest

Heart disease is one of the major health concerns worldwide. People are collapsing suddenly regardless of age, fitness level, or lifestyle habits. So, we thought — why not build a system that can predict heart disease early using machine learning?

In this project, we developed models using **Decision Tree** and **Random Forest** algorithms to predict the presence of heart disease based on medical attributes. Our goal was to create something simple, interpretable, and clinically useful, especially for early diagnosis and preventive care.

---

## Team Members

- **Yashashwini Devineni**
- **Sai Venkata Anil Thota**

---

## Objectives

- Use classification algorithms (Decision Tree, Random Forest) for heart disease prediction.
- Clean, preprocess, and analyze the Cleveland Heart Disease dataset.
- Compare unpruned and pruned Decision Trees with Random Forest.
- Identify key features contributing to the prediction.

---

## Dataset Info

- **Source**: [UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Total Samples**: 303 patient records
- **Attributes**: 14 features including:
  - Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Max Heart Rate, etc.
- **Target Variable**: 0 (No Disease) / 1 (Heart Disease Present)

---

## Tech Stack

- **Language**: Python 3.x
- **Platform**: Jupyter Notebook / Google Colab
- **Libraries**:
  - `pandas`, `numpy` – data handling
  - `matplotlib`, `seaborn` – data visualization
  - `scikit-learn` – model development and evaluation

---

## Setup Instructions

### Google Colab (Easy Way)
1. Open the notebook [`HDP_implementationfinal-2.ipynb`](./HDP_implementationfinal-2.ipynb) in Google Colab.
2. Just run the cells one by one — no setup needed.

### Local Machine (Advanced)
```bash
# Clone the repo
git clone https://github.com/YashashwiniDevineni/HeartDiseasePrediction.git
cd HeartDiseasePrediction

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

---

## Implementation Steps

1. **Preprocessing**
   - Checked for missing values and handled them
   - Converted categorical columns to numerical
   - Normalized continuous features

2. **EDA (Exploratory Data Analysis)**
   - Analyzed distributions and correlations
   - Visualized trends and outliers

3. **Model Building**
   - Trained an **Unpruned Decision Tree**
   - Tuned and trained a **Pruned Decision Tree** using cost complexity pruning
   - Trained a **Random Forest Classifier** using `RandomizedSearchCV` for hyperparameter tuning

4. **Evaluation Metrics**
   - Used Accuracy, Precision, Recall, F1-Score
   - Plotted Confusion Matrix
   - Analyzed Feature Importances

---

## Results

| Model                  | Accuracy |
|------------------------|----------|
| Unpruned Decision Tree | 75%      |
| Pruned Decision Tree   | 80%      |
| Random Forest          | **88%**  |

- The **Random Forest** gave the best performance overall.
- Key features: **Chest Pain Type**, **Age**, **Cholesterol**, **Resting BP**

---

## Visualizations

- Confusion matrices for all models
- Tree diagrams (pruned vs unpruned)
- Feature importance bar chart from Random Forest

---

## Conclusion

- **Decision Trees** are easy to interpret but can overfit if not pruned.
- **Pruned Decision Tree** generalized better than the raw tree.
- **Random Forest** was the most robust and accurate.
- The models could be helpful in real-time clinical settings for quick screening.

---

## Future Scope

- Try out advanced ensemble methods like **XGBoost** or **Gradient Boosting**
- Build a user-friendly web interface using **Flask** or **Streamlit**
- Use real-world datasets with lifestyle/genetic indicators
- Work with healthcare professionals for actual deployment/testing

---

## References

1. Smith & Lee (2023) – *Machine Learning in Cardiovascular Disease Prediction*
2. Brown & Green (2024) – *Decision Tree Algorithms for Medical Applications*
3. Wang & Chen (2023) – *Predicting Heart Disease with Decision Trees*
4. More references can be found in the full project report.
