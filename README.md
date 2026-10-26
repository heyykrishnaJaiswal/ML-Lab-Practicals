# ML-Lab-Practicals
Machine learning practicals on heart disease prediction using Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Descent and Ensemble methods in Python (scikit-learn).

# ML Practicals - Heart Disease Prediction

These are my machine learning practicals for final year. I used the same heart disease dataset in all of them and tried a different algorithm each time to see how they compare.

Krishna Jaiswal | Roll No. 21 | Section C

## Dataset

I used `heart.csv` (1025 rows, 14 columns). There are no missing values. The goal is to predict whether a person has heart disease (`target` = 1) or not (`target` = 0) from 13 features like age, sex, chest pain type, cholesterol, blood pressure, max heart rate and so on.

## What's inside

| No. | Practical | Accuracy |
|-----|-----------|----------|
| 1 | Logistic Regression | 80.3% |
| 2 | KNN | 73.2% |
| 3 | SVM | 68.3% |
| 4 | Decision Tree | 98.5% (79.0% after pruning) |
| 5 | Random Forest | 98.5% |
| 6 | Gradient Descent | - |
| 7 | Ensemble techniques | best is Voting, 80.3% |

Short summary of each one:

1. **Logistic Regression** - cleaned the data, removed duplicates, split 80/20 and trained the model. Also plotted a confusion matrix.
2. **KNN** - KNeighborsClassifier with the default k = 5.
3. **SVM** - SVC with default settings.
4. **Decision Tree** - tried a normal tree, one with `max_depth=12`, and one pruned with `ccp_alpha`. Plotted all the trees.
5. **Random Forest** - RandomForestClassifier with 100 trees.
6. **Gradient Descent** - a small example that fits a straight line by hand using gradient descent and plots the error going down.
7. **Ensemble techniques** - compared Bagging, Random Forest, AdaBoost, Gradient Boosting and a Voting classifier using accuracy, precision, recall, F1 and ROC-AUC.

Results from practical 7:

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Bagging | 0.770 | 0.794 | 0.837 |
| Random Forest | 0.754 | 0.776 | 0.861 |
| AdaBoost | 0.738 | 0.742 | 0.869 |
| Gradient Boosting | 0.721 | 0.746 | 0.826 |
| Voting | 0.803 | 0.824 | 0.878 |

## Steps followed in each notebook

1. Import libraries
2. Load the data and look at it (head, info, describe)
3. Check for missing values
4. Split into X and y
5. Train/test split (80/20, random_state=42)
6. Train the model and check accuracy

## A few things I noticed

- The dataset has a lot of duplicate rows. In practicals 1 and 7 I removed them, but in 2 to 5 I didn't, so the very high 98.5% for Decision Tree and Random Forest is probably because the same rows are in both train and test.
- I didn't scale the features for KNN and SVM, which is probably why they scored lower.

## Running it

```
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Keep `heart.csv` in the same folder as the notebooks. Some notebooks have a `os.chdir(...)` line with my own folder path, so change or delete that line.

Built with Python, pandas, NumPy, scikit-learn, matplotlib and seaborn.
