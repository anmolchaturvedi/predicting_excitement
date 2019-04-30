## Donors Choose: Predicting Donor Excitement for Classroom Projects

### Goal: 
Predict whether projects listed on Donors Choose will reach full funding.

### Methods:
I performed exploratory analysis and predictive modeling utilizing Python pandas and scik-itlearn on 350,000+ records of [Donors Choose](https://www.donorschoose.org/) data, including geographical features, school type (i.e., charter, magnet, etc.), teacher program (i.e., Teach for America), project topic (i.e., books, technology, trips), requested funding amount, and funding match eligibility.

I trained multiple Naive Bayes, Logistic Regression, K-Nearest Neightbors, Bagging, and AdaBoost classifiers each tuned to a different set of hyperparameters, and evaluated them based on AUC-ROC performance to select the best model.


### Files Included
- [`donors_choose.ipynb`](https://github.com/lorenh516/predicting_excitement/blob/master/donors_choose.ipynb): My analysis, model creation, and model evaluation
- [`ml_explore.py`](https://github.com/lorenh516/predicting_excitement/blob/master/ml_explore.py): Helper functions for data exploration
- [`ml_pipeline_lch.py`](https://github.com/lorenh516/predicting_excitement/blob/master/ml_pipeline_lch.py): Helper functinos for data cleaning and preprocessing
- [`magiclooping.py`](https://github.com/lorenh516/predicting_excitement/blob/master/magiclooping.py) and [`ml_modeling.py`](https://github.com/lorenh516/predicting_excitement/blob/master/ml_modeling.py): Helper functions for model creation and evaluation
