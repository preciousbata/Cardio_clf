# Import Libraries
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import f1_score

# MODEL INSTANTIAION
model_dict = {
    'Logistic Regession': LogisticRegression(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Support Vector Clasifier': SVC(),
    'K-Nearest Neighbour': KNeighborsClassifier(),
    'Catboost': CatBoostClassifier(),
    'XGBoost':XGBClassifier()
}


def model_train(df_train,target):
    """
    This function loops over a dictionary of model instantiation to fit training data

    Parameters
    -----------
    df_train: Training Dataset
    target: Response 
    Scoring: Default evaluation metric is F1-score
    """
    for key, model_instantiation in model_dict.items():
        score = cross_val_score(model_instantiation, df_train, target, n_jobs=-1, cv=3, scoring='f1')
        avg_score= score.mean()
        print(f'{key}: The F1_score is {avg_score}')

    
