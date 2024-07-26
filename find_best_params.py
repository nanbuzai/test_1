from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import pandas as pd
def train(df, test_param):
    fixed_params = {
    'learning_rate': 0.1,
    'objective': 'binary',  # 目标函数
        }
    x_train, x_test , y_train, y_test =  train_test_split(df.drop(columns = ['label']),df['label'],random_state=42)

    lgb =  LGBMClassifier(**fixed_params)
    gsearch1 = GridSearchCV(lgb,param_grid = test_param, scoring='roc_auc',cv=5,n_jobs=-1)
    gsearch1.fit(x_train,y_train)
    param_columns = list(map(lambda x: 'param_'+x,test_param.keys()))
    param_columns.append('mean_test_score')
    return pd.DataFrame(gsearch1.cv_results_)[param_columns], gsearch1.best_params_
