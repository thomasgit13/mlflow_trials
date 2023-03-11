

import pandas as pd
import numpy as np 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics

import mlflow

warnings.filterwarnings('ignore')
data_path = 'data/banking.csv'
data = pd.read_csv(data_path)
data = data.dropna()

def preprocessing(data):
    data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
    data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
    data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])
    cat_vars=[
        'job','marital','education','default','housing',
        'loan','contact','month','day_of_week','poutcome'
        ]
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(data[var], prefix=var)
        data1=data.join(cat_list)
        data=data1
    data_vars=data.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]
    final_data=data[to_keep]
    final_data.columns = final_data.columns.str.replace('.','_')
    final_data.columns = final_data.columns.str.replace(' ','_')
    return final_data

final_data = preprocessing(data)
data = final_data.loc[:, final_data.columns != 'y']

import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
logged_model = 'runs:/d19fe3bdea3542a1a35ed5d058ca7985/random_forest_model_without_oversampling'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
# Predict on a Pandas DataFrame.
import pandas as pd
print(loaded_model.predict(pd.DataFrame(data)))




