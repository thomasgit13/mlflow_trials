#### pip install mlflow

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

def get_metrics(y_true, y_pred, y_pred_prob):
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 2), 
        'precision': round(precision_score(y_true, y_pred), 2), 
        'recall': round(recall_score(y_true, y_pred), 2), 
        'entropy': round(log_loss(y_true, y_pred_prob), 2)
        }
def create_roc_auc_plot(clf, X_data, y_data):
    metrics.plot_roc_curve(clf, X_data, y_data) 
    plt.savefig('images/roc_auc_curve.png')

final_data = preprocessing(data)
X = final_data.loc[:, final_data.columns != 'y']
y = final_data.loc[:, final_data.columns == 'y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,stratify = y, random_state=47)


model = RandomForestClassifier(
    n_estimators=140,
    max_depth=5,
    min_samples_split=152,
    criterion='gini',
    class_weight={0:0.9,1:0.3},
    random_state=101
)

model.fit(X_train, y_train)
y_pred= model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

model_metrics = get_metrics(y_test, y_pred, y_pred_prob)
params = model.get_params()
logged_param_names = [
    'class_weight',
    'max_depth',
    'min_samples_split',
    'n_estimators',
    'random_state'
]

params ={i:params[i] for i in logged_param_names}
create_roc_auc_plot(model, X_test, y_test)

############### mlflow tracking section ####################

experiment_name = 'march_11_experiments'
model_name = 'random_forest_model_without_oversampling'
run_name = 'first_run'
custom_tags={
    'model_building':'Thomaskutty Reji',
    'data_extraction_contributor':'Sarah',
    'leads_generation_script':'Maria'
    }

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name=run_name):
    ### logging parameters,metrics,model,tags
    mlflow.log_params(params)
    mlflow.log_metrics(model_metrics)
    mlflow.sklearn.log_model(model,model_name)
    mlflow.set_tags(custom_tags)

## setting backend uri (if we want model registry functionality)
## tracking_uri is optional 

# mysql mlflow database server start command 
# create database mlflow_tracking_database; 
# mlflow server --backend-store-uri mysql+pymysql://root:mysqlpass@localhost/mlflow_tracking_database  -h 127.0.0.1 -p 5000
# mlflow server --backend-store-uri sqlite:///mlflow.db  --host 127.0.0.1 -p 5000



########## Adding model to the registery ##################3
# method 1 : mlflow.sklearn.log_model(model, "model",registered_model_name="iris-classifier")
# method 2 : 
# import mlflow
# with mlflow.start_run(run_name=run_name) as run:
#     result = mlflow.register_model(
#         "runs:/dff923c9e0924e8e968eaed4cab33ee9/model",
#         "iris-classifier-2"
#     )

# method 3 : Registering an empty model 
# import mlflow
# client = mlflow.tracking.MlflowClient()
# client.create_registered_model("basic-classifier-method-3")

