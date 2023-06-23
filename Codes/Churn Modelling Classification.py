#!/usr/bin/env python
# coding: utf-8

# # Churn Modelling Classification

# ***
# _**Importing the required libraries & packages**_

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pickle
import ydata_profiling as pf
plt.rcParams['figure.figsize']=20,14
import warnings
warnings.filterwarnings('ignore')


# _**Changing The Default Working Directory Path & Reading the Dataset using Pandas Command**_

# In[2]:


os.chdir('C:\\Users\\Shridhar\\OneDrive\\Desktop\\Top Mentor\\Batch 74 Day 20')
df=pd.read_csv('Churn_Modelling.csv')


# ## Data Cleaning:
# _**Dropping the column which is not needed for model fitting**_

# In[3]:


df.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)


# _**Automated Exploratory Data Analysis (EDA) with ydata_profiling(pandas_profiling)**_

# In[4]:


pf.ProfileReport(df)


# _**Label Encoding the <span style= "color:green">Gender</span> column**_

# In[5]:


df['Gender']=df['Gender'].astype('category')
df['Gender']=df['Gender'].cat.codes


# _**One Hot Encoding the <span style = "color:green"> Geography </span> column using pandas get dummies command**_

# In[6]:


geo=pd.get_dummies(df['Geography'],drop_first=True)
df=pd.concat([geo,df],axis=1)


# _**Dropping out the <span style = "color:green"> Geography </span> column after one hot encoding**_

# In[7]:


df.drop(['Geography'],axis=1,inplace=True)


# _**Assigning the dependent and independent variable**_

# In[8]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# ## Data Preprocessing:
# _**Standardizing the independent variable of the dataset**_

# In[9]:


sc=StandardScaler()
x=sc.fit_transform(x)


# ## Model Fitting:

# _**Defining the Function for the ML algorithms using GridSearchCV Algorithm and splitting the dependent variable & independent variable into training and test dataset and Predicting the Dependent Variable by fitting the given model and create the pickle file of the model with the given Algo_name. Further getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Classification Report and Confusion Matrix between the predicted values and dependent test dataset.**_

# In[10]:


def FitModel(x,y,algo_name,algorithm,gridsearchParams,cv):
    np.random.seed(10)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
    grid=GridSearchCV(estimator=algorithm,param_grid=gridsearchParams,cv=cv,
                     scoring='accuracy', verbose=0,n_jobs=-1)
    grid_result = grid.fit(x_train,y_train)
    best_params = grid_result.best_params_
    pred = grid_result.predict (x_test)
    pickle.dump(grid_result,open(algo_name,'wb'))
    print('Algorithm Name :',algo_name)
    print ('\n Best Params :', best_params)
    print ('\n Classification Report :\n',classification_report(y_test,pred))
    print ('\n Accuracy Score {}%'.format(100* accuracy_score(y_test,pred)))
    print ('\n Confusion Matrix :\n',confusion_matrix(y_test,pred))


# _**Running the function with some appropriate parameters and fitting the Support Vector Machine Classifiers Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name SVC.**_

# In[11]:


param= {'C':[0.1,1,10,100,1000],
       'gamma':[0.0001,0.001,0.1,1,3,5,10,100]}
FitModel(x,y,'SVC',SVC(),param,cv=10)


# _**Running the function with some appropriate parameters and fitting the Random Forest Classifiers Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name Random Forest.**_

# In[12]:


param = {'n_estimators':[500,600,800,1000],
        'criterion':['entropy','gini']}
FitModel(x,y,'Random Forest',RandomForestClassifier(),param,cv=7)


# _**Running the function with some appropriate parameters and fitting the XGBoost Classifiers Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name XGBoost.**_

# In[13]:


param = {'n_estimators':[555,666,777,888,999]}
FitModel(x,y,'XGBoost',XGBClassifier(),param,cv=5)


# _**Running the function with some appropriate parameters and fitting the Extra Tree Classifiers Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name Extra Tree.**_

# In[14]:


param = {'n_estimators':[500,600,800,1000],
        'criterion':['entropy','gini'],
        'max_features':['auto','sqrt']}
FitModel(x,y,'Extra Tree',ExtraTreesClassifier(),param,cv=4)


# _**Running the function with empty parameters since the catboost model doesn't need any special parameters and fitting the CatBoost Classifiers Algorithm and getting the Algorithm Name, Empty Parameters of the algorithm, Accuracy Score in percentage format, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name CatBoost.**_

# In[15]:


param={}
FitModel(x,y,'CatBoost',CatBoostClassifier(),param,cv=10)


# _**Running the function with empty parameters since the lightgbm model doesn't need any special parameters and fitting the LGBM Classifiers Algorithm and getting the Algorithm Name, Empty Parameters of the algorithm, Accuracy Score in percentage format, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name LightGBM.**_

# In[16]:


param = {}
FitModel(x,y,'LightGBM',LGBMClassifier(),param,cv=10)


# _**Loading the pickle file with the algorithm which gives highest accuracy percentage**_

# In[17]:


model=pickle.load(open('Extra Tree','rb'))


# _**Predicting the dependent variable using the loaded pickle file and getting the Accuracy Score in percentage format, Classification Report and Confusion Matrix between the predicted values and dependent variable**_

# In[18]:


pred1=model.predict(x)
print ('\n Classification Report :\n',classification_report(y,pred1))
print ('\n Accuracy Score {}%'.format(100* accuracy_score(y,pred1)))
print ('\n Confusion Matrix :\n',confusion_matrix(y,pred1))


# _**Making the Predicted value as a new dataframe and concating it with the given data**_

# In[19]:


prediction=pd.DataFrame(pred1,columns=['Prediction of Exited(Approx.)'])
pred_df=pd.concat([df,prediction],axis=1)


# _**Exporting the Data With Prediction of Exited to a csv file**_

# In[20]:


pred_df.to_csv('Predicted Churn Modelling.csv',index=False)


# _**Plotting the line graph to represent the Accuracy between Predicted value and Actual Value and saving the PNG file**_

# In[21]:


sns.set_style('darkgrid')
plt.plot(y,pred1)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Accuracy between Actual Value and Predicted Value')
plt.savefig('Accuracy between Actual Value and Predicted Value.png')
plt.show()

