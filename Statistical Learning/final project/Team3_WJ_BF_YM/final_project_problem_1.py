#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import data 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                    header = None)

test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test' ,
                   skiprows = 1, header = None)
total =pd.concat([train, test])
col_name = ['age', 'workclass', 'fnlwgt', 'education', 'education_num','marital_status', 'occupation',
              'relationship',  'race', 'sex', 'capital_gain', 'capital_loss',
              'hours_per_week', 'native_country', 'wage_class']
total.columns = col_name


# In[2]:


total = total.replace(' <=50K.', ' <=50K')
total = total.replace(' >50K.',' >50K')


# In[3]:


#data preprocessing
total = pd.get_dummies(total,columns = ['workclass', 'education','marital_status','occupation',
                                        'relationship',  'race','sex', 'native_country'])
y = total.wage_class
x = total.drop(columns = ['wage_class'])
y = pd.get_dummies(y,drop_first = True)
y = y.values.reshape((y.values.shape[0],))

scaler = StandardScaler()
saved_cols = x.columns
sd1 = scaler.fit_transform(x)
x = pd.DataFrame(sd1, columns = saved_cols)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.1)


# In[5]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()
params = {'max_depth': list(np.arange(3,10)) + [None],
         'min_samples_leaf': [1,2,3],
         'n_estimators': [10,100]}
grid = GridSearchCV(RF, param_grid=params, cv=4,verbose =3).fit(xtrain, ytrain)
best_RF = grid.best_estimator_
print(best_RF.get_params())


# In[6]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
params = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(log_reg, param_grid=params, cv=4,verbose =3).fit(xtrain, ytrain)
best_log_reg = grid.best_estimator_
print(best_log_reg.get_params())


# In[4]:


#SVM
from sklearn.svm import SVC # "Support vector classifier"
svm = SVC()
params = {'kernel':['poly','rbf']
          ,'C': [100,10,1,50]}
grid = GridSearchCV(svm, param_grid=params, cv=4,verbose =3).fit(xtrain, ytrain)
best_svm = grid.best_estimator_
print(best_svm.get_params())


# In[14]:


print('Random Forest')
print(best_RF.score(xtest,ytest))
print('Logistic Regression')
print(best_log_reg.score(xtest,ytest))
print('SVM')
print(best_svm.score(xtest,ytest))


# In[ ]:




