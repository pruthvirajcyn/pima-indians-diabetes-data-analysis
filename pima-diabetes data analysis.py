#!/usr/bin/env python
# coding: utf-8

# # PIMA Indian Diabetes
# 
# The Pima Indians diabeties data has 768 instances, each instance having 8 attributes with which the patients were medically tested upon. Using the data of the tested patients, based on the 8 attributes, we use different classfication ML techniques to predict if the patient is actually a diabetic or not.
# The attributes on which the patients were tested are:
# 1)Pregnencies
# 2)Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3)Diastolic blood pressure (mm Hg)
# 4)Triceps skin fold thickness (mm)
# 5)2-Hour serum insulin (mu U/ml)
# 6)BMI: Body mass index (weight in kg/(height in m)^2)
# 7)Diabetes pedigree function
# 8)Age (In years)
# 
# Finally the 9th column is terms of 0 and 1, 1: Tested positive for diabetes, 0: tested negative for diabetes
# preg = Number of times pregnant
# 
# 
# 

# We start of by importing the required libraries

# In[4]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns


# Importing the data from pima-indians-diabetes.csv using pandas into the atrributes the patients were tested upon.
# We have a look at the data using .head() method

# In[5]:


All_features=['Pregnencies','PlasmaGlucose','DBP','Skin_thickness','insulin','BMI','DPedigreeFn','Age','Diabetic']
dataset=pd.read_csv("F:\cellstrat\ML\module 6\pima-indians-diabetes.data", names=All_features)
dataset.head()


# Now we look at the data description to find out if there any missing values and the trend in the data
# 

# In[6]:


dataset.describe()


# Thankfully, there are no missing values.
# Now we look upon the data for any corelation between the paramters

# In[7]:


X_array=dataset.iloc[: ,:-1].values
Y_array=dataset['Diabetic'].values
X_features=['Pregnencies','PlasmaGlucose','DBP','Skin_thickness','insulin','BMI','DPedigreeFn','Age']
scatter_matrix(dataset)
plt.xticks(rotation='vertical')
# plt.ylabels(rotation=60)
plt.show()


# The distribution is more or less random. We cannot make out any corelation from the above distributions
# 

# Now we normalize the data between 0 to 1 usinf MinMaxScaler, so that all out input features are have equalimportance for prediction

# In[8]:


Scalar=preprocessing.MinMaxScaler()
Scaled_X=Scalar.fit_transform(X_array)
Scaled_X


# Lets split the data into testing and training samples.

# In[9]:


from sklearn.model_selection import train_test_split
(train_input,test_input,train_output,test_output)=train_test_split(Scaled_X,Y_array,train_size=0.75,random_state=31)


# ### K-Near Neighbors

# Lets start witk K-nearest neighbors model

# In[10]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(train_input,train_output)
y_pred=knn.predict(test_input)
knn.score(test_input,test_output)


# By varying the neighbors we can get better results 

# In[11]:


knn_scores=[]
a=[]
for k in range(1,50):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_input,train_output)
    knn_scores.append(knn.score(test_input,test_output))
    a.append(k)
    
q=knn_scores.index(max(knn_scores))+1
maxi=max(knn_scores)
print("The best result was obtained for k : %d is %.2f" %(q,maxi))


# ### Decision Tress and Random forest

# Now lets use Decision tress and random forest to check out the accuracy of the model

# In[12]:


DecTree=DecisionTreeClassifier(random_state=31)
DecResult=DecTree.fit(train_input,train_output)
DecTree.score(test_input,test_output)


# Lets plot the decision tree classifier

# In[13]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()  
tree.export_graphviz(DecResult, out_file=dot_data,  
                         feature_names=X_features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  


# The accuracy with decision tress is stil quite less. So lets try with Random forest which is a form ensenble learning. It selected random tress from the lot and takes the average to give better results.

# In[14]:


RandFor=RandomForestClassifier(n_estimators=10)
RandForRes=RandFor.fit(train_input,train_output)
RandForRes.score(test_input,test_output)


# Much better than our Decision tree results. Lets try if the results actually change by taking different n_estimators

# In[15]:


scores=[]
for n in range(5,50):
    RandFor=RandomForestClassifier(n_estimators=n)
    RandForRes=RandFor.fit(train_input,train_output)
    scores.append(RandForRes.score(test_input,test_output))
maxi=max(scores)    
t=scores.index(max(scores))+1
print("The maximum accuracy was obtained for %d estimators as %.2f" %(t,maxi))
    


# ### Support vector machines

# Lets try Support vector machines for classification and see how well the hyperplane can classify the diabetics

# In[16]:


svm_linear = SVC(kernel='linear',C=1,gamma='scale')
svm_linear.fit(train_input,train_output)
svm_linear.score(test_input,test_output)


# Lets try with other kernels

# In[17]:


svm_polynomial= SVC(kernel='poly',C=1,gamma='scale')
svm_polynomial.fit(train_input,train_output)
svm_polynomial.score(test_input,test_output)


# In[18]:


svm_rbf= SVC(kernel='rbf',C=1,gamma='scale')
svm_rbf.fit(train_input,train_output)
svm_rbf.score(test_input,test_output)


# Using rbf kernel we get the better results in case of SVM's

# ### Logistic regression

# Let's now try logistic regression, the simplest binary classification model when compared to all the other models

# In[19]:


LR=LogisticRegression(solver='newton-cg')
LR.fit(train_input,train_output)
LR.score(test_input,test_output)


# Just by using an appropriate solver, we got the best results when compared all the other classfication models.

# ### Deep Neural Network

# In[36]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers  import Dense, Dropout

model=Sequential()
model.add(Dense(512,input_dim=8,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

NN=model.fit(train_input,train_output,batch_size=100,epochs=100,verbose=2,validation_data=(test_input,test_output))


# In[39]:


score=model.evaluate(test_input,test_output,verbose=0)
Accuracy=score[1]
print("the accuracy obtained on using neural networks was %.2f" %(Accuracy))


# It turns out that even with a complicated nerual network model we able to achieve only 78% accuracy, where are with a simple logistic regression model we achieved 77.60% accuracy.
# I gues we can conclude that the most simple model gave almost the best accuracy. 

# In[ ]:




