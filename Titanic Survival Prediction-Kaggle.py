#!/usr/bin/env python
# coding: utf-8

# #### * Importing Required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### * Reading Data set

# In[2]:


train_data=pd.read_csv("titanic_train.csv")


# In[3]:


train_data


# In[4]:


train_data.shape


# In[5]:


train_data.info()


# In[6]:


train_data.describe()


# ### EDA

# #### * Missing Values

# In[7]:


train_data.head()


# In[8]:


train_data.isnull()


# In[9]:


## Visualising using Seaborn Library
## yticklabels=False to remove the row number for all datasets
## cbar=False to disable the colour meter with probability from 0 to 1

sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False)


# In[10]:


## Check count of survived
train_data.value_counts("Survived")


# In[11]:


## See count using visuals

sns.countplot(x='Survived',data=train_data,)


# Analysis: Most of the passengers died when compared to survived

# In[12]:


## To Get absolute values in plot

ax = sns.countplot(x=train_data['Survived'],order=train_data['Survived'].value_counts(ascending=False).index);

abs_values = train_data['Survived'].value_counts(ascending=False).values

ax.bar_label(container=ax.containers[0], labels=abs_values)


# In[13]:


# To have the background of plot as grids

sns.set_style('whitegrid')

# check survival rate based on sex of passenger

sns.countplot(x="Survived",hue="Sex",data=train_data)


# Analysis: Most among survived are Females when compared to Men(More than double)

# In[14]:


## Now lets check survived based on passenger class

sns.set_style("whitegrid")
sns.countplot(x="Survived",hue="Pclass",data=train_data)


# Analysis: Most people died are from class 3 while Most people survived are from class 1

# In[15]:


# To Check on distribution with Age of passengers

sns.displot(train_data['Age'].dropna())


# Analysis: It is a bell curve graph which is normal distribution which states most of the data lies int the centre and near to mean.

# #### Data Cleaning
# ##### We have to check and clean null values

# In[16]:


## Age column has few missing values and also age plays important role in data, so we have to fill instead of dropping whole column
## Also we can fill null values in age with some central tendency value like mean, median or something, But we see different mean values based on Pclass of people, so we fill in same way


# In[17]:


sns.boxplot(x="Pclass",y='Age',data=train_data)


# Analysis: The People in class 1 and 2 are older than class 3

# In[18]:


def agefill(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[19]:


# Filling the null values in Age
train_data['Age']=train_data[["Age","Pclass"]].apply(agefill,axis=1)


# In[20]:


# Now we can see that all the values in Age are showing false for isnull - so all values are filled
train_data['Age'].isnull().value_counts()


# In[21]:


# Check with heat again
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False)


# Conclusion : Apart from cabin, we dont have null values in any column

# In[22]:


# As Cabin is not relavant to our Model, we can drop cabin column from our data
train_data.drop('Cabin',axis=1,inplace=True)
train_data.head()


# In[23]:


# Now lets see heat map again
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False)


# In[24]:


# we see only negligible null values in Embarked, so dropping rows na values exist
train_data.dropna(inplace=True)


# In[25]:


sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False)


# #### Converting categorical values and dropping irrelavant columns

# In[26]:


# Converting sex column categories to 1 and 2 values

train_data.Sex[train_data.Sex == 'male'] = 1
train_data.Sex[train_data.Sex == 'female'] = 2
train_data.head()


# In[27]:


train_data.drop(['Name','Ticket','Embarked'],axis=1,inplace=True)


# In[28]:


train_data.head()


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train,X_test,Y_train,Y_test=train_test_split(train_data.drop('Survived',axis=1),train_data['Survived'],test_size=0.30,random_state=1)


# Training and Predicting

# In[31]:


from sklearn.linear_model import LogisticRegression


# In[32]:


model1=LogisticRegression()
model1.fit(X_train,Y_train)


# In[33]:


predictions=model1.predict(X_test)


# In[34]:


from sklearn.metrics import confusion_matrix


# In[35]:


accuracy=confusion_matrix(Y_test,predictions)


# In[36]:


accuracy


# In[37]:


from sklearn.metrics import accuracy_score


# In[38]:


accuracy=accuracy_score(Y_test,predictions)
accuracy


# In[39]:


predictions


# #### As of Now we have made model of train dataset and splitted data and tested the same train data, model gave accuracy of 85%

# #### Now we are going to test the test dataset given in kaggle website

# In[40]:


test_data=pd.read_csv("titanic_test.csv")


# In[41]:


test_data.head()


# In[42]:


## Lets see the null values
sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False)


# Analysis: we have null values in age column, cabin and fare, we have to clean them

# In[43]:


test_data=test_data.drop('Cabin',axis=1)


# In[44]:


test_data.head()


# In[45]:


## Now lets see the mean of age column based on pclass
sns.boxplot(x="Pclass",y='Age',data=test_data)


# In[46]:


def agefill(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 42
        elif Pclass==2:
            return 26
        else:
            return 24
    else:
        return Age


# In[47]:


test_data['Age']=test_data[["Age","Pclass"]].apply(agefill,axis=1)


# In[48]:


## we could see all the Age columns are filled with mean values based on Pclass
test_data['Age'].isnull().value_counts()


# In[49]:


test_data.isnull().value_counts()


# In[50]:


test_data.Sex[test_data.Sex == 'male'] = 1
test_data.Sex[test_data.Sex == 'female'] = 2
test_data.head()


# In[51]:


test_data.drop(['Name','Ticket','Embarked'],axis=1,inplace=True)


# In[52]:


test_data.head()


# In[53]:


train_data.head()


# In[54]:


test_data["Fare"].fillna('0',inplace=True)


# In[55]:


submission_predictions=model1.predict(test_data)


# In[56]:


test_ids=test_data["PassengerId"]


# In[57]:


df=pd.DataFrame({"PassengerId":test_ids.values,
                "Survived":submission_predictions})


# In[58]:


# df.to_csv("Submission.csv",index=False)


# In[59]:


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)  


# In[60]:


from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier.fit(X_train, Y_train)  


# In[61]:


random_predict= classifier.predict(X_test)


# In[62]:


accuracy=accuracy_score(Y_test,random_predict)
accuracy


# In[63]:


submission_predict_random=classifier.predict(test_data)


# In[64]:


df1=pd.DataFrame({"PassengerId":test_ids.values,
                "Survived":submission_predict_random})


# In[65]:


# df1.to_csv("Submission2.csv",index=False)


# In[66]:


df2=pd.DataFrame({"PassengerId":test_ids.values,
                "Survived":submission_predict_random})


# In[67]:


# df2.to_csv("Submission3.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[70]:


train_data.head()


# In[72]:


train_data1=train_data.drop("Survived",axis=1)


# In[73]:


train_data1


# In[88]:


X_train,X_test,Y_train,Y_test=train_test_split(train_data.drop('Survived',axis=1),train_data['Survived'],test_size=0.20,random_state=1)


# In[89]:


from sklearn.tree import DecisionTreeClassifier

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth 
dt = DecisionTreeClassifier()

# Fit dt to the training set
dt.fit(X_train, Y_train) # it will ask all possible questions, compute the information gain and choose the best split

# Predict test set labels
y_pred = dt.predict(X_test)
y_pred


# In[94]:


from sklearn.metrics import accuracy_score, roc_auc_score, plot_roc_curve
#we compute the eval metric on test/validation set only primarily

# Predict test set labels
y_pred = dt.predict(X_test) 

# Compute test set accuracy
acc = accuracy_score(Y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))
acc = roc_auc_score(Y_test, y_pred)
print("Test set auc: {:.2f}".format(acc))
plot_roc_curve(dt, X_test, Y_test)


acc = roc_auc_score(Y_train, dt.predict(X_train))
print("Train set AUC : ", acc)


# In[96]:


##there are 1 param to play with - max depth - to choose the best parameter, I will try different combinations and hoose the one which has the best accuracy/auc/any eval metric on cross-validation or test data-set
from sklearn.model_selection import cross_val_score #this will help me to do cross- validation
import numpy as np

for depth in [1,2,3,4,5,6,7,8,9,10,20]:
  dt = DecisionTreeClassifier(max_depth=depth) # will tell the DT to not grow past the given threhsold
  # Fit dt to the training set
  dt.fit(X_train, Y_train) # the model is trained
  trainAccuracy = accuracy_score(Y_train, dt.predict(X_train)) # this is useless information - i am showing to prove a point
  dt = DecisionTreeClassifier(max_depth=depth) # a fresh model which is not trained yet
  valAccuracy = cross_val_score(dt, train_data.drop('Survived',axis=1), train_data['Survived'], cv=10) # syntax : cross_val_Score(freshModel,fts, target, cv= 10/5)
  print("Depth  : ", depth, " Training Accuracy : ", trainAccuracy, " Cross val score : " ,np.mean(valAccuracy))


# In[103]:


X_train,X_test,Y_train,Y_test=train_test_split(train_data.drop('Survived',axis=1),train_data['Survived'],test_size=0.20,random_state=1)


# In[110]:


dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train,Y_train)


# In[111]:


Submission_pred=dt.predict(test_data)


# In[112]:


df4=pd.DataFrame({"PassengerId":test_ids.values,
                "Survived":Submission_pred})


# In[113]:


df3.to_csv("Submission4.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




