#!/usr/bin/env python
# coding: utf-8

# ## Import the relevant libraries

# In[1]:


import pandas as pd
import numpy as np


# ## Load the data

# In[2]:


data_preprocessed = pd.read_csv('df_preprocessed.csv')


# In[3]:


data_preprocessed.head()


# ## Create the targets

# In[4]:


data_preprocessed['Absenteeism Time in Hours'].median()


# In[5]:


targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 
                   data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)


# In[6]:


targets


# In[7]:


data_preprocessed['Excessive Absenteeism'] = targets


# In[8]:


data_preprocessed.head()


# ## A comment on the targets

# In[9]:


targets.sum() / targets.shape[0]


# In[10]:


data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours','Day of the Week',
                                            'Daily Work Load Average','Distance to Work'],axis=1)


# In[11]:


data_with_targets is data_preprocessed


# In[12]:


data_with_targets.head()


# ## Select the inputs for the regression

# In[13]:


data_with_targets.shape


# In[14]:


data_with_targets.iloc[:,:14]


# In[15]:


data_with_targets.iloc[:,:-1]


# In[16]:


unscaled_inputs = data_with_targets.iloc[:,:-1]


# ## Standardize the data

# In[17]:


from sklearn.preprocessing import StandardScaler

absenteeism_scaler = StandardScaler()


# In[18]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    def __init__(self,columns):
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# In[19]:


unscaled_inputs.columns.values


# In[20]:


#columns_to_scale = ['Month Value','Day of the Week', 'Transportation Expense', 'Distance to Work',
       #'Age', 'Daily Work Load Average', 'Body Mass Index', 'Children', 'Pet']

columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Education']


# In[21]:


columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]


# In[22]:


absenteeism_scaler = CustomScaler(columns_to_scale)


# In[23]:


absenteeism_scaler.fit(unscaled_inputs)


# In[24]:


scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)


# In[25]:


scaled_inputs


# In[26]:


scaled_inputs.shape


# ## Split the data into train & test and shuffle

# ### Import the relevant module

# In[27]:


from sklearn.model_selection import train_test_split


# ### Split

# In[28]:


train_test_split(scaled_inputs, targets)


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, #train_size = 0.8, 
                                                                            test_size = 0.2, random_state = 20)


# In[30]:


print (x_train.shape, y_train.shape)


# In[31]:


print (x_test.shape, y_test.shape)


# ## Logistic regression with sklearn

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ### Training the model

# In[33]:


reg = LogisticRegression()


# In[34]:


reg.fit(x_train,y_train)


# In[35]:


reg.score(x_train,y_train)


# ### Manually check the accuracy

# In[36]:


model_outputs = reg.predict(x_train)
model_outputs


# In[37]:


y_train


# In[38]:


model_outputs == y_train


# In[39]:


np.sum((model_outputs==y_train))


# In[40]:


model_outputs.shape[0]


# In[41]:


np.sum((model_outputs==y_train)) / model_outputs.shape[0]


# ### Finding the intercept and coefficients

# In[42]:


reg.intercept_


# In[43]:


reg.coef_


# In[44]:


unscaled_inputs.columns.values


# In[45]:


feature_name = unscaled_inputs.columns.values


# In[46]:


summary_table = pd.DataFrame (columns=['Feature name'], data = feature_name)

summary_table['Coefficient'] = np.transpose(reg.coef_)

summary_table


# In[47]:


summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# ## Interpreting the coefficients

# In[48]:


summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)


# In[49]:


summary_table


# In[50]:


summary_table.sort_values('Odds_ratio', ascending=False)


# ## Testing the model

# In[51]:


reg.score(x_test,y_test)


# In[52]:


predicted_proba = reg.predict_proba(x_test)
predicted_proba


# In[53]:


predicted_proba.shape


# In[54]:


predicted_proba[:,1]


# ## Save the model

# In[55]:


import pickle


# In[56]:


with open('model', 'wb') as file:
    pickle.dump(reg, file)


# In[57]:


with open('scaler','wb') as file:
    pickle.dump(absenteeism_scaler, file)


# In[ ]:




