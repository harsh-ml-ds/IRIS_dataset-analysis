#!/usr/bin/env python
# coding: utf-8

# # Mod1-Dimentional Reduction

# Demo-1-IRIS dataset

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[4]:


names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'target']
df=pd.read_csv('iris.csv', names=names)


# In[5]:


df.head()


# In[14]:


features=['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
x=df.loc[:, features].values    #values of columns in features seprated and stored in dataframe x
y=df.loc[:, ['target']].values  #values of column 'target' seprated and stoed in dataframe y


# In[21]:


sc=StandardScaler()
x=sc.fit_transform(x)


# In[36]:


pca=PCA(n_components=2)   #here the components(features) are reduced to 2 so that the 2 most features are going to be responsible
principalComponents=pca.fit_transform(x)
principal_df=pd.DataFrame(data=principalComponents, columns={'PrincipalComponent1', 'PrincipalComponent2'})
principal_df


# In[32]:


final_df=pd.concat([principal_df, df['target']], axis=1)
print(final_df.target[56], final_df.target[2], final_df.target[149])


# In[40]:


fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component1', fontsize=15)
ax.set_ylabel('Principal Component2', fontsize=15)
ax.set_title('2 components PCA', fontsize=20)
targets=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors=['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep=final_df['target']==target
    ax.scatter(final_df.loc[indicesToKeep, 'PrincipalComponent1'], final_df.loc[indicesToKeep, 'PrincipalComponent2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()


# In[42]:


pca.explained_variance_ratio_     #gives the contribution of each Principal Component i dataset


# Demo-2-Using LDA

# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


plt.rcParams['font.size']=14
plt.rcParams['lines.markersize']=10


# In[48]:


names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
df1=pd.read_csv('iris.csv', names=names)
df1.head()


# In[56]:


x=df1.iloc[:, 0:4].values     #storing the values of columns:{sepal-length, sepal-width, petal-length, petal-width}
y=df1.Class.map({'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}).values   #storing Class column as 1,2,3


# In[58]:


#Training and Testing datasets
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)


# In[64]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[65]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[66]:


lda=LDA(n_components=1)
x_train=lda.fit_transform(x_train, y_train)
x_test=lda.transform(x_test)


# In[69]:


plt.figure(figsize=(12,4))
plt.scatter(x_test, np.zeros(len(x_test)), c=y_test)
plt.grid()
plt.show()


# In[70]:


'''GRID:-
The grid() function of axes object sets visibility of grid inside the figure to on or off. 
You can also display major / minor (or both) ticks of the grid. Additionally color, linestyle and 
linewidth properties can be set in the grid() function.
'''


# In[71]:


from sklearn.ensemble import RandomForestClassifier


# In[72]:


classifier=RandomForestClassifier()


# In[73]:


classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)


# In[74]:


from sklearn.metrics import accuracy_score
print("Accuracy Score: ", accuracy_score(y_test, y_pred))


# In[ ]:




