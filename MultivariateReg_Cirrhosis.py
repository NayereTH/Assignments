#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


##softmax will give us several classes/probabiliy vs sigmoid gives us binary
def softmax(h):
    return (np.exp(h.T)/np.sum(np.exp(h),axis=1)).T
   

#define cross entropy

def cross_entropy(Y,P_hat):
    return -(1/len(Y))*np.sum(np.sum(Y*np.log(P_hat),axis=1),axis=0)

def accuracy(y,y_hat):
    return np.mean(y==y_hat)

def indices_to_one_hot(data,nb_classes):
    targets= np.array(data).reshape(-1) ##or  ((data).reshape(-1))
    return np.eye(nb_classes)[targets]

    


# In[3]:


class MVLogisticRegression():
    def __init__(self, thresh=0.5):
        self.thresh=thresh
        
    def fit(self, X,y, eta=2e-1, epochs=1e3, show_curve= False):
        epochs= int(epochs)
        N, D= X.shape
        K=len(np.unique(y))
        y_values= np.unique(y, return_index=False) ##actual values
        Y= indices_to_one_hot(y,K).astype(int)
        self.W= np.random.randn(D,K)
        #self.A= np.random.randn(N,1)
        self.B= np.random.randn(1,K)
        J=np.zeros(int(epochs))
        
        for epoch in range(epochs):
            P_hat= self.__forward__(X)
            J[epoch]= cross_entropy(Y,P_hat)
            self.W -= eta*(1/N)*X.T@(P_hat-Y) ##updating
            self.B -= eta*(1/N)*np.sum(P_hat-Y,axis=0)
        if show_curve:
            plt.figure()
            plt.plot(J)
            plt.xlabel("epochs")
            plt.ylabel("$\mathcal{J}$")
            plt.title("training curve" )
            plt.show()
            
            
    def __forward__(self,X):
        
        return softmax(X@self.W +self.B)
    
    def predict(self,X):
        return np.argmax(self.__forward__(X), axis=1)
           


# In[4]:


import os
os.getcwd()


# In[5]:


df= pd.read_csv("C:\\Users\\Joe\\Downloads\\Cirrhosis.csv")


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


missing_values=["NA","n/a","--","","N/A","na","None","none","NONE","n.a.","?"]
df=pd.read_csv("C:\\Users\\Joe\\Downloads\\Cirrhosis.csv",na_values=missing_values)


# In[9]:


df.isnull().sum()


# In[10]:


#df["cholestAndTrigly"]=df["triglicerides"]+df[""]


# In[57]:


df["cholesterol"].interpolate(method="linear",dirction="forward",inplace=True)
#df["cholesterol"].fillna(df["cholesterol"].mean(), inplace=True)
df["triglicerides"].interpolate(method="linear",dirction="forward",inplace=True)
df["platelets"].fillna(df["platelets"].mean(), inplace=True)
df["copper"].fillna(df["copper"].mean(), inplace=True)
df.isnull().sum()


# In[55]:


accuracy(df["cholesterol"],df1)


# In[12]:


"""
#list_num= ['duration','status','drug', 'age', 'ascites', 'hepatomology', 'spiders', 'edema', 'bilirubin', \
           ##'cholestero0l', 'albumin', 'copper','phosphatase','SGOT','triglicerides','platelets','prothrombin','stage']

list_num=['index','drug','duration','age','sex','hepatomology','phosphatase','platelets','copper','status']
for columnsA in df1:
  for columnsB in df1:
    if columnsA != columnsB and columnsA in list_num and  columnsB in list_num:
      fig, ax = plt.subplots(figsize=(8,4))
      ax.scatter(df1[columnsA], df1[columnsB])
      ax.set_xlabel(columnsA,fontname="Times New Roman" ,fontsize=20, fontweight="bold")
      ax.set_ylabel(columnsB,fontname="Times New Roman" ,fontsize=20, fontweight="bold")
      plt.xticks(fontweight="bold")
      plt.yticks(fontweight="bold")
      plt.show()
  """  


# In[13]:


"""
df= df.drop(df[df["copper"]>=300].index,axis=0)
df=df.drop(df[df["phosphatase"]>= 8500].index,axis=0)
df= df.drop(df[df["platelets"]>= 500].index,axis=0)
df= df.drop(df[df["status"]>=400].index,axis=0)
"""


# In[14]:


import seaborn as sns
#sns.heatmap(df11.drop(["kitchen_features","fireplaces","floor_covering"], axis=1), linewidths=0.5)
corr= df.corr()

sns.heatmap(corr,linewidths=0.5,cmap="viridis")


# In[15]:


df1=df
#df["mix1"]=df["edema"]+df["bilirubin"]+df["spiders"]+df["hepatomology"]+df["ascites"]+df["prothrombin"]
df1["mix1"]=df1["edema"]+df1["bilirubin"]+df1["spiders"]+df1["ascites"]+df1["prothrombin"]
#df1= df.drop(["ascites","spiders","edema","bilirubin","prothrombin","status","copper"], axis=1)
df1["mix2"]=df1["copper"]+df1["status"]
#df1=df1.drop(["copper","status"], axis=1)
df1["mix3"]=df1["cholesterol"]+df1["SGOT"]+df1["triglicerides"]
df1["mixA"]=df1["mix1"]+df1["mix2"]+df1["mix3"]
#df1=df1.drop(["SGOT","triglicerides"], axis=1)
#df1["mix1"]=df["edema"]+df["bilirubin"]+df["spiders"]+df["ascites"]+df["prothrombin"]
df1.head()

"""
"""


# In[16]:


import seaborn as sns
corr= df1.corr()

sns.heatmap(corr,linewidths=0.5,cmap="viridis")


# In[17]:


"""#df["index","duration","drug","sex","cholesterol","platelets","stage"]
df1= df.drop(["age","ascites","hepatomology","spiders","edema","bilirubin","albumin","copper","phosphatase",\
          "SGOT","triglicerides","prothrombin","status"], axis=1)
df1.head()"""

df1=df


# In[18]:


df1["stage"]=df1["stage"].replace(1,0).replace(2,1).replace(3,2).replace(4,3)
df1["stage"].value_counts()


# In[19]:


#X=df1.to_numpy()
df1.head()


# In[20]:


#y=df1["stage"].to_numpy()
#y.shape
#X=X[:,0:6]
#X= df1[['index','drug','duration','age','sex','hepatomology','phosphatase','platelets',"mix1","mix2",'albumin','stage']].to_numpy()
X= df1[['hepatomology','status','mix1','sex','stage']].to_numpy() ##accuracy 0.587

#X= df1[['hepatomology','status','mix3','sex','edema','stage']].to_numpy() #0.56

#X= df1[['hepatomology','status','mix1','sex','mix3','age','stage']].to_numpy()


# In[21]:


#import seaborn as sns
#corr= df1[['hepatomology','status','mix1','sex','stage']].corr()

#sns.heatmap(corr,linewidths=0.5,cmap="viridis")


# In[22]:


import random
random.seed(230)
random.shuffle(X) # shuffles the ordering of filenames (deterministic given the chosen seed)
split_1 = int(0.80 * len(X))
#split_2 = int(0.9 * len(X))
#split_3= int(0.9 * len(X))
train_filenames = X[:split_1]
train_X= train_filenames[:,0:4]
train_X= train_X / train_X.max(axis=0)
train_Y= train_filenames[:,4]
#print(train_Y)

train_X = np.array(train_X, dtype=np.float64)
train_Y = np.array(train_Y, dtype=np.int64)

"""

valid_filenames = X[split_1:split_2]

valid_X= valid_filenames[:,0:4]
valid_X= valid_X / valid_X.max(axis=0)
valid_Y= valid_filenames[:,4]
#print(train_Y.shape)

valid_X = np.array(valid_X, dtype=np.float64)
valid_Y = np.array(valid_Y, dtype=np.int64)
"""

test_filenames = X[split_1:]

test_X= test_filenames[:,0:4]
test_X= test_X / test_X.max(axis=0)
test_Y= test_filenames[:,4]
#print(train_Y.shape)

test_X = np.array(test_X, dtype=np.float64)
test_Y = np.array(test_Y, dtype=np.int64)



"""
"""
print(len(np.unique(train_Y)))
print(test_X.shape)


# In[23]:


logreg= MVLogisticRegression()
#print(len(test_Y))


# In[24]:


logreg.fit(train_X,train_Y,eta=1e-2, epochs=1e5, show_curve=True)


# In[26]:


y_hat_train= logreg.predict(train_X)
#print(y_hat_train)
#y_hat_valid= logreg.predict(valid_X)
y_hat_test= logreg.predict(test_X)
print(test_X.shape)
#train_X.shape


# In[27]:


#accuracy(train_Y,y_hat_train)
print(y_hat_train)


# In[28]:


accuracy(train_Y,y_hat_train)


# In[29]:


accuracy(test_Y,y_hat_test)
#len(test_Y)


# In[30]:


#accuracy(valid_Y,y_hat_valid)


# In[31]:


#y_hat_test


# In[40]:


from sklearn.metrics import confusion_matrix
y_true = test_Y
y_pred = y_hat_test

classes=[0,1,2,3]

cm =confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
cm


# In[58]:


"""
from sklearn.metrics import plot_confusion_matrix


cnf_matrix = confusion_matrix(y_true, y_pred,labels=[0,1,2,3])
#np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1,2,3],title='Confusion matrix, without normalization')

"""


# In[ ]:



import seaborn as sns

plt.figure(2,figsize=(8,6))
sns.heatmap(cnf_matrix,annot=True,vmax=1,vmin=0,square=True,cmap="viridis" ,fmt='g',annot_kws={"size": 16})
#plt.set_label_position("top")
plt.xticks([0.5,1.5,2.5,3.5],labels=[0,1,2,3])
plt.yticks([0.5,1.5,2.5,3.5],labels=[0,1,2,3])
plt.ylabel('True label',fontsize=20, fontweight="bold")
plt.xlabel('Predicted label',fontsize=20, fontweight="bold")
plt.set_ylim([0,0.5,1,1.5])
sns.set(font_scale=1.5)


# In[34]:



sns.heatmap(cnf_matrix/np.sum(cnf_matrix), annot=True, 
            fmt='g', cmap='Blues')


# In[48]:


#cm = np.round (cm.astype('int'),2)
fig, ax = plt.subplots(figsize=(8,8))
plt.imshow(cm, cmap=plt.cm.Pastel1)
#plt.ylabel('Actual')
#plt.xlabel('Predicted')
#plt.show(block=False)
for i in range(4):
    for j in range(4):
        plt.text(j,i, str(cm[i][j]),fontweight="bold",fontsize=20)
      
        plt.xticks([-0.5,0.5,1.5,2.5,3.5],labels=[1,2,3,4],fontsize=16, fontweight="bold")
        plt.yticks([-0.5,0.5,1.5,2.5,3.5],labels=[1,2,3,4],fontsize=16, fontweight="bold")
plt.ylabel('True label',fontsize=20, fontweight="bold")
plt.xlabel('Predicted label',fontsize=20, fontweight="bold")
plt.savefig("./ConfusionMatricsTestSetKNN.png" , dpi=150)
plt.show()


# In[ ]:




