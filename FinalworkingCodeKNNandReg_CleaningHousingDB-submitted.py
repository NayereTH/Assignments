#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[28]:


import os
os.getcwd()


# In[29]:


df= pd.read_csv("./raw_house_data.csv")
df


# In[30]:


#df.head()


# In[31]:


df.isnull().sum()
##df.notnull().sum()


# In[32]:


missing_values=["NA","n/a","--","","N/A","na","None","none","NONE","n.a.","?"]
df1= pd.read_csv('./raw_house_data.csv',na_values=missing_values)


# In[33]:


df1.isnull().sum()
#df1.shape


# In[34]:


df1.dtypes


# In[35]:


Min1=df1["sold_price"].min()
Mean1=df1["sold_price"].mean()
Max1=df1["sold_price"].max()

Mean2=df1["lot_acres"].mean()
Min2=df1["lot_acres"].min()
Max2=df1["lot_acres"].max()


# In[36]:


list1=[]
list2=[]
from statistics import mean
for i in df1["lot_acres"]:
  if  Min2<i<Mean2:
    list1.append(i)

  else:
    if Mean2<i<Max2:
      list2.append(i)

print(Mean2)
print(mean(list1))
print(mean(list2))


# In[37]:


for i,j in zip(df1["sold_price"],df1["lot_acres"]):
  if Mean1<i<Max1:
    df1["lot_acres"].fillna(mean(list2), inplace=True)
  else:
    df1["lot_acres"].fillna(Mean2, inplace=True)


# In[38]:


df1["lot_acres"].isnull().sum()


# In[39]:


#df1["floor_covering"].apply(lambda x : x.fillna(x.value_counts().index[0]))
N=df1["floor_covering"].value_counts().index[0]
df1["floor_covering"].fillna(N, inplace=True)
##same for kichen features
N=df1["kitchen_features"].value_counts().index[0]
df1["kitchen_features"].fillna("Unknown", inplace=True)


# In[40]:


df1.isnull().sum()


# In[41]:


df1["sqrt_ft"].interpolate(method="linear",dirction="forward",inplace=True)
df1["garage"].interpolate(method="linear",dirction="forward",inplace=True)
df1.isnull().sum()


# In[42]:


bool1 = pd.isnull(df1['bathrooms'])
bool2 = pd.isnull(df1['garage'])
df1[bool1]
#df1[bool2]


# In[43]:


for i,j in zip(df1["bedrooms"],df1["bathrooms"]):
  if 0<i<2:
    df1["bathrooms"].fillna(1, inplace=True)
  elif 2<=i<=4:
    df1["bathrooms"].fillna(2, inplace=True)
  elif 4<i<=6:
    df1["bathrooms"].fillna(3, inplace=True)
  elif 6<i<10:
    df1["bathrooms"].fillna(4, inplace=True)
df1.isnull().sum()


# In[44]:


#df1=df1.drop(["HOA"],axis=1)
df2=df1.drop(axis=1,columns="HOA")
df2.shape


# In[45]:


##outliers

df3= df2.drop(df2[df2["sold_price"]> 4e6].index,axis=0)
#df3[:]
df4=df3.drop(df3[df3["longitude"]> -109.5].index,axis=0)
df5= df4.drop(df4[df4["latitude"]> 32.8].index,axis=0)
df6= df5.drop(df5[df5["taxes"]> 600000].index,axis=0)
df7= df6.drop(df6[df6["bedrooms"]> 9].index,axis=0)
df7= df7.drop(df7[df7["bedrooms"]< 2].index,axis=0)
df8= df7.drop(df7[df7["bathrooms"]> 30].index,axis=0)
df9= df8.drop(df8[df8["sqrt_ft"]> 20000].index,axis=0)
df10= df9.drop(df9[df9["garage"]> 27].index,axis=0)
df11= df10.drop(df10[df10["lot_acres"]> 250].index,axis=0)


# In[46]:


df11.shape
plt.scatter(df11["taxes"],df11["sqrt_ft"])


# In[47]:


df11.to_csv('cleanedDB.csv')


# In[48]:


df11.isnull().sum()


# In[49]:


df12= df11.drop(["bathrooms", "sold_price","latitude","MLS","kitchen_features","floor_covering"], axis=1)
df12.head()
df13=df12.drop(["taxes"], axis=1)
X=df12.to_numpy()
#X for classification problem: class:bedrooms other features: garage,latitude
XX=df13.to_numpy()
#X_class= df13["MLS","latitude"].to_numpy()
#y_class=df13["bedrooms"].to_numpy()
y=df12["taxes"]
y=X[:,3:4]


#XX = np.array(df13)
yy = np.array(y)

y.shape


# In[50]:


df11.shape
#df13.isnull().sum() ##9 features


# In[51]:


X=X[:,5:]
df13.max()
X.shape[1] #seconde element of shape


# In[52]:


X1=X[:,3]
print(X1)
#X1.shape

print(df12["garage"])


# In[53]:


X=df11.to_numpy()
df11.head()
X1=X[:,3]

Columns=["longitude","lot_acres","year_built","bedrooms","garage","sqrt_ft"]
for indx, val in enumerate(Columns):
    for indx1, val1 in enumerate(Columns):
        if indx!=indx1:
            plt.scatter(X[:,indx],X[:,indx1],c=X1,s=8,alpha=0.5)
            plt.xlabel(val)
            plt.ylabel(val1)
            plt.show()   

"""   
"""


# In[54]:


def conditions(i):
        
    if 0<i["bedrooms"]<5 :
        val=1
    else :
        val=2
    return val
"""

    if 1<i<2:
        return 2
if 2<i<3:
        return 3
    #i#f 3<i<4:
    #    return 4
    #if 4<i<5:
    #    return 5
    #if 5<i<6:
    #    return 6
    #if 6<i<7:
            return 7
        if 7<i<8:
            return 8
        if 8<i<9:
            return 9
        if 9<i<10:
            return 10
        if 10<i<11:
            return 11
"""


# In[55]:


df12["bedroomsAA"]=df12.apply(conditions,axis=1)


# In[30]:


df12


# In[31]:


check_categories=["zipcode","bedrooms","garage","fireplaces","kitchen_features","floor_covering","year_built"]
DD={}
for i in check_categories:
    #DD[i]=(list(df12[i].value_counts().count()))
    DD[i]= len(list(df12[i].value_counts().index))
    
print(DD)


# In[ ]:


df11


# In[86]:


def OLS(Y,Y_hat,N):
    return (1/(2*N)*np.sum((Y-Y_hat)**2))
def R2(Y,Y_hat):
    return (1-(np.sum((Y-Y_hat)**2)/np.sum((Y-np.mean(Y))**2)))


# In[87]:


class OURLinearRegression():
    
    def fit(self, X, y, eta = 1e-3, epochs = 1e3, show_curve = False, lambd = 0, p = 1):
        epochs = int(epochs)
        N, D = X.shape
        Y = y
        
        self.W = np.random.randn(D)
        
        J = np.zeros(epochs)
        
        for epoch in range(epochs):
            Y_hat = self.predict(X)
            J[epoch] = OLS(Y, Y_hat, N)  #(lambd/p*N)*np.linalg.norm(self.W, ord=p, keepdims=True)
            self.W -= eta*(1/N)*(X.T@(Y_hat - Y)) # (1/N)*(lambd*np.abs(self.W)**(p-1)*np.sign(self.W)))
                    
         
        if show_curve:
            plt.figure()
            plt.plot(J)
            plt.xlabel("epochs")
            plt.ylabel("$\mathcal{J}$")
            plt.title("Training Curve")
            plt.show()        
    def predict(self, X):
        return X @ self.W


# In[88]:


myReg=OURLinearRegression()
df11.max()


# In[210]:


plt.scatter( df11["latitude"],df11["longitude"],c=df11["latitude"])
plt.colorbar()


# In[171]:


#del X3,Y3, trainLR_X, testLR_Y
#del Y3
#XA=df12[["Garage"]]

#XA = pd.DataFrame(np.c_[df12["garage"], df12['zipcode']], columns=['garage','zipcode'])
#XA = pd.DataFrame(np.c_[df12["garage"]], columns=['garage'])
#YA = np.transpose(df12["taxes"])
#y= X[:,4]
#XB=X[:,:3]
#y.shape


#XA = df12.iloc[:,6:7].values
  
#YA = df12.iloc[:,3].values

#XA = df12.iloc[:,5:7].values
#YA = df12.iloc[:,3:4].values


#X3=df11[['garage','prediction','taxes']].copy()

###real test commends
"""
X3=df11[['prediction','taxes']].copy()
X3=X3.to_numpy()
X3_normed = X3 / X3.max(axis=0)

Y3=X3[:,1]
X3=X3[:,0:1]
X3 = np.array(X3, dtype=np.float64)
Y3 = np.array(Y3, dtype=np.float64)
"""

train_X1=trainK_X1[:,4:5]
print(trainK_X1)
dev_X1=devK_X1[:,4:5]
test_X1=testK_X1[:,4:5]
"""

##"sumFeatures",'lot_acres','sqrt_ft','taxes','prediction'
XC_normed = XC2 / XC2.max(axis=0)
y_LR= XC_normed[:,3]
X_LR= XC_normed[:,2::2]
X_LR = np.array(X_LR, dtype=np.float64)
y_LR = np.array(y_LR, dtype=np.float64)
"""

"""
import random
X3_normed.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(X3_normed) # shuffles the ordering of filenames (deterministic given the chosen seed)
split_1 = int(0.8 * len(X3_normed))
split_2 = int(0.9 * len(X3_normed))
split_3= int(0.9 * len(X3_normed))
train_filenames = X3_normed[:split_1]
#print(train_filenames.shape)
trainLR_Y= train_filenames[:,2]
trainLR_X= train_filenames[:,0:2]

trainLR_X = np.array(trainLR_X, dtype=np.float64)
trainLR_Y = np.array(trainLR_Y, dtype=np.int64)
#print(trainK_X.shape)


dev_filenames = X3_normed[split_1:split_2]
devLR_Y= dev_filenames[:,2]
devLR_X= dev_filenames[:,0:2]

devLR_X = np.array(devLR_X, dtype=np.float64)
devLR_Y = np.array(devLR_Y, dtype=np.int64)

test_filenames = X3_normed[split_2:]
testLR_Y= test_filenames[:,2]
testLR_X= test_filenames[:,0:2]

testLR_X = np.array(testLR_X, dtype=np.float64)
testLR_Y = np.array(testLR_Y, dtype=np.int64)
"""

#print(X_LR[0])
#print(XC2[0])
print(train_X1.shape)


# In[1217]:


#plt.scatter(df11["taxes"],df11["sqrt_ft"])


# In[172]:


#del X3, trainK_X1
#del Y3
myReg.fit(train_X1,trainK_Y,epochs=1e4,eta=1e-11,show_curve=True)


# In[175]:


#del trainK_X1 train_Y
#del Y3
y_hat=myReg.predict(test_X1)


# In[176]:


#del trainK_Y
#del trainK_X1
R2(test_Y,y_hat)


# In[137]:


#plt.figure()
#plt.scatter(trainK_Y,y_hat,s=8)

fig, ax = plt.subplots(figsize=(12,6))
#fig,ax =plt.subplots(nrows,ncols)
ax.scatter(test_Y, y_hat)
ax.set_xlabel("ActualValues_Tax",fontname="Times New Roman" ,fontsize=30, fontweight="bold")
ax.set_ylabel("PredictedValues_Tax",fontname="Times New Roman" ,fontsize=30, fontweight="bold")
plt.xticks(fontweight="bold",fontsize=12)
plt.yticks(fontweight="bold",fontsize=12)
plt.show()


# In[73]:


##classification
##build classifier()
class KNNClassifier():
    def fit(self,X,y):
        self.X=X
        self.y=y
        
    def predict(self, X, K, epsilon=1e-3):
        N=len(X)
        y_hat= np.zeros(N)
        ## find distance and measure distance of each points
        for i in range(N):
            dist2=np.sum((self.X-X[i])**2,axis=1)
            idxt =np.argsort(dist2)[:K]
            gamma_k=1/(np.sqrt(dist2[idxt])+ epsilon)
            y_hat[i] = np.bincount(self.y[idxt], weights=gamma_k).argmax()
            
        return y_hat
df11.max()


# In[211]:


#import math
#df11["sumFeatures1"]=abs(df11["latitude"]-df11["longitude"])+df11["zipcode"]
#df11["sqrt_ft2"]=df11["sqrt_ft"]/1280
#df11["bedrooms2"]=df11["bedrooms"]/9
#df11["latitude2"]= df11["latitude"]/32
#df11.shape

df11["NewFeatures"].value_counts()


# In[212]:


#del XC1,YC1,trainK_X ,testK_X, trainK_X1, trainK_Y
#del trainK_X

df11.head()
#df11["sumFeatures"]= df["sqrt_ft"]/12808
#(df11["latitude"]/32)
#+(df["sqrt_ft"]/12808)
#str(df11["bedrooms"]/9)

#df11["sumFeatures"]= df11["sqrt_ft"]/12808
XC1=df11[['taxes','NewFeatures','longitude','latitude','sqrt_ft']].copy() ##but the sqrt_ft is used only for linear Regression
#XC1=df12[["NewFeatures",'sqrt_ft','bedrooms',]].copy() 
XC1=XC1.to_numpy()


XC2=XC1

import random
#XC1.sort()
  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(XC1) # shuffles the ordering of filenames (deterministic given the chosen seed)
split_1 = int(0.8 * len(XC1))
split_2 = int(0.9 * len(XC1))
split_3= int(0.9 * len(XC1))
train_filenames = XC1[:split_1]
#print(train_filenames.shape)
#trainLR_Y=train_filenames[:,0]
trainK_Y= train_filenames[:,0]
#trainK_Xsqrft= train_filenames[:,1]
trainK_X= train_filenames[:,1:7] ##the previous number was [:,2:4]
#trainLR_X= train_filenames[:,]

trainK_X = np.array(trainK_X, dtype=np.float64)
trainK_Y = np.array(trainK_Y, dtype=np.int64)
#trainK_Xsqrft= np.array(trainK_Xsqrft, dtype=np.int64)
print(trainK_X.shape)


dev_filenames = XC1[split_1:split_2]
#devLR_Y= dev_filenames[:,0]
dev_Y= dev_filenames[:,0]
#dev_Xsqrft= dev_filenames[:,1]
dev_X= dev_filenames[:,1:7] ##the previous was [:,2:4]

dev_X = np.array(dev_X, dtype=np.float64)
dev_Y = np.array(dev_Y, dtype=np.int64)
#dev_Xsqrft= np.array(dev_Xsqrft, dtype=np.int64)

test_filenames = XC1[split_2:]
#testLR_Y= test_filenames[:,0]
test_Y= test_filenames[:,0]
#test_Xsqrft= test_filenames[:,1]
test_X= test_filenames[:,1:7] ##the previous was [:,2:4]

test_X = np.array(test_X, dtype=np.float64)
test_Y = np.array(test_Y, dtype=np.int64)
#test_Xsqrft= np.array(test_Xsqrft, dtype=np.int64)

##partitions

####
#YC1=XC1[:,0]
#XC1_normed = XC1 / XC1.max(axis=0)
#XC1=XC1[:,1:7]
#XC1 = np.array(XC1, dtype=np.float64)
#YC1 = np.array(YC1, dtype=np.int64)


print(trainLR_Y.shape)
print(XC1[:,0])


# In[216]:


#del XC1, YC1, trainK_X1, trainK_Y, trainK_X, test_X, test_X1, dev_X, dev_X1
Knn= KNNClassifier()
Knn.fit(trainK_X,trainK_Y)
print(len(set(trainK_Y)))


# In[167]:


##train
y_hat_KNN=Knn.predict(trainK_X,15)
yy= y_hat_KNN.reshape(3970,1)

#train_Y=trainLR_Y.reshape(3970,1)

##validation
y_hat_KNN_Val=Knn.predict(dev_X,15)
yy_dev= y_hat_KNN_Val.reshape(496,1)

#dev_Y= devLR_Y.reshape(496,1)

##test
y_hat_KNN_test=Knn.predict(test_X,15)
yy_test= y_hat_KNN_test.reshape(497,1)

#test_Y= testLR_Y.reshape(497,1)
print(y_hat_KNN.shape)
print(yy.shape)
#print(dev_X)


# In[168]:


#df11["prediction"]=Knn.predict(trainK_X,150)
#XC2=np.append(XC2,yy,axis=1)

##train
trainK_X1= np.append(trainK_X,yy,axis=1)
#train_X1= np.append(trainK_X1,train_Y,axis=1)

##validation
devK_X1= np.append(dev_X,yy_dev,axis=1)
#dev_X1= np.append(devK_X1,dev_Y,axis=1)

##test
testK_X1= np.append(test_X,yy_test,axis=1)
#test_X1= np.append(testK_X1,test_Y,axis=1)

#print(XC2.shape)
#print(Knn.predict(XC1,15)[0])
#print(XC2[0])
#print(df11["prediction"])
#print(train_X1)


# In[169]:


#df11["prediction"]=Knn.predict(trainK_X,150)
#XC2=np.append(XC2,yy,axis=1)
#trainK_X1= np.append(trainK_X,yy,axis=1)
#dev_X1= np.append(dev_X,yy_dev,axis=1)
#test_X1= np.append(test_X,yy_test,axis=1)
#print(XC2.shape)
#print(Knn.predict(XC1,15)[0])
#print(XC2[0])
#print(df11["prediction"])
#print(dev_X1)


# In[1366]:


##define accuaracy
def accuaracy(y,y_hat_KNN):
  return np.mean(y==y_hat_KNN)


# In[1367]:


#del YC1
accuaracy(trainK_Y,y_hat_KNN)


# In[1360]:


plt.figure()
#plt.scatter(YC1,y_hat_KNN,s=8)
plt.scatter(XC1[:,0],YC1,c=y_hat_KNN)

#df11.max()


# In[ ]:





# In[67]:


"""
#def conditions_garage(i):
#    if 0<=i["garage"]<=3:
#        val=1
#    if 3<i["garage"]<=5:
#        val=2
#    if 5<i["garage"]<=10:
#        val=3
#    if 10<i["garage"]<=30:
#        val=4
#    return val
"""


def conditions(i):
    if (i["latitude"]<31.5) | (i["longitude"]<(-110.9)):
        val=1
    if (31.5<=i["latitude"]<31.7) | ((-110.9)<=i["longitude"]<(-110.7)):
        val=2
    if (31.7<=i["latitude"]<31.9) | ((-110.7)<=i["longitude"]<(-110.5)):
        val=3
    if (31.9<=i["latitude"]<32.1) | ((-110.5)<=i["longitude"]<(-110.3)):
        val=4
    if (32.1<=i["latitude"]<32.3) | ((-110.3)<=i["longitude"]<(-110.1)):
        val=5
    if (32.3<=i["latitude"]<32.5) | ((-110.1)<=i["longitude"]<(-109.9)):
        val=6
       
    return val
"""
    if 2.5<=i["bedrooms"]<3:
        val=2
    if 3<=i["bedrooms"]<3.5:
        val=3
    if 3.5<=i["bedrooms"]<4:
        val=4
    if 4<=i["bedrooms"]<4.5:
        val=5
    if 4.5<=i["bedrooms"]<5:
        val=6
    if 5<=i["bedrooms"]<5.5:
        val=7
    if 5.5<=i["bedrooms"]<6:
        val=8
    if 6<=i["bedrooms"]<6.5:
        val=9
    if 6.5<=i["bedrooms"]<7:
        val=10
    if 7<=i["bedrooms"]<7.5:
        val=11
    if 7.5<=i["bedrooms"]<8:
        val=12
    if 8<=i["bedrooms"]<8.5:
        val=13
    if 8.5<=i["bedrooms"]<15:
        val=14
    return val

#def conditions_bedrooms(i):#
#    if 0<=i["bedrooms"]<=3:
#        val=1
#    if 3<i["bedrooms"]<12:
#        val=2
#    
#    return val

def conditions_latitude(i):
    if 0<=i["latitude"]<31.7:
        val=1
    if 31.7<=i["latitude"]<32.1:
        val=2
    if 32.1<=i["latitude"]<32.3:
        val=3
    if 32.3<=i["latitude"]<40:
        val=4
    else:
        val=0
    return val

def conditions_sumFeatures(i):
    
    val= int(df["sqrt_ft"]/12808)+int(df11["bedrooms"]/9)+int(df11["latitude"]/32)
    return val
"""


# In[68]:


#df12["garageAA"]=df12.apply(conditions_garage,axis=1)
#df11["sqrt_bed"]=df11.apply(conditions_sqrtFT,axis=1)
#df11["latitude_ADD"]=df11.apply(conditions_latitude,axis=1)
#df11.max()
df11["NewFeatures"]=df11.apply(conditions,axis=1)


# In[69]:


#df14=df11
#df14.head()


# In[ ]:


X=df14.to_numpy()
X1=X[:,8]    
    
#plt.scatter(X[:,10],X[:,12],c=X1,s=8,alpha=0.5)
plt.scatter(X[:,10],X[:,8])
plt.show()


# In[ ]:


df11['prediction']=df11['prediction'].replace(0,1)


# In[ ]:


#df11.head()


# In[ ]:


#import random
#XC.sort()  # make sure that the filenames have a fixed order before shuffling
#random.seed(230)
#random.shuffle(XC) # shuffles the ordering of filenames (deterministic given the chosen seed)
#split_1 = int(0.8 * len(XC))
#split_2 = int(0.9 * len(XC))
#split_3= int(0.9 * len(XC))
#train_filenames = XC[:split_1]
#dev_filenames = XC[split_1:split_2]
#test_filenames = XC[split_2:]


# In[ ]:


#len(dev_filenames)


# In[177]:


#a=latitude , b= bedrooms, c=sqrt_ft, d=garage
#X_test= np.array([[a,b,c,d]])
#X_test= np.array([[30,6,16000,3]])
def pred_result(X_test):
    D= len(X_test)
    #X_test1= X_test[:,0:3]
    y_hat_KNN=Knn.predict(X_test,15)
    yy= y_hat_KNN.reshape(D,1)
    X_test2= np.append(X_test,yy,axis=1)
    X_test2= X_test2[:,4:5]
    y_hat=myReg.predict(X_test2)
    return y_hat
    
    


# In[206]:


pred_result(np.array([[7,-100,28,1500]])) #NewFeature using(longitude,latitude),longitude, latitude, sqrt_ft


# In[200]:


def valA(i,j): #i=latitude, j=longitude
    if (i<31.5) | (j<(-110.9)):
        val=1
    if (31.5<=i<31.7) | ((-110.9)<=j<(-110.7)):
        val=2
    if (31.7<=i<31.9) | ((-110.7)<=j<(-110.5)):
        val=3
    if (31.9<=i<32.1) | ((-110.5)<=j<(-110.3)):
        val=4
    if (32.1<=i<32.3) | ((-110.3)<=j<(-110.1)):
        val=5
    if (32.3<=i<32.5) | ((-110.1)<=j<(-109.9)):
        val=6
    else:
        val=7
    return val


# In[203]:


valA(28, (-100))


# In[ ]:




