#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# LOAD THE DATASET
# ##https://www.youtube.com/watch?v=MJ1vWb1rGwM

# In[2]:


from sklearn.datasets import load_boston


# In[3]:


boston=load_boston()


# In[4]:


boston.keys()


# In[5]:


#lets see the description of the dataset


# In[6]:


print(boston.DESCR)


# In[7]:


print(boston.data)


# In[8]:


print(boston.target)


# In[9]:


print(boston.feature_names)


# PREPATING DATASET

# In[10]:


df=pd.DataFrame(boston.data,columns=boston.feature_names)


# In[11]:


df.head()


# In[12]:


df['price']=boston.target


# In[13]:


df.head()


# In[14]:


df.info()


# In[15]:


#describing the stats
df.describe()


# In[16]:


#check the missing values
df.isnull().sum()


# In[17]:


#exploratory data analaysis
#coreltion is important to find in any Lin Reg
#check1-dep and indep var are linear-correlated nor not
df.corr()


# In[18]:


import seaborn as sns
sns.pairplot(df)


# In[19]:


plt.scatter(df["CRIM"],df['price'])
plt.xlabel("crime rate")
plt.ylabel("price")


# In[20]:


#-CORR


# In[21]:


plt.scatter(df["LSTAT"],df['price'])
plt.xlabel("LSTATS")
plt.ylabel("price")


# In[22]:


##+CORR


# In[23]:


sns.regplot(x="RM",y="price",data=df)


# In[24]:


sns.regplot(x="LSTAT",y="price",data=df)


# In[25]:


##-corr


# In[26]:


sns.regplot(x="CHAS",y="price",data=df)


# In[27]:


##no corr


# In[28]:


sns.regplot(x="PTRATIO",y="price",data=df)


# In[29]:


##-CORR


# In[30]:


##divide fearutes into dep,indep


# In[31]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[32]:


x.head()


# In[33]:


y.head()


# In[34]:


##train test split


# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=95)


# In[36]:


x_train


# In[37]:


y_train


# In[38]:


x_test


# In[39]:


y_test


# In[40]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[41]:


x_train=scaler.fit_transform(x_train)


# In[42]:


x_test=scaler.transform(x_test)


# MODEL TRAINING

# In[43]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()


# In[80]:


reg.fit(x_train,y_train)


# In[45]:


##print coeff


# In[46]:


print(reg.coef_)


# In[47]:


print(reg.intercept_)


# In[48]:


#on which params model is trained
reg.get_params()


# In[49]:


##prediction with test data
reg_pred=reg.predict(x_test)


# In[50]:


reg_pred


# In[51]:


##plot scatter plot for pred
#graphs looks like +corr
# model is performing good as predicted values follow the actual test values and forms linear pattern
plt.scatter(y_test,reg_pred)


# In[52]:


#residuals is actual-pred
resd=y_test-reg_pred


# In[53]:


resd


# In[54]:


sns.displot(resd,kind="kde")
# a KDE plot smooths the observations with a Gaussian kernel, producing a continuous density estimate:
#histplot
#Plot a histogram of binned counts with optional normalization or smoothing.

#kdeplot
#Plot univariate or bivariate distributions using kernel density estimation.


# In[55]:


sns.displot(resd,kind="hist")
#we can see ouliers but data is normal distribution
#check2-errors should be normal dist


# In[56]:


# scatter plot for residuals and pred
plt.scatter(resd,reg_pred)
# no format or relation-i.e uniformly distibuted means model is performing good for residuls/errors and pred


# In[81]:


#performance metrics
#np=numpy square fun
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# R SQUARE AND ADJUSTED R SQUARE,
# r SQURE VALUE SHOULD BE HIGH FOR GOOD MODEL
# adj r-square:1-[(1-r2)(N-1)/N-p-1],
# p=features/pred,N=no.of dt pts
# 

# In[58]:


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)


# In[59]:


1-((1-score)*(len(y_test)-1))/(len(y_test)-x_test.shape[1]-1)
# adj r2 should be less than r2


# new data pred- while pred we usually have 1 dt pt
# $-so in shape we have columns buts rows have nothing so we reshape it to get (1,13)

# In[60]:


boston.data[0]


# In[61]:


boston.data[0].shape


# In[62]:


boston.data[0].reshape(1,-1)


# In[63]:


boston.data[0].reshape(1,-1).shape


# In[64]:


#transform new data-standard scaler
scaler.transform(boston.data[0].reshape(1,-1))


# In[65]:


reg.predict(scaler.transform(boston.data[0].reshape(1,-1)))


# In[69]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
mse=cross_val_score(lin_reg,x_train,y_train,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# In[76]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lin_reg=LinearRegression()
lin_reg.fit(x_test,y_test)
mse=cross_val_score(lin_reg,x_test,y_test,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# In[77]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV##FOR hyper para tuning


# In[83]:


ridge=Ridge()
params={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-1,1,5,10,20,30,40,45,50,60,70,80,90,100]}
ridge_regressor=GridSearchCV(ridge,params,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x_train,y_train)


# In[72]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[78]:


ridge=Ridge()
params={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-1,1,5,10,20,30,40,45,50,60,70,80,90,100]}
ridge_regressor=GridSearchCV(ridge,params,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x_test,y_test)


# In[79]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[73]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV#


# In[74]:


lasso=Lasso()
params={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-1,1,5,10,20,30,40,45,50,60,70,80,90,100]}
lasso_regressor=GridSearchCV(lasso,params,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(x_train,y_train)


# In[75]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[89]:


def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv=5)).mean()
    return rmse
    

def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return mae, mse, rmse, r_squared


# In[91]:


lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
predictions = lin_reg.predict(x_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(lin_reg)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "LinearRegression","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
#models = models.append(new_row, ignore_index=True)


# In[94]:


ridge = Ridge()
ridge.fit(x_train, y_train)
predictions = ridge.predict(x_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(ridge)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "Ridge","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
#models = models.append(new_row, ignore_index=True)


# In[96]:


lasso = Lasso()
lasso.fit(x_train, y_train)
predictions = lasso.predict(x_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(lasso)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "Lasso","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
#models = models.append(new_row, ignore_index=True)


# In[99]:


from sklearn.svm import SVR
#from xgboost import XGBRegressor


# In[101]:


svr = SVR(C=100000)
svr.fit(x_train, y_train)
predictions = svr.predict(x_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(svr)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "SVR","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
#models = models.append(new_row, ignore_index=True)


# In[102]:


from xgboost import XGBRegressor


# In[ ]:


xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01)
xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)

mae, mse, rmse, r_squared = evaluation(y_test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r_squared)
print("-"*30)
rmse_cross_val = rmse_cv(xgb)
print("RMSE Cross-Validation:", rmse_cross_val)

new_row = {"Model": "XGBRegressor","MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r_squared, "RMSE (Cross-Validation)": rmse_cross_val}
models = models.append(new_row, ignore_index=True)

