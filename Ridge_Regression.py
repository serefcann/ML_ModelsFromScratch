import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

root_mean_squared_error = lambda y,pred: np.sqrt(np.mean((y - pred) ** 2))
def r_squared(y,pred):
    ssr = ((y-pred)**2).sum()
    sst = ((y-np.mean(y))**2).sum()
    return (1 - (ssr/sst)).values[0]


class Ridge_Regression():
    def __init__(self,lamda = 2):
        self.lamda = lamda
        self.weights = 0
    
    def fit(self,xtrain,ytrain):      
        xtrain = np.hstack([np.ones((xtrain.shape[0],1)),xtrain]) # add intercept to xtrain
        n_samples,n_features =  xtrain.shape
        I = np.identity(n_features) # identity matrix
        I[0][0] = 0 # not to penalize intercept
        
        penalization = (self.lamda * I)
        B = np.linalg.inv(xtrain.T @ xtrain + penalization) @ xtrain.T @ ytrain
        self.weights = B
        
    def predict(self,xtest):
        xtest = np.hstack([np.ones((xtest.shape[0],1)),xtest])
        predicts = xtest @ self.weights
        return predicts
    
data = pd.read_csv("C:\\Users\\şerefcanmemiş\\Downloads\\teams.csv")
data.head()
X = data[['age','athletes']].copy()
y = data[['medals']].copy()

xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2)

# standardization is must because of that large numbers of xtrain compared to lambda value
train_mean = np.mean(xtrain)
train_std = np.std(xtrain)
xtrain_scaled = (xtrain - train_mean) / train_std # scale xtrain
xtest_scaled = (xtest - train_mean) / train_std # scale xtest with mean xtrain and std xtrain

model = Ridge_Regression()
model.fit(xtrain = xtrain_scaled, ytrain = ytrain)
ridge_predicts = model.predict(xtest_scaled)

root_mean_squared_error(ytest,ridge_predicts) # rmse 44.12
r_squared(ytest,ridge_predicts) # R^2 0.76


# Lets find the best lambda for our Ridge Regression model with given data
lambdas = np.logspace(-3, 8, 100)  # 0.001 to 10000
list_rmse = []
for lamda in lambdas:
    model = Ridge_Regression(lamda = lamda)
    model.fit(xtrain = xtrain_scaled, ytrain = ytrain)
    ridge_predicts = model.predict(xtest_scaled)
    list_rmse.append(root_mean_squared_error(ytest,ridge_predicts))
best_lambda = lambdas[np.argmin(list_rmse)]

model = Ridge_Regression(best_lambda)
model.fit(xtrain = xtrain_scaled, ytrain = ytrain)
ridge_predicts = model.predict(xtest_scaled)

root_mean_squared_error(ytest,ridge_predicts) # rmse 17.60
r_squared(ytest,ridge_predicts) # R^2 0.94



# For test purposes
from sklearn.linear_model import Ridge
model = Ridge(alpha=2).fit(xtrain_scaled,ytrain)
print((model.predict(xtest_scaled) - ridge_predicts).sum()) # for test my model result (so close)