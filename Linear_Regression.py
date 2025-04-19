import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:\\Users\\şerefcanmemiş\\Downloads\\teams.csv")
data.head()

X = data[['age','athletes']].copy()
y = data[['medals']].copy()

def mean_squared_error(y,predictions):
    SSR = ((y - predictions)**2).sum()
    SSR = SSR.values[0]
    return SSR

def R2_score(y,predictions):
    SSR = ((y - predictions)**2).sum()
    SSR = SSR.values[0]
    SST = ((y - np.mean(y))**2).sum()
    SST = SST.values[0]
    R2 = 1 - (SSR/SST)
    return R2

class LinearRegression():
    def __init__(self):
        self.weights = None
        
    def fit(self,X,y):
        X['intercept'] = 1
        cols = ['intercept'] + [col for col in X.columns if col != 'intercept']
        X = X[cols]
        X_T = X.T
        B = np.linalg.inv(X_T @ X) @ X_T @ y
        self.weights = B

    def predict(self,X):
        X['intercept'] = 1
        cols = ['intercept'] + [col for col in X.columns if col != 'intercept']
        X = X[cols].values
        predictions = X @ self.weights
        return predictions
    
    
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
model = LinearRegression()
model.fit(x_train,y_train)
preds = model.predict(x_test)
R2_score(y_test,preds)
mean_squared_error(y_test,preds)












