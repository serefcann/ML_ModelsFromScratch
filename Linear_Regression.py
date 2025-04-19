import numpy as np
import pandas as pd

data = pd.read_csv("C:\\Users\\şerefcanmemiş\\Downloads\\teams.csv")
data.head()

X = data[['age','athletes']].copy()
y = data[['medals']].copy()

def mse(y,predictions):
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
    
model = LinearRegression()
model.fit(X,y)
preds = model.predict(X)
R2_score(y,preds)
mse(y,preds)












