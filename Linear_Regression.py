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
    def __init__(self,method = 'normal',epoch = 300):
        self.weights = None
        self.method = method
        self.epoch = epoch
        
    def fit(self,X,y):
        n_samples,n_features = X.shape
        X = np.hstack(np.ones((n_samples,1)),X)
        X_T = X.T
        if self.method == 'normal':
            B = np.linalg.inv(X_T @ X) @ X_T @ y
            self.weights = B
        if self.method == 'gradient':
            y_values = y.values
            X_values = X.values
            
            self.weights = np.zeros(n_features,1)   
            for _ in self.epoch:
                
            

    def predict(self,X):
        X['intercept'] = 1
        cols = ['intercept'] + [col for col in X.columns if col != 'intercept']
        X = X[cols].values
        predictions = X @ self.weights
        return predictions
    
    
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
model = LinearRegression(method='gradient')
model.fit(x_train,y_train)
preds = model.predict(x_test)
R2_score(y_test,preds)
mean_squared_error(y_test,preds)




y = y.values
X = X.values
n_samples,n_features = X.shape
weights = np.zeros((n_features,1))   
for _ in 100:
    predictions = X @ weights
    error = (y - predictions)
    deltaM = error.T @ X
    weights = deltaM * 0.01




