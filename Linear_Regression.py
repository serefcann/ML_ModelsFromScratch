import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    def __init__(self,method = 'normal',epoch = 300,learning_rate = 0.01):
        self.weights = None
        self.method = method
        self.epoch = epoch
        self.learning_rate = learning_rate
        
    def fit(self,X,y):
        if isinstance(X,pd.DataFrame): X = X.values
        if isinstance(y,pd.DataFrame): y = y.values
        n_samples,n_features = X.shape
        X = np.hstack([np.ones((n_samples,1)),X])
        X_T = X.T
        
        if self.method == 'normal':
            B = np.linalg.inv(X_T @ X) @ X_T @ y
            self.weights = B
            print(B)
            
        if self.method == 'gradient':
            self.weights = np.zeros((n_features + 1,1))
            for _ in range(self.epoch):
                y_predicted = X @ self.weights
                error = (y - y_predicted)
                dm = (-2/n_samples) * (X.T @ error)
                self.weights -= self.learning_rate * dm

    def predict(self,X):
        n_samples,n_features = X.shape
        X = np.hstack([np.ones((n_samples,1)),X])
        predictions = X @ self.weights
        return predictions
    
# Scale is must otherwise gradient descent performs decrease
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)

model = LinearRegression(method='normal')
model.fit(x_train,y_train)
preds = model.predict(x_test)
R2_score(y_test,preds)
mean_squared_error(y_test,preds)

model = LinearRegression(method='gradient')
model.fit(x_train,y_train)
preds = model.predict(x_test)
R2_score(y_test,preds)
mean_squared_error(y_test,preds)
