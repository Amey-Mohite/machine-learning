import pandas as pd 
import numpy as np

data=pd.read_csv("G:\study\machine learning\class\python\gapminder.csv")
from sklearn.linear_model import LinearRegression,Ridge,LassoCV,ElasticNetCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.model_selection import train_test_split as tts 
lin_reg=LinearRegression()

X=data.drop(["life"],1)
y=data["life"]
X=pd.get_dummies(X)

X_train,X_test,y_train,y_test = tts(X,y,test_size = 0.2, random_state=42)
