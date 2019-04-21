import pandas as pd 
import numpy as np

data=pd.read_csv("G:\study\machine learning\class\python\gapminder.csv")
numerial_data=data.select_dtypes(include=np.number)
import seaborn as sns
sns.heatmap(numerial_data.corr(),annot = True)
data.head()
data.isnull().sum()
features_by_filter_method=["HIV","GDP","BMI_female","child_mortality"]
X=data.drop(["life"],1)
y=data["life"]
X=pd.get_dummies(X)
from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
col_names=list(X)
X=scalar.fit_transform(X)
X=pd.DataFrame(X)
X.columns=col_names
X2=X1
X3=data[data["Region"]]
X2=pd.concat(X1,X3)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import cross_val_score,GridSearchCV

from sklearn.neighbors import KNeighborsRegressor
knn =  KNeighborsRegressor()
from sklearn.model_selection import train_test_split as tts 
X1=X[features_by_filter_method]

#cross validation 
cross_val_score(knn,X1,y,cv=10,scoring="r2")

cross_val_score(knn,X,y,cv=10,scoring="r2")

params={"n_neighbors":np.arange(1,16)}
params
knn_cv=GridSearchCV(knn,param_grid=params,cv=10,scoring="r2")


#grid search with all variables
knn_cv.fit(X,y)
knn_cv.best_params_
knn_cv.best_score_
best_model=knn_cv.best_estimator_

#gridsearch with feature selection
knn_cv1=GridSearchCV(knn,param_grid=params,cv=10,scoring="r2")
knn_cv1.fit(X1,y)
knn_cv1.best_params_
knn_cv1.best_score_
best_model1=knn_cv1.best_estimator_



#normal KNN
X_train,X_test,y_train,y_test = tts(X,y,test_size = 0.2, random_state=42)
X_train.shape
X_test.shape
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
mean_squared_error(y_test,y_pred)
np.sqrt(mean_squared_error(y_test,y_pred))
r2_score(y_test,y_pred)
