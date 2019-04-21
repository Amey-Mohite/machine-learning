import pandas as pd
df=pd.read_csv("G:\study\machine learning\class\python\diabetes.csv")
df.head()

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts

X = df.drop(["diabetes"],1) 
y=df["diabetes"]
X_train,X_test,y_train,y_test=tts(X,y,test_size=0.3,random_state=42)
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
accuracy_score(y_test,y_pred)
from sklearn.model_selection import cross_val_score
cross_val_score(knn,X,y,cv=10).mean()

from sklearn.model_selection import GridSearchCV
import numpy as np
params={"n_neighbors":np.array([1,2,3,4,5,6,7,8,9,])}
knn_cv=GridSearchCV(knn,param_grid=params)
knn_cv.fit(X,y)
print(knn_cv.best_params_)
print(knn_cv.best_score_)


from sklearn.metrics import accuracy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split as tts



X1=df.drop(["age"],1)
y1=df["age"]

X1_train,X1_test,y1_train,y1_test=tts(X1,y1,test_size=0.3,random_state=42)
knn=KNeighborsRegressor()
knn.fit(X1_train,y1_train)
y1_pred=knn.predict(X1_test)
accuracy_score(y1_test,y1_pred)
from sklearn.model_selection import cross_val_score
cross_val_score(knn,X1,y1,cv=10).mean()

from sklearn.model_selection import GridSearchCV
import numpy as np
params={"n_neighbors":np.array([1,2,3,4,5,6,7,8,9])}
knn_cv=GridSearchCV(knn,param_grid=params)
knn_cv.fit(X1,y1)
print(knn_cv.best_params_)
print(knn_cv.best_score_)



from sklearn.preprocessing import MinMaxScaler
col=list(X)
scaler=MinMaxScaler()
X=


knn=KNeighborsRegressor()
cross_val_score(knn,X,y,cv=10,scoring="r2")

df2=pd.concat(X,y)
df_1=df.copy()
df2=df2.drop(["diabetes"],1)
import seaborn as  