from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

data=pd.read_csv(r"G:\study\machine learning\class\vehicle-dataset-from-cardekho\car data.csv")

data.head()
data.describe()

#Make a copy
data_original=data.copy()


#check for null value
data.isnull().sum()
data.dtypes

#Select dependent and independent Variables
X=data.drop(["Selling_Price"],1) 
y=data["Selling_Price"]

#create Dummy Variable
X_dummy = pd.get_dummies(X)


X_train, X_test, y_train, y_test = tts(
    X_dummy,
    y,
    test_size=0.3,
    random_state=0)
 
X_train.shape, X_test.shape


scaler=StandardScaler()
scaler.fit(X_train.fillna(0))

X_train=scaler.fit_transform(X_train)
X_train=pd.DataFrame(X_train)
X_test=scaler.fit_transform(X_test)
X_test=pd.DataFrame(X_test)

##################################################################  Models  ############################################################################

#Random Forest

def run_randomForests(X_train,X_test,y_train,y_test):
    rf=RandomForestRegressor(n_estimators=200,random_state=39,max_depth=4)
    rf.fit(X_train,y_train)
    print('Train set')
    pred=rf.predict(X_train)
    print('Random Forests Mean squared Error :{}'.format(mean_squared_error(y_train,pred)))
    print('Random Forests r2_score :{}'.format(r2_score(y_train,pred)))
    print('Tesst set')
    pred=rf.predict(X_test)
    print('Random Forests Mean Squared error :{}'.format(mean_squared_error(y_test,pred)))
    print('Random Forests r2_score :{}'.format(r2_score(y_test,pred))) 

#Linear Regression
    
def run_linearRegression(X_train,X_test,y_train,y_test):
    lin_reg=LinearRegression()
    lin_reg.fit(X_train,y_train)
    print('Train set')
    pred=lin_reg.predict(X_train)
    print('Linear Regression Mean Squared error :{}'.format(mean_squared_error(y_train,pred)))
    print('Linear Regression r2_score :{}'.format(r2_score(y_train,pred)))
    print('Tesst set')
    pred=lin_reg.predict(X_test)
    print('Linear Regression Mean Squared error :{}'.format(mean_squared_error(y_test,pred)))
    print('Linear Regression r2_score :{}'.format(r2_score(y_test,pred))) 
    
# KNN
    
def run_KNNRegression(X_train,X_test,y_train,y_test):
    knn=KNeighborsRegressor()
    knn.fit(X_train,y_train)
    print('Train set')
    pred=knn.predict(X_train)
    print('KNN Mean Squared error :{}'.format(mean_squared_error(y_train,pred)))
    print('KNN r2_score :{}'.format(r2_score(y_train,pred)))
    print('Tesst set')
    pred=knn.predict(X_test)
    print('KNN Mean Squared error :{}'.format(mean_squared_error(y_test,pred)))
    print('KNN r2_score :{}'.format(r2_score(y_test,pred)))     
    
###################################################### Feature Selection   #######################################################################  

#Without feature seletion

print("Random Forest:")
run_randomForests(X_train.fillna(0),X_test.fillna(0),y_train,y_test)

print("Linear Regression:")
run_linearRegression(X_train.fillna(0),X_test.fillna(0),y_train,y_test)

print("KNN:")
run_KNNRegression(X_train.fillna(0),X_test.fillna(0),y_train,y_test)

    
# forward Feature Selection

sfs1=SFS(RandomForestRegressor(n_jobs=4,n_estimators=10),
         k_features=10,
         forward=True,
         floating=False,
         verbose=2,
         scoring='r2',
         cv=3
         )

sfs1=sfs1.fit(np.array(X_train.fillna(0)),y_train)
select_feat_forward= X_train.columns[list(sfs1.k_feature_idx_)]
select_feat_forward

print("Random Forest")
run_randomForests(X_train[select_feat_forward].fillna(0),X_test[select_feat_forward].fillna(0),y_train,y_test)

print("Linear Regression")
run_linearRegression(X_train[select_feat_forward].fillna(0),X_test[select_feat_forward].fillna(0),y_train,y_test)

print("KNN")
run_KNNRegression(X_train[select_feat_forward].fillna(0),X_test[select_feat_forward].fillna(0),y_train,y_test)


#Backword Feature Selection
#sfs1=SFS(RandomForestRegressor(n_jobs=4),
#         k_features=10,
#         forward=False,
#        floating=False,
#         verbose=1,
#         scoring='r2',
#         cv=3
#         )

#sfs1=sfs1.fit(np.array(X_train.fillna(0)),y_train)
#select_feat_backward= X_train.columns[list(sfs1.k_feature_idx_)]
#select_feat_backward

#print("Random Forest")
#run_randomForests(X_train[select_feat_backward].fillna(0),X_test[select_feat_backward].fillna(0),y_train,y_test)
#print("Linear Regression")
#run_linearRegression(X_train[select_feat_backward].fillna(0),X_test[select_feat_backward].fillna(0),y_train,y_test)
#print("KNN")
#run_KNNRegression(X_train[select_feat_backward].fillna(0),X_test[select_feat_backward].fillna(0),y_train,y_test)


#Feature Selection by Linear Regression Coefficient

sel_=SelectFromModel(LinearRegression())
sel_.fit(scaler.transform(X_train.fillna(0)),y_train)

selected_feat_linear= X_train.columns[(sel_.get_support())]
len(selected_feat_linear)

print("Random Forest")
run_randomForests(X_train[selected_feat_linear].fillna(0),X_test[selected_feat_linear].fillna(0),y_train,y_test)
print("Linear Regression")
run_linearRegression(X_train[selected_feat_linear].fillna(0),X_test[selected_feat_linear].fillna(0),y_train,y_test)
print("KNN")
run_KNNRegression(X_train[selected_feat_linear].fillna(0),X_test[selected_feat_linear].fillna(0),y_train,y_test)


#Univariate Analysis

univariate = f_regression(X_train.fillna(0),y_train)
univariate

univariate = pd.Series(univariate[1])
univariate.index=X_train.columns
univariate.sort_values(ascending=False,inplace=True)

univariate.sort_values(ascending=False).plot.bar(figsize=(20,8))

sel_ =SelectPercentile(f_regression,percentile=10).fit(X_train.fillna(0),y_train)
selected_feat_univariate=X_train.columns[sel_.get_support()]

print("Random Forest")
run_randomForests(X_train[selected_feat_univariate].fillna(0),X_test[selected_feat_univariate].fillna(0),y_train,y_test)
print("Linear Regression")
run_linearRegression(X_train[selected_feat_univariate].fillna(0),X_test[selected_feat_univariate].fillna(0),y_train,y_test)
print("KNN")
run_KNNRegression(X_train[selected_feat_univariate].fillna(0),X_test[selected_feat_univariate].fillna(0),y_train,y_test)


#Mutual Information

mi=mutual_info_regression(X_train.fillna(0),y_train)
mi
mi=pd.Series(mi)
mi.index=X_train.columns
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(20,8))

sel_= SelectPercentile(mutual_info_regression,percentile=10).fit(X_train.fillna(0),y_train)
selected_feat_mutual_information=X_train.columns[sel_.get_support()]

print("Random Forest")
run_randomForests(X_train[selected_feat_mutual_information].fillna(0),X_test[selected_feat_mutual_information].fillna(0),y_train,y_test)
print("Linear Regression")
run_linearRegression(X_train[selected_feat_mutual_information].fillna(0),X_test[selected_feat_mutual_information].fillna(0),y_train,y_test)
print("KNN")
run_KNNRegression(X_train[selected_feat_mutual_information].fillna(0),X_test[selected_feat_mutual_information].fillna(0),y_train,y_test)


############################################################  Grid_Search_CV with Ridge #######################################

ridge = RidgeCV(alphas=np.arange(0.1,2,0.1))
ridge_model = ridge.fit(X_train,y_train)
print('Train set')
pred=ridge_model.predict(X_train)
print('RidgeCV Mean Squared error :{}'.format(mean_squared_error(y_train,pred)))
print('RidgeCV r2_score :{}'.format(r2_score(y_train,pred)))
print('Tesst set')
pred=ridge_model.predict(X_test)
print('RidgeCV Mean Squared error :{}'.format(mean_squared_error(y_test,pred)))
print('RidgeCV r2_score :{}'.format(r2_score(y_test,pred))) 


################################################################  Grid_Search_CV with Lasso #######################################

lasso = LassoCV(alphas=np.arange(0.1,2,0.1))
lasso_model = lasso.fit(X_train,y_train)
print('Train set')
pred=lasso_model.predict(X_train)
print('LassoCV Mean Squared error :{}'.format(mean_squared_error(y_train,pred)))
print('LassoCV r2_score :{}'.format(r2_score(y_train,pred)))
print('Tesst set')
pred=lasso_model.predict(X_test)
print('LassoCV Mean Squared error :{}'.format(mean_squared_error(y_test,pred)))
print('LassoCV r2_score :{}'.format(r2_score(y_test,pred))) 












