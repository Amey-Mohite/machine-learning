import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import roc_auc_score,classification,roc_curve
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import f_classif,f_regression
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import tree
data = pd.read_csv(r"C:\Users\Amey\Desktop\finance.csv")  
data.shape
data.head()
data.describe()

#Make a copy
data_original=data.copy()

#check for null value
data.isnull().sum()
data.dtypes

data['y']=np.where(data['y']=='yes',1,0)


numerical_data=data.select_dtypes(include=np.number)
categorical_data = data.select_dtypes(include=np.object)

categorical_data_dummy = pd.get_dummies(categorical_data)

data_final = pd.concat([numerical_data,categorical_data_dummy],axis=1) 

X_train, X_test, y_train, y_test = tts(
    data_final.drop(labels=['y'], axis=1),
    data_final['y'],
    test_size=0.3,
    random_state=0)
 
X_train.shape, X_test.shape

X_train_original = X_train.copy()
X_test_original = X_test.copy()

###########################################################################
#random forest
def run_randomForests(X_train,X_test,y_train,y_test):
    rf=RandomForestClassifier(n_estimators=200,random_state=39,max_depth=4)
    rf.fit(X_train,y_train)
    print('Train set')
    y_pred=rf.predict(X_train)
    pred=rf.predict_proba(X_train)
    print('Random Forests roc_auc :{}'.format(roc_auc_score(y_train,pred[:,1])))
    print("Accuracy For Random Forest:",accuracy_score(y_train, y_pred))
    print('Test set')
    y_pred=rf.predict(X_test)
    pred=rf.predict_proba(X_test)
    print('Random Forests roc_auc :{}'.format(roc_auc_score(y_test,pred[:,1])))
    print("Accuracy For Random Forest:",accuracy_score(y_test, y_pred))
    

#logistic Regresssion
def run_logistic(X_train, X_test, y_train, y_test):
    logit = LogisticRegression()
    logit.fit(X_train,y_train)
    y_pred=logit.predict(X_test)
    matrix =confusion_matrix(y_test, y_pred)   
    print("Confusion_matrix",matrix)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    print("Accuracy:",accuracy_score(y_test, y_pred))
    y_pred_proba = logit.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="ROC, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    
    

        
# original
scaler = StandardScaler().fit(X_train)
run_logistic(scaler.transform(X_train), scaler.transform(X_test),y_train, y_test)
######################################################################################################
#forward step
sfs1=SFS(RandomForestClassifier(n_jobs=4),
         k_features=10,
         forward=True,
         floating=False,
         verbose=2,
         scoring='roc_auc',
         cv=3
         )

sfs1=sfs1.fit(np.array(X_train),y_train)
select_feat= X_train.columns[list(sfs1.k_feature_idx_)]
select_feat
run_randomForests(X_train[select_feat],X_test[select_feat],y_train,y_test)

#####################################################################################################
#mutual information
mi=mutual_info_classif(X_train.fillna(0),y_train)

mi=pd.Series(mi)

mi.index=X_train.columns

mi.sort_values(ascending=False)

mi.sort_values(ascending=False).plot.bar(figsize=(20,8))

sel_= SelectKBest(mutual_info_classif,k=10).fit(X_train.fillna(0),y_train)
sum(sel_.get_support())

X_train.columns[sel_.get_support()]

X_train_mutual=sel_.transform(X_train)

X_test_mutual=sel_.transform(X_test)


run_randomForests(X_train_mutual,X_test_mutual,y_train,y_test)    

scaler = StandardScaler().fit(X_train_mutual)
 
run_logistic(scaler.transform(X_train_mutual), scaler.transform(X_test_mutual),y_train, y_test)

####################################################################################################
#ANNOVA
univariate = f_classif(X_train,y_train)

univariate = pd.Series(univariate[1])

univariate.index=X_train.columns

univariate.sort_values(ascending=False,inplace=True)

univariate.sort_values(ascending=False).plot.bar(figsize=(20,8))

sel_ = SelectKBest(f_classif,k=10).fit(X_train,y_train)

X_train.columns[sel_.get_support()]

X_train_univariate=sel_.transform(X_train)

X_test_univariate=sel_.transform(X_test)

run_randomForests(X_train_univariate,X_test_univariate,y_train,y_test)    

scaler = StandardScaler().fit(X_train_univariate)
run_logistic(scaler.transform(X_train_univariate), scaler.transform(X_test_univariate),y_train, y_test)











