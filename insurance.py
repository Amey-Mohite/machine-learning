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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 

data = pd.read_csv(r"C:\Users\Amey\Desktop\insurance3r2.csv")
data.describe()
data.shape
data.head()

#Make a copy
data_original=data.copy()
data.isnull().sum()
data.dtypes

##Univariate analysis

##Histogram for nomarlity
sns.distplot( data["age"] , color="skyblue")
sns.distplot( data["bmi"] , color="olive")
sns.distplot( data["steps"] , color="gold")
sns.distplot( data["charges"] , color="teal")

#Boxplot for outliers
sns.boxplot( y=data["age"] , color="skyblue")
sns.boxplot( y=data["bmi"] , color="olive")
sns.boxplot( y=data["steps"] , color="gold")
sns.boxplot( y=data["charges"] , color="teal")
sns.countplot(x='insuranceclaim',data=data)

##Biviarate analysis
sns.catplot(x="insuranceclaim", y="age", hue="sex", kind="swarm", data=data);
sns.stripplot(x = "insuranceclaim", y = "bmi", data = data)
#can add mre plotss

#analyis

X_train, X_test, y_train, y_test = tts(
    data.drop(labels=['insuranceclaim'], axis=1),
    data['insuranceclaim'],
    test_size=0.3,
    random_state=0)
 
X_train.shape, X_test.shape

X_train_original = X_train.copy()
X_test_original = X_test.copy()

#Random Forest
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
    print("Logistic: train set")
    y_pred = logit.predict(X_train)
    print("Logistic:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
    print ("Logistic:Accuracy : ", accuracy_score(y_train,y_pred)*100)
    print("Logistic: test set")
    y_pred=logit.predict(X_test)
    matrix =confusion_matrix(y_test, y_pred)   
    print("logistic Confusion_matrix",matrix)
    print ("Logistic:Accuracy : ", accuracy_score(y_test,y_pred)*100)
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
    print("Logistic:Accuracy:",accuracy_score(y_test, y_pred))
    y_pred_proba = logit.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="ROC, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    
#Decision Tree
def run_decision_tree(X_train,X_test,y_train,y_test):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5) 
    clf_entropy.fit(X_train, y_train)  
    print("Decision_tree:train set")
    y_pred = clf_entropy.predict(X_train)
    print("Decision_tree:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
    print ("Decision_tree:Accuracy : ", accuracy_score(y_train,y_pred)*100)
    print("Decision_tree:test set")
    y_pred = clf_entropy.predict(X_test)
    print("Decision_tree:Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print ("Decision_tree:Accuracy : ", accuracy_score(y_test,y_pred)*100)

def run_KNN(X_train,X_test,y_train,y_test):
    knn=KNeighborsClassifier()
    knn.fit(X_train,y_train)
    print("KNN:train set")
    y_pred = knn.predict(X_train)
    print("KNN:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
    print ("KNN:Accuracy : ", accuracy_score(y_train,y_pred)*100)
    print("KNN:train set")
    y_pred = knn.predict(X_test)
    print("KNN:Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print ("KNN:Accuracy : ", accuracy_score(y_test,y_pred)*100)    
        
# original
scaler = StandardScaler().fit(X_train_original)
run_logistic(scaler.transform(X_train_original), scaler.transform(X_test_original),y_train, y_test)
run_KNN(X_train_original,X_test_original,y_train,y_test)
run_decision_tree(X_train_original,X_test_original,y_train,y_test)
run_randomForests(X_train_original,X_test_original,y_train,y_test)
#PCA
#5 components
data2=data.drop(labels=['insuranceclaim'],axis=1)
data2 = StandardScaler().fit_transform(data2)
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(data2)
principalDf = pd.DataFrame(data = principalComponents,columns = ['PC1', 'PC2','PC3','PC4','PC5'])
pcadata = pd.concat([principalDf, data[['insuranceclaim']]], axis = 1)
X_train, X_test, y_train, y_test = tts(
    pcadata.drop(labels=['insuranceclaim'], axis=1),
    pcadata['insuranceclaim'],
    test_size=0.3,
    random_state=0)
#scaler = StandardScaler().fit(X_train)
#run_logistic(scaler.transform(X_train), scaler.transform(X_test),y_train, y_test)
run_logistic(X_train,X_test,y_train, y_test)

#7 components
data2=data.drop(labels=['insuranceclaim'],axis=1)
data2 = StandardScaler().fit_transform(data2)
pca = PCA(n_components=7)
principalComponents = pca.fit_transform(data2)
principalDf = pd.DataFrame(data = principalComponents,columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','PC7'])
pcadata = pd.concat([principalDf, data[['insuranceclaim']]], axis = 1)
X_train, X_test, y_train, y_test = tts(
    pcadata.drop(labels=['insuranceclaim'], axis=1),
    pcadata['insuranceclaim'],
    test_size=0.3,
    random_state=0)
#scaler = StandardScaler().fit(X_train)
#run_logistic(scaler.transform(X_train), scaler.transform(X_test),y_train, y_test)
run_logistic(X_train,X_test,y_train, y_test)
run_KNN(X_train,X_test,y_train,y_test)
run_decision_tree(X_train,X_test,y_train,y_test)
run_randomForests(X_train,X_test,y_train,y_test)
#####
#forward stewise selection
X_train = X_train_original.copy()
X_test = X_test_original.copy()
sfs1=SFS(RandomForestClassifier(n_jobs=4),
         k_features=4,
         forward=True,
         floating=False,
         verbose=2,
         scoring='roc_auc',
         cv=3
         )

sfs1=sfs1.fit(np.array(X_train),y_train)
select_feat= X_train.columns[list(sfs1.k_feature_idx_)]
select_feat
run_logistic(X_train[select_feat],X_test[select_feat],y_train, y_test)
run_randomForests(X_train[select_feat],X_test[select_feat],y_train, y_test)
run_KNN(X_train[select_feat],X_test[select_feat],y_train,y_test)
run_decision_tree(X_train[select_feat],X_test[select_feat],y_train,y_test)



































