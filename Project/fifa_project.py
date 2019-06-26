import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as sm
from scipy.stats import poisson
import random
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
#Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

result=pd.read_csv("results.csv")
ranking=pd.read_csv("ranking.csv",encoding='latin-1')

""" Data Cleaning """

def meanWeighted(l,year,country):
    if l>2:
        currentYear = int(ranking.ix[(ranking['country_full']==country) & (ranking['year']==[year])]['mean_pts'].head(1))
        previousYear = int(ranking.ix[(ranking['country_full']==country) & (ranking['year']==[year-1])]['mean_pts'].head(1))
        previous2Year = int(ranking.ix[(ranking['country_full']==country) & (ranking['year']==[year-2])]['mean_pts'].head(1))
        weighted = (0.5*currentYear)+(0.3*previousYear)+(0.2*previous2Year)
    elif l==2 and year>=2012:
        currentYear = int(ranking.ix[(ranking['country_full']==country) & (ranking['year']==[year])]['mean_pts'].head(1))
        previousYear = int(ranking.ix[(ranking['country_full']==country) & (ranking['year']==[year-1])]['mean_pts'].head(1))
        weighted = (0.6*currentYear)+(0.4*previousYear)
    else:
        weighted = int(ranking[(ranking['year']==year)]['mean_pts'].head(1))
    return weighted

countries = ranking.country_full.unique().tolist()
countries.sort()

result = result[result.home_team.isin(countries)]
result = result[result.away_team.isin(countries)]

#set(result.away_team.unique().tolist()).difference(set(result.home_team.unique().tolist()))

ranking['pts_diff'] = round(ranking['total_points'] - ranking['previous_points'])
ranking['rank_date'] = pd.to_datetime(ranking['rank_date'], format='%d-%m-%Y')
ranking.rename(columns={'rank_date':'date'},inplace=True)
ranking['day'] = ranking['date'].dt.day
ranking['month'] = ranking['date'].dt.month
ranking['year'] = ranking['date'].dt.year
ranking['mean_pts'] = ranking.groupby(['country_full','year']).transform(lambda x: x.mean())['total_points']
ranking['mean_weighted'] = 0
for country in countries:
    years = ranking.ix[ranking['country_full']==country]['year'].unique().tolist()
    years.sort(reverse=True)
    l = len(years)
    for year in years:
        ranking['mean_weighted'] = np.where((ranking['year']==year) & (ranking['country_full']==country),meanWeighted(l,year,country),ranking['mean_weighted'])
        l -= 1

current_ranking = ranking[ranking['date']=='2019-06-14']
ranking.drop(['date','previous_points'],axis=1,inplace=True)

result['date'] = pd.to_datetime(result['date'], format='%d-%m-%Y')
result['day'] = result['date'].dt.day
result['month'] = result['date'].dt.month
result['year'] = result['date'].dt.year

result['results'] = np.where(result['home_score']>result['away_score'],0,np.where(result['home_score']==result['away_score'],1,2))
result['impt'] = np.where(result['tournament']=='Friendly',0,1)
result['host'] = np.where((result['country']==result['home_team']) | (result['country']==result['away_team']),1,0)
result.drop(['date','city','tournament','country'],axis=1,inplace=True)


"""    
mean = pd.DataFrame(ranking.groupby(['country_full','year'])['total_points'].mean())
mean.reset_index(inplace=True)
mean.rename(columns={'total_points':'mean_pts'},inplace=True)
result = pd.merge(left=result, right=mean, how='left', left_on=['home_team','year'], right_on=['country_full','year']).drop(['country_full'],axis=1)
result = pd.merge(left=result, right=mean, how='left', left_on=['away_team','year'], right_on=['country_full','year']).drop(['country_full'],axis=1)
result.rename(columns={'mean_pts_x':'home_mean','mean_pts_y':'away_mean'},inplace=True)
result = result.fillna(0)
result['mean_diff'] = result['home_mean'] - result['away_mean']
result.drop(['home_mean','away_mean'],axis=1,inplace=True)
"""

result = pd.merge(left=result, right=ranking, how='left', left_on=['home_team','year','month'], right_on=['country_full','year','month'], suffixes=('_x','_y')).drop(['country_full','day_y','total_points','confederation'],axis=1)
result = pd.merge(left=result, right=ranking, how='left', left_on=['away_team','year','month'], right_on=['country_full','year','month'], suffixes=('_x','_y')).drop(['country_full','day','total_points','confederation'],axis=1)
result.rename(columns={'rank_x':'home_rank','rank_y':'away_rank','pts_diff_x':'home_pts_diff','pts_diff_y':'away_pts_diff','mean_pts_x':'home_mean','mean_pts_y':'away_mean','mean_weighted_x':'home_weighted','mean_weighted_y':'away_weighted'},inplace=True)
result.update(result[['home_rank','away_rank','home_pts_diff','away_pts_diff','home_mean','away_mean','home_weighted','away_weighted']].fillna(0))
#result[['home_rank','away_rank','home_pts_diff','away_pts_diff']] = result[['home_rank','away_rank','home_pts_diff','away_pts_diff']].fillna(0)
result['rank_diff'] = result['home_rank'] - result['away_rank']
result['mean_diff'] = result['home_mean'] - result['away_mean']
result['weighted_diff'] = result['home_weighted'] - result['away_weighted']
result.drop(['home_rank','away_rank','home_mean','away_mean','home_weighted','away_weighted'],axis=1,inplace=True)



"""
Predicting match results as 
Win/Draw/Lose
"""

features = result.loc[0:,['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff']]
labels = result.loc[:,['results']]

sc = StandardScaler()
features = sc.fit_transform(features)
x_train,x_test,y_train,y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
x_train = pd.DataFrame(x_train,columns=['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff'])
x_test = pd.DataFrame(x_test,columns=['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff'])

k_fold = KFold(n_splits=5, shuffle=True, random_state=0)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
Logistic_Reg = dict()
#param_grid = dict(C=(0.0001,0.001,0.005,0.01,0.1,0.5,1))
#lr_class = GridSearchCV(LogisticRegression(),cv=5,param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
lr_class = LogisticRegression(C=0.001,n_jobs=-1)
lr_class.fit(x_train,y_train)
#print(lr_class.best_params_)
res_predict = lr_class.predict(x_test)
train_predict = lr_class.predict(x_train)
#Logistic_Reg['train_cm'] = confusion_matrix(y_train,train_predict)
#Logistic_Reg['test_cm'] = confusion_matrix(y_test,res_predict)
Logistic_Reg['Train Accuracy'] = round(np.mean(cross_val_score(lr_class,x_train,y_train,cv=k_fold,scoring="accuracy")),2)
Logistic_Reg['Test Accuracy'] = round(accuracy_score(y_test,res_predict),2)
Logistic_Reg['Train Precision'] = round(precision_score(y_train,train_predict,average='macro'),2)
Logistic_Reg['Test Precision'] = round(precision_score(y_test,res_predict,average='macro'),2)
Logistic_Reg['Train F1-Score'] = round(f1_score(y_train,train_predict,average='macro'),2)
Logistic_Reg['Test F1-Score'] = round(f1_score(y_test,res_predict,average='macro'),2)

#SVM RBF
from sklearn.svm import SVC
SVM = dict()
#param_grid = dict(C=(0.001,0.01,0.1,0.5,1,2),gamma=(0.001,0.01,0.1,0.5,1,2))
#svm_class = GridSearchCV(SVC(kernel="rbf"),cv=5,param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
svm_class = SVC(kernel='rbf',gamma=0.001,C=2,probability=True)
svm_class.fit(x_train,y_train)
#print(svm_class.best_params_)
res_predict = svm_class.predict(x_test)
train_predict = svm_class.predict(x_train)
#SVM['train_cm'] = confusion_matrix(y_train,train_predict)
#SVM['test_cm'] = confusion_matrix(y_test,res_predict)
SVM['Train Accuracy'] = round(np.mean(cross_val_score(svm_class,x_train,y_train,cv=k_fold,scoring="accuracy")),2)
SVM['Test Accuracy'] = round(accuracy_score(y_test,res_predict),2)
SVM['Train Precision'] = round(precision_score(y_train,train_predict,average='macro'),2)
SVM['Test Precision'] = round(precision_score(y_test,res_predict,average='macro'),2)
SVM['Train F1-Score'] = round(f1_score(y_train,train_predict,average='macro'),2)
SVM['Test F1-Score'] = round(f1_score(y_test,res_predict,average='macro'),2)

#LinearSVM
SVML = dict()
#param_grid = dict(C=(0.001,0.01,0.1,0.5,1,2),gamma=(0.001,0.01,0.1,0.5,1,2))
#svml_class = GridSearchCV(SVC(kernel="linear"),cv=5,param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
svml_class = SVC(kernel='linear', gamma=0.001, C=0.5, probability=True)
svml_class.fit(x_train,y_train)
#print(svml_class.best_params_)
res_predict = svml_class.predict(x_test)
train_predict = svml_class.predict(x_train)
#SVML['train_cm'] = confusion_matrix(y_train,train_predict)
#SVML['test_cm'] = confusion_matrix(y_test,res_predict)
SVML['Train Accuracy'] = round(np.mean(cross_val_score(svml_class,x_train,y_train,cv=k_fold,scoring="accuracy")),2)
SVML['Test Accuracy'] = round(accuracy_score(y_test,res_predict),2)
SVML['Train Precision'] = round(precision_score(y_train,train_predict,average='macro'),2)
SVML['Test Precision'] = round(precision_score(y_test,res_predict,average='macro'),2)
SVML['Train F1-Score'] = round(f1_score(y_train,train_predict,average='macro'),2)
SVML['Test F1-Score'] = round(f1_score(y_test,res_predict,average='macro'),2)

#KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = dict()
#param_grid = dict(n_neighbors=np.arange(10,70),weights=("uniform","distance"),p=(1,2))
#knn_class = GridSearchCV(KNeighborsClassifier(),cv=5,param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
knn_class = KNeighborsClassifier(n_neighbors=11,p=2,weights='distance',n_jobs=-1)
knn_class.fit(x_train,y_train)
#print(knn_class.best_params_)
res_predict = knn_class.predict(x_test)
train_predict = knn_class.predict(x_train)
#KNN['train_cm'] = confusion_matrix(y_train,train_predict)
#KNN['test_cm'] = confusion_matrix(y_test,res_predict)
KNN['Train Accuracy'] = round(np.mean(cross_val_score(knn_class,x_train,y_train,cv=k_fold,scoring="accuracy")),2)
KNN['Test Accuracy'] = round(accuracy_score(y_test,res_predict),2)
KNN['Train Precision'] = round(precision_score(y_train,train_predict,average='macro'),2)
KNN['Test Precision'] = round(precision_score(y_test,res_predict,average='macro'),2)
KNN['Train F1-Score'] = round(f1_score(y_train,train_predict,average='macro'),2)
KNN['Test F1-Score'] = round(f1_score(y_test,res_predict,average='macro'),2)

"""
from matplotlib.colors import ListedColormap
features_set, labels_set = features, labels
features1, features2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1))
plt.contourf(features1, features2,res_predict.T, cmap = ListedColormap(('red', 'green')))
plt.xlim(features1.min(), features1.max())
plt.ylim(features2.min(), features2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
DT = dict()
#param_grid = dict(max_depth=np.arange(4,10),min_samples_leaf=np.arange(1,8),min_samples_split=np.arange(2,8),max_leaf_nodes=np.arange(30,100,10))
#dt_class = GridSearchCV(DecisionTreeClassifier(),cv=5,param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
dt_class = DecisionTreeClassifier(max_depth= 9, max_leaf_nodes= 80, min_samples_leaf= 6, min_samples_split= 2)
dt_class.fit(x_train,y_train)
#print(dt_class.best_params_)
res_predict = dt_class.predict(x_test)
train_predict = dt_class.predict(x_train)
#DT['train_cm'] = confusion_matrix(y_train,train_predict)
#DT['test_cm'] = confusion_matrix(y_test,res_predict)
DT['Train Accuracy'] = round(np.mean(cross_val_score(dt_class,x_train,y_train,cv=k_fold,scoring="accuracy")),2)
DT['Test Accuracy'] = round(accuracy_score(y_test,res_predict),2)
DT['Train Precision'] = round(precision_score(y_train,train_predict,average='macro'),2)
DT['Test Precision'] = round(precision_score(y_test,res_predict,average='macro'),2)
DT['Train F1-Score'] = round(f1_score(y_train,train_predict,average='macro'),2)
DT['Test F1-Score'] = round(f1_score(y_test,res_predict,average='macro'),2)

#RandomForest
from sklearn.ensemble import RandomForestClassifier
RF = dict()
#param_grid = dict(max_depth=np.arange(3,10),min_samples_leaf=np.arange(1,10),min_samples_split=np.arange(2,6),max_leaf_nodes=np.arange(50,120,10))
#param_grid = dict(n_estimators = np.arange(50,500,50))
#rf_class = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
rf_class = RandomForestClassifier(n_estimators=250,max_depth= 8, max_leaf_nodes=100, min_samples_leaf= 5, min_samples_split= 5, n_jobs=-1)
rf_class.fit(x_train,y_train)
#print(rf_class.best_params_)
res_predict = rf_class.predict(x_test)
train_predict = rf_class.predict(x_train)
#RF['train_cm'] = confusion_matrix(y_train,train_predict)
#RF['test_cm'] = confusion_matrix(y_test,res_predict)
RF['Train Accuracy'] = round(np.mean(cross_val_score(rf_class,x_train,y_train,cv=k_fold,scoring="accuracy")),2)
RF['Test Accuracy'] = round(accuracy_score(y_test,res_predict),2)
RF['Train Precision'] = round(precision_score(y_train,train_predict,average='macro'),2)
RF['Test Precision'] = round(precision_score(y_test,res_predict,average='macro'),2)
RF['Train F1-Score'] = round(f1_score(y_train,train_predict,average='macro'),2)
RF['Test F1-Score'] = round(f1_score(y_test,res_predict,average='macro'),2)

#XGBoost
import xgboost as xgb
XGB = dict()
#param_grid = dict(n_estimators=np.arange(100,101,50),max_depth=np.arange(6,12),learning_rate=(0.0001,0.001,0.01,0.1))
#xgb_class = GridSearchCV(xgb.XGBClassifier(),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
xgb_class = xgb.XGBClassifier(max_depth=10, n_estimators=50, learning_rate=0.1, n_jobs=-1)
xgb_class.fit(x_train,y_train)
#print(xgb_class.best_params_)
res_predict = xgb_class.predict(x_test)
train_predict = xgb_class.predict(x_train)
#XGB['train_cm'] = confusion_matrix(y_train,train_predict)
#XGB['test_cm'] = confusion_matrix(y_test,res_predict)
XGB['Train Accuracy'] = round(np.mean(cross_val_score(xgb_class,x_train,y_train,cv=k_fold,scoring="accuracy")),2)
XGB['Test Accuracy'] = round(accuracy_score(y_test,res_predict),2)
XGB['Train Precision'] = round(precision_score(y_train,train_predict,average='macro'),2)
XGB['Test Precision'] = round(precision_score(y_test,res_predict,average='macro'),2)
XGB['Train F1-Score'] = round(f1_score(y_train,train_predict,average='macro'),2)
XGB['Test F1-Score'] = round(f1_score(y_test,res_predict,average='macro'),2)




clfResults = pd.DataFrame.from_dict([Logistic_Reg,SVM,SVML,KNN,DT,RF,XGB])
index=pd.Series(['Logistic Regession','Support Vector Machine (RBF)','Support Vector Machine (Linear)','K-Nearest Neighbors','Decision Tree','Random Forest','XGBoost'])
clfResults.set_index(index,inplace=True)
col = clfResults.columns.tolist()
col = [col[i] for i in [3,0,5,2,4,1]]
clfResults = clfResults[col]

"""
Predicting goals scored by
Home and Away teams
"""

features = result.loc[0:,['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff']]
labels_home = result.loc[:,['home_score']]
labels_away = result.loc[:,['away_score']]

sc = StandardScaler()
features = sc.fit_transform(features)
x_home_train,x_home_test,y_home_train,y_home_test = train_test_split(features, labels_home, test_size=0.2, random_state=0)
x_away_train,x_away_test,y_away_train,y_away_test = train_test_split(features, labels_away, test_size=0.2, random_state=0)

x_home_train = pd.DataFrame(x_train,columns=['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff'])
x_home_test = pd.DataFrame(x_test,columns=['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff'])
x_away_train = pd.DataFrame(x_train,columns=['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff'])
x_away_test = pd.DataFrame(x_test,columns=['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff'])


k_fold = KFold(n_splits=5, shuffle=True, random_state=0)


"""
HOME Goals!
"""

#Logistic Regression
from sklearn.linear_model import LogisticRegression
Logistic_Reg_Home = dict()
#param_grid = dict(C=(0.0001,0.001,0.005,0.01,0.1,0.5,1))
#lr_home = GridSearchCV(LogisticRegression(),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
lr_home = LogisticRegression(C=0.0001, n_jobs=-1)
lr_home.fit(x_home_train,y_home_train)
#print(lr_home.best_params_)
res_predict = lr_home.predict(x_home_test)
train_predict = lr_home.predict(x_home_train)
#Logistic_Reg_Home['train_cm'] = confusion_matrix(y_home_train,train_predict)
#Logistic_Reg_Home['test_cm'] = confusion_matrix(y_home_test,res_predict)
Logistic_Reg_Home['Train Accuracy'] = round(np.mean(cross_val_score(lr_home,x_home_train,y_home_train,cv=k_fold,scoring="accuracy")),2)
Logistic_Reg_Home['Test Accuracy'] = round(accuracy_score(y_home_test,res_predict),2)
Logistic_Reg_Home['Train Precision'] = round(precision_score(y_home_train,train_predict,average='macro'),2)
Logistic_Reg_Home['Test Precision'] = round(precision_score(y_home_test,res_predict,average='macro'),2)
Logistic_Reg_Home['Train F1-Score'] = round(f1_score(y_home_train,train_predict,average='macro'),2)
Logistic_Reg_Home['Test F1-Score'] = round(f1_score(y_home_test,res_predict,average='macro'),2)

#SVM RBF
from sklearn.svm import SVC
SVM_Home = dict()
#param_grid = dict(C=(0.001,0.01,0.1,0.5,1,2),gamma=(0.001,0.01,0.1,0.5,1,2))
#svm_home = GridSearchCV(SVC(kernel="rbf",random_state=0),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
svm_home = SVC(kernel='rbf', gamma=0.001, C=2, probability=True)
svm_home.fit(x_home_train,y_home_train)
#print(svm_home.best_params_)
res_predict = svm_home.predict(x_home_test)
train_predict = svm_home.predict(x_home_train)
#SVM_Home['train_cm'] = confusion_matrix(y_home_train,train_predict)
#SVM_Home['test_cm'] = confusion_matrix(y_home_test,res_predict)
SVM_Home['Train Accuracy'] = round(np.mean(cross_val_score(svm_home,x_home_train,y_home_train,cv=k_fold,scoring="accuracy")),2)
SVM_Home['Test Accuracy'] = round(accuracy_score(y_home_test,res_predict),2)
SVM_Home['Train Precision'] = round(precision_score(y_home_train,train_predict,average='macro'),2)
SVM_Home['Test Precision'] = round(precision_score(y_home_test,res_predict,average='macro'),2)
SVM_Home['Train F1-Score'] = round(f1_score(y_home_train,train_predict,average='macro'),2)
SVM_Home['Test F1-Score'] = round(f1_score(y_home_test,res_predict,average='macro'),2)

#KNN
from sklearn.neighbors import KNeighborsClassifier
KNN_Home = dict()
#param_grid = dict(n_neighbors=np.arange(10,70),weights=("uniform","distance"),p=(1,2))
#knn_home = GridSearchCV(KNeighborsClassifier(),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
knn_home = KNeighborsClassifier(n_neighbors=19, p=1, weights='distance', n_jobs=-1)
knn_home.fit(x_home_train,y_home_train)
#print(knn_home.best_params_)
res_predict = knn_home.predict(x_home_test)
train_predict = knn_home.predict(x_home_train)
#KNN_Home['train_cm'] = confusion_matrix(y_home_train,train_predict)
#KNN_Home['test_cm'] = confusion_matrix(y_home_test,res_predict)
KNN_Home['Train Accuracy'] = round(np.mean(cross_val_score(knn_home,x_home_train,y_home_train,cv=k_fold,scoring="accuracy")),2)
KNN_Home['Test Accuracy'] = round(accuracy_score(y_home_test,res_predict),2)
KNN_Home['Train Precision'] = round(precision_score(y_home_train,train_predict,average='macro'),2)
KNN_Home['Test Precision'] = round(precision_score(y_home_test,res_predict,average='macro'),2)
KNN_Home['Train F1-Score'] = round(f1_score(y_home_train,train_predict,average='macro'),2)
KNN_Home['Test F1-Score'] = round(f1_score(y_home_test,res_predict,average='macro'),2)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
DT_Home = dict()
#param_grid = dict(max_depth=np.arange(4,10),min_samples_leaf=np.arange(1,8),min_samples_split=np.arange(2,8),max_leaf_nodes=np.arange(30,100,10))
#dt_home = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
dt_home = DecisionTreeClassifier(max_depth= 5, max_leaf_nodes= 50, min_samples_leaf= 5, min_samples_split= 2)
dt_home.fit(x_home_train,y_home_train)
#print(dt_home.best_params_)6 70 1 6
res_predict = dt_home.predict(x_home_test)
train_predict = dt_home.predict(x_home_train)
#DT_Home['train_cm'] = confusion_matrix(y_home_train,train_predict)
#DT_Home['test_cm'] = confusion_matrix(y_home_test,res_predict)
DT_Home['Train Accuracy'] = round(np.mean(cross_val_score(dt_home,x_home_train,y_home_train,cv=k_fold,scoring="accuracy")),2)
DT_Home['Test Accuracy'] = round(accuracy_score(y_home_test,res_predict),2)
DT_Home['Train Precision'] = round(precision_score(y_home_train,train_predict,average='macro'),2)
DT_Home['Test Precision'] = round(precision_score(y_home_test,res_predict,average='macro'),2)
DT_Home['Train F1-Score'] = round(f1_score(y_home_train,train_predict,average='macro'),2)
DT_Home['Test F1-Score'] = round(f1_score(y_home_test,res_predict,average='macro'),2)


#RandomForest
from sklearn.ensemble import RandomForestClassifier
RF_Home = dict()
#param_grid = dict(max_depth=np.arange(3,10),min_samples_leaf=np.arange(1,10),min_samples_split=np.arange(2,6),max_leaf_nodes=np.arange(50,120,10))
#param_grid = dict(n_estimators = np.arange(50,500,50))
#rf_home = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
rf_home = RandomForestClassifier(n_estimators=100,max_depth= 7, max_leaf_nodes=110, min_samples_leaf= 3, min_samples_split= 2, n_jobs=-1)
rf_home.fit(x_home_train,y_home_train)
#print(rf_home.best_params_)
res_predict = rf_home.predict(x_home_test)
train_predict = rf_home.predict(x_home_train)
#RF_Home['train_cm'] = confusion_matrix(y_home_train,train_predict)
#RF_Home['test_cm'] = confusion_matrix(y_home_test,res_predict)
RF_Home['Train Accuracy'] = round(np.mean(cross_val_score(rf_home,x_home_train,y_home_train,cv=k_fold,scoring="accuracy")),2)
RF_Home['Test Accuracy'] = round(accuracy_score(y_home_test,res_predict),2)
RF_Home['Train Precision'] = round(precision_score(y_home_train,train_predict,average='macro'),2)
RF_Home['Test Precision'] = round(precision_score(y_home_test,res_predict,average='macro'),2)
RF_Home['Train F1-Score'] = round(f1_score(y_home_train,train_predict,average='macro'),2)
RF_Home['Test F1-Score'] = round(f1_score(y_home_test,res_predict,average='macro'),2)

#XGBoost
import xgboost as xgb
XGB_Home = dict()
#param_grid = dict(n_estimators=np.arange(50,500,50),max_depth=np.arange(6,12),learning_rate=(0.0001,0.001,0.01,0.1))
#xgb_home = GridSearchCV(xgb.XGBClassifier(),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
xgb_home = xgb.XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.01, n_jobs=-1)
xgb_home.fit(x_home_train,y_home_train)
#print(xgb_home.best_params_)
res_predict = xgb_home.predict(x_home_test)
train_predict = xgb_home.predict(x_home_train)
#XGB_Home['train_cm'] = confusion_matrix(y_home_train,train_predict)
#XGB_Home['test_cm'] = confusion_matrix(y_home_test,res_predict)
XGB_Home['Train Accuracy'] = round(np.mean(cross_val_score(xgb_home,x_home_train,y_home_train,cv=k_fold,scoring="accuracy")),2)
XGB_Home['Test Accuracy'] = round(accuracy_score(y_home_test,res_predict),2)
XGB_Home['Train Precision'] = round(precision_score(y_home_train,train_predict,average='macro'),2)
XGB_Home['Test Precision'] = round(precision_score(y_home_test,res_predict,average='macro'),2)
XGB_Home['Train F1-Score'] = round(f1_score(y_home_train,train_predict,average='macro'),2)
XGB_Home['Test F1-Score'] = round(f1_score(y_home_test,res_predict,average='macro'),2)

clfResultsHome = pd.DataFrame.from_records([Logistic_Reg_Home,SVM_Home,KNN_Home,DT_Home,RF_Home,XGB_Home],index=['Logistic Regession','Support Vector Machine (RBF)','K-Nearest Neighbors','Decision Tree','Random Forest','XGBoost'])
col = clfResultsHome.columns.tolist()
col = [col[i] for i in [3,0,5,2,4,1]]
clfResultsHome = clfResultsHome[col]


"""
AWAY Goals!
"""

#Logistic Regression
from sklearn.linear_model import LogisticRegression
Logistic_Reg_Away = dict()
#param_grid = dict(C=(0.0001,0.001,0.005,0.01,0.1,0.5,1))
#lr_away = GridSearchCV(LogisticRegression(),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
lr_away = LogisticRegression(C=0.001, n_jobs=-1)
lr_away.fit(x_away_train,y_away_train)
#print(lr_away.best_params_)
res_predict = lr_away.predict(x_away_test)
train_predict = lr_away.predict(x_away_train)
#Logistic_Reg_Away['train_cm'] = confusion_matrix(y_away_train,train_predict)
#Logistic_Reg_Away['test_cm'] = confusion_matrix(y_away_test,res_predict)
Logistic_Reg_Away['Train Accuracy'] = round(np.mean(cross_val_score(lr_away,x_away_train,y_away_train,cv=k_fold,scoring="accuracy")),2)
Logistic_Reg_Away['Test Accuracy'] = round(accuracy_score(y_away_test,res_predict),2)
Logistic_Reg_Away['Train Precision'] = round(precision_score(y_away_train,train_predict,average='macro'),2)
Logistic_Reg_Away['Test Precision'] = round(precision_score(y_away_test,res_predict,average='macro'),2)
Logistic_Reg_Away['Train F1-Score'] = round(f1_score(y_away_train,train_predict,average='macro'),2)
Logistic_Reg_Away['Test F1-Score'] = round(f1_score(y_away_test,res_predict,average='macro'),2)

#SVM RBF
from sklearn.svm import SVC
SVM_Away = dict()
#param_grid = dict(C=(0.001,0.01,0.1,0.5,1,2),gamma=(0.001,0.01,0.1,0.5,1,2))
#svm_away = GridSearchCV(SVC(kernel="rbf",random_state=0),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
svm_away = SVC(kernel='rbf', gamma=0.001, C=2, probability=True)
svm_away.fit(x_away_train,y_away_train)
#print(svm_away.best_params_)
res_predict = svm_away.predict(x_away_test)
train_predict = svm_away.predict(x_away_train)
#SVM_Away['train_cm'] = confusion_matrix(y_away_train,train_predict)
#SVM_Away['test_cm'] = confusion_matrix(y_away_test,res_predict)
SVM_Away['Train Accuracy'] = round(np.mean(cross_val_score(svm_away,x_away_train,y_away_train,cv=k_fold,scoring="accuracy")),2)
SVM_Away['Test Accuracy'] = round(accuracy_score(y_away_test,res_predict),2)
SVM_Away['Train Precision'] = round(precision_score(y_away_train,train_predict,average='macro'),2)
SVM_Away['Test Precision'] = round(precision_score(y_away_test,res_predict,average='macro'),2)
SVM_Away['Train F1-Score'] = round(f1_score(y_away_train,train_predict,average='macro'),2)
SVM_Away['Test F1-Score'] = round(f1_score(y_away_test,res_predict,average='macro'),2)

#KNN
from sklearn.neighbors import KNeighborsClassifier
KNN_Away = dict()
#param_grid = dict(n_neighbors=np.arange(10,70),weights=("uniform","distance"),p=(1,2))
#knn_away = GridSearchCV(KNeighborsClassifier(),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
knn_away = KNeighborsClassifier(n_neighbors=10, p=2, weights='distance', n_jobs=-1)
knn_away.fit(x_away_train,y_away_train)
#print(knn_away.best_params_)
res_predict = knn_away.predict(x_away_test)
train_predict = knn_away.predict(x_away_train)
#KNN_Away['train_cm'] = confusion_matrix(y_away_train,train_predict)
#KNN_Away['test_cm'] = confusion_matrix(y_away_test,res_predict)
KNN_Away['Train Accuracy'] = round(np.mean(cross_val_score(knn_away,x_away_train,y_away_train,cv=k_fold,scoring="accuracy")),2)
KNN_Away['Test Accuracy'] = round(accuracy_score(y_away_test,res_predict),2)
KNN_Away['Train Precision'] = round(precision_score(y_away_train,train_predict,average='macro'),2)
KNN_Away['Test Precision'] = round(precision_score(y_away_test,res_predict,average='macro'),2)
KNN_Away['Train F1-Score'] = round(f1_score(y_away_train,train_predict,average='macro'),2)
KNN_Away['Test F1-Score'] = round(f1_score(y_away_test,res_predict,average='macro'),2)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
DT_Away = dict()
#param_grid = dict(max_depth=np.arange(4,10),min_samples_leaf=np.arange(1,8),min_samples_split=np.arange(2,8),max_leaf_nodes=np.arange(30,100,10))
#dt_away = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
dt_away = DecisionTreeClassifier(max_depth= 6, max_leaf_nodes= 90, min_samples_leaf= 6, min_samples_split= 2)
dt_away.fit(x_away_train,y_away_train)
#print(dt_away.best_params_) 8 90 5 3
res_predict = dt_away.predict(x_away_test)
train_predict = dt_away.predict(x_away_train)
#DT_Away['train_cm'] = confusion_matrix(y_away_train,train_predict)
#DT_Away['test_cm'] = confusion_matrix(y_away_test,res_predict)
DT_Away['Train Accuracy'] = round(np.mean(cross_val_score(dt_away,x_away_train,y_away_train,cv=k_fold,scoring="accuracy")),2)
DT_Away['Test Accuracy'] = round(accuracy_score(y_away_test,res_predict),2)
DT_Away['Train Precision'] = round(precision_score(y_away_train,train_predict,average='macro'),2)
DT_Away['Test Precision'] = round(precision_score(y_away_test,res_predict,average='macro'),2)
DT_Away['Train F1-Score'] = round(f1_score(y_away_train,train_predict,average='macro'),2)
DT_Away['Test F1-Score'] = round(f1_score(y_away_test,res_predict,average='macro'),2)

#RandomForest
from sklearn.ensemble import RandomForestClassifier
RF_Away = dict()
#param_grid = dict(max_depth=np.arange(3,10),min_samples_leaf=np.arange(1,10),min_samples_split=np.arange(2,6),max_leaf_nodes=np.arange(50,120,10))
#param_grid = dict(n_estimators = np.arange(50,500,50))
#rf_away = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
rf_away = RandomForestClassifier(n_estimators=100,max_depth= 7, max_leaf_nodes=90, min_samples_leaf= 4, min_samples_split= 2, n_jobs=-1)
rf_away.fit(x_away_train,y_away_train)
#print(rf_away.best_params_)
res_predict = rf_away.predict(x_away_test)
train_predict = rf_away.predict(x_away_train)
#RF_Away['train_cm'] = confusion_matrix(y_away_train,train_predict)
#RF_Away['test_cm'] = confusion_matrix(y_away_test,res_predict)
RF_Away['Train Accuracy'] = round(np.mean(cross_val_score(rf_away,x_away_train,y_away_train,cv=k_fold,scoring="accuracy")),2)
RF_Away['Test Accuracy'] = round(accuracy_score(y_away_test,res_predict),2)
RF_Away['Train Precision'] = round(precision_score(y_away_train,train_predict,average='macro'),2)
RF_Away['Test Precision'] = round(precision_score(y_away_test,res_predict,average='macro'),2)
RF_Away['Train F1-Score'] = round(f1_score(y_away_train,train_predict,average='macro'),2)
RF_Away['Test F1-Score'] = round(f1_score(y_away_test,res_predict,average='macro'),2)

#XGBoost
import xgboost as xgb
XGB_Away = dict()
#param_grid = dict(n_estimators=np.arange(50,500,50),max_depth=np.arange(6,12),learning_rate=(0.0001,0.001,0.01,0.1))
#xgb_away = GridSearchCV(xgb.XGBClassifier(random_state=0),param_grid=param_grid,scoring="f1_macro",n_jobs=-1)
xgb_away = xgb.XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.01, n_jobs=-1)
xgb_away.fit(x_away_train,y_away_train)
#print(xgb_away.best_params_)
res_predict = xgb_away.predict(x_away_test)
train_predict = xgb_away.predict(x_away_train)
#XGB_Away['train_cm'] = confusion_matrix(y_away_train,train_predict)
#XGB_Away['test_cm'] = confusion_matrix(y_away_test,res_predict)
XGB_Away['Train Accuracy'] = round(np.mean(cross_val_score(xgb_away,x_away_train,y_away_train,cv=k_fold,scoring="accuracy")),2)
XGB_Away['Test Accuracy'] = round(accuracy_score(y_away_test,res_predict),2)
XGB_Away['Train Precision'] = round(precision_score(y_away_train,train_predict,average='macro'),2)
XGB_Away['Test Precision'] = round(precision_score(y_away_test,res_predict,average='macro'),2)
XGB_Away['Train F1-Score'] = round(f1_score(y_away_train,train_predict,average='macro'),2)
XGB_Away['Test F1-Score'] = round(f1_score(y_away_test,res_predict,average='macro'),2)

clfResultsAway = pd.DataFrame.from_records([Logistic_Reg_Away,SVM_Away,KNN_Away,DT_Away,RF_Away,XGB_Away],index=['Logistic Regession','Support Vector Machine (RBF)','K-Nearest Neighbors','Decision Tree','Random Forest','XGBoost'])
col = clfResultsAway.columns.tolist()
col = [col[i] for i in [3,0,5,2,4,1]]
clfResultsAway = clfResultsAway[col]


"""
Current Ability and Potential
of the players of
each country.
"""

squads = pd.read_csv('squads_upd.csv')
fifa19 = pd.read_csv('fifa19_cleaned.csv')

"""
names = pd.read_csv('names.csv')
fifa19 = pd.merge(left=fifa19, right=names, how='left', on=['ID'], suffixes=('_x','_y'))
fifa_names = fifa19['Full Name'].str.split(" ", n=1, expand=True)
fifa19['First Name'] = fifa_names[0]
fifa19['Last Name'] = fifa_names[1]
fifa19 = fifa19.fillna('-')
fifa19.to_csv('fifa_19.csv',encoding='utf-8')
squads_names = squads['player'].str.split(" ", n=1, expand=True)
squads['First Name'] = squads_names[0]
squads['Last Name'] = squads_names[1]
squads=squads.fillna('#')
squads.to_csv('squads_upd.csv',encoding='utf-8')
"""
teams = squads.team.unique().tolist()
fifa19 = fifa19[fifa19.Nationality.isin(teams)].reset_index(drop=True)
fifa19_stats = fifa19[fifa19.Nationality.isin(squads.team) & (fifa19['Full Name'].isin(squads['player']) |
        fifa19['Name'].isin(squads['player']) | fifa19['First Name'].isin(squads['Last Name']) )]
fifa19_stats.reset_index(inplace=True,drop=True)
fifa19_stats = fifa19_stats.groupby('Nationality').apply(lambda x: (x.sort_values('Overall',ascending=False)).head(23)).reset_index(drop=True)
#fifa19_stats.groupby('Nationality').count()

for team in teams:
    count = fifa19_stats[fifa19_stats.Nationality==team]['Full Name'].count()
    if count >= 23:
        continue
    squadUpdate = fifa19[(fifa19.Nationality==team) & ( ~fifa19['Full Name'].isin(fifa19_stats['Full Name']) )].sort_values('Overall',ascending=False).head(23-count)
    fifa19_stats = pd.concat([fifa19_stats,squadUpdate]).drop_duplicates().reset_index(drop=True)

grp = fifa19_stats.groupby('Nationality').apply(lambda x: round((x.sort_values('Overall',ascending=False)).mean(),2)).sort_values('Potential')
grp['Points to potential'] = grp['Potential'] - grp['Overall']
grp = grp.sort_values(by = 'Overall')
current = grp['Overall']
potential = grp['Points to potential']
ind = np.arange(12)
width = 0.5
plt.figure(figsize=(10,10))
p1 = plt.bar(ind,current,width)
p2 = plt.bar(ind,potential,width,bottom=current)
plt.ylabel('Ability')
plt.xlabel('Current and Potential Ability for each country')
plt.xticks(ind,(grp.index),rotation=90)
plt.legend((p1[0],p2[0]),('Current','Potential'))
plt.show()

grp = grp.reset_index()


"""
Adding variables to 
build a Poisson Model.
"""

df=pd.read_csv("spi.csv")
spi=pd.merge(left=df,right=grp,how='left',left_on=['Name'],right_on=['Nationality']).dropna().drop(['Nationality'],axis=1)
team=pd.read_csv('history team.csv',delimiter='\t')
spi=pd.merge(left=spi,right=team,how='left',left_on=['Name'],right_on=['Team']).drop(['Team'],axis=1)
fixtures=pd.read_csv('copa_fixtures.csv')
fixtures=pd.merge(left=fixtures,right=spi,how='left',left_on=['Team'],right_on=['Name']).drop(['Name'],axis=1).fillna(0)
fixtures["avg score"] = round(fixtures['GF'] / fixtures['GP'],2)
fixtures["avg conceded"] = round(fixtures['GA'] / fixtures['GP'],2)

fixtures.iloc[0:,[5,6,7,9,-2,-1]].fillna(0,inplace=True)
sc=StandardScaler()

fixtures[['Part.','Overall','Potential']]=sc.fit_transform(fixtures[['Part.','Overall','Potential']])


"""
Predicting Copa America 2019
"""

features = result.loc[0:,['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff']]
labels = result.loc[:,['results']]

sc = StandardScaler()
features = sc.fit_transform(features)
features = pd.DataFrame(features,columns=['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff'])

lr_class = LogisticRegression(C=0.001,n_jobs=-1)
lr_class.fit(features,labels)

svm_class = SVC(kernel='rbf',C=2,gamma=0.001,probability=True)
svm_class.fit(features,labels)

knn_class = KNeighborsClassifier(n_neighbors=12,p=2,weights='distance',n_jobs=-1)
knn_class.fit(features,labels)

dt_class = DecisionTreeClassifier(max_depth= 9, max_leaf_nodes= 80, min_samples_leaf= 6, min_samples_split= 2)
dt_class.fit(features,labels)

rf_class = RandomForestClassifier(n_estimators=250,max_depth= 8, max_leaf_nodes=100, min_samples_leaf= 5, min_samples_split= 5, n_jobs=-1)
rf_class.fit(features,labels)

xgb_class = xgb.XGBClassifier(max_depth=10, n_estimators=50, learning_rate=0.1, n_jobs=-1)
xgb_class.fit(features,labels)


#Group Stage
from itertools import combinations
opponents = ['First match \nagainst', 'Second match\n against', 'Third match\n against']
fixtures['Points'] = 0
fixtures['Total_Prob'] = 0
fixtures.set_index('Team',inplace=True)
current_ranking.set_index('country_full',inplace=True)

for group in list(fixtures['Group'].unique()):
    print('---Group {}---'.format(group))
    for home, away in combinations(fixtures.query('Group == "{}"'.format(group)).index, 2):
        print('{} vs. {}: '.format(home, away), end='')
        match = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]), columns=['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff'])
        match['host'] = 1 if home=='Brazil' or away=='Brazil' else 0
        match['impt'] = 1
        match['home_pts_diff'] = current_ranking.loc[home,'pts_diff']
        match['away_pts_diff'] = current_ranking.loc[away,'pts_diff']
        match['rank_diff'] = current_ranking.loc[home,'rank'] - current_ranking.loc[away,'rank']
        match['weighted_diff'] = current_ranking.loc[home,'mean_weighted'] - current_ranking.loc[away,'mean_weighted']
        
        home_win_prob = rf_class.predict_proba(match)[:,0][0]
        away_win_prob = rf_class.predict_proba(match)[:,2][0]
        draw_prob = rf_class.predict_proba(match)[:,1][0]
        
        home_diff_in_countries = (0.35*(fixtures.loc[home,'SPI']-fixtures.loc[away,'SPI']) +
                              0.25*(fixtures.loc[home,'Potential']-fixtures.loc[away,'Potential'])+
                              0.20*(fixtures.loc[home,'Part.']-fixtures.loc[away,'Part.'])-
                              0.05*(fixtures.loc[home,'Age']-fixtures.loc[away,'Age'])+
                              0.05*(fixtures.loc[home,'Height']-fixtures.loc[away,'Height'])+
                              0.10*(fixtures.loc[home,'Overall']-fixtures.loc[away,'Overall']))
        
        away_diff_in_countries = (0.4*(fixtures.loc[away,'SPI']-fixtures.loc[home,'SPI']) +
                              0.25*(fixtures.loc[away,'Potential']-fixtures.loc[home,'Potential'])+
                              0.25*(fixtures.loc[away,'Part.']-fixtures.loc[home,'Part.'])-
                              0.05*(fixtures.loc[away,'Age']-fixtures.loc[home,'Age'])+
                              0.05*(fixtures.loc[away,'Height']-fixtures.loc[home,'Height'])+
                              0.20*(fixtures.loc[away,'Overall']-fixtures.loc[home,'Overall']))
        
        home_prob_goals.append(0.7*poisson.pmf(i,mean_home_goals) + 0.3*home_prob_goals_rfmodel[i])
        away_prob_goals.append(0.7*poisson.pmf(i,mean_away_goals) + 0.3*away_prob_goals_rfmodel[i])
        
        mean_home_goals = max(0,base_home_goals + home_diff_in_countries)
        mean_away_goals = max(0,base_away_goals + away_diff_in_countries)
        home_prob_goals = list()
        away_prob_goals = list()
        points = 0
        if max(home_win_prob,away_win_prob,draw_prob) == away_win_prob:
            print("{} wins with a probability of {:.2f}% ".format(away, away_win_prob))
            fixtures.loc[away, 'Points'] += 3
            fixtures.loc[home, 'Total_Prob'] += home_win_prob
            fixtures.loc[away, 'Total_Prob'] += away_win_prob
        elif max(home_win_prob,away_win_prob,draw_prob) == draw_prob:
            points = 1
            print("Draw with probability of {:.2f}%".format(draw_prob))
            fixtures.loc[home, 'Points'] += 1
            fixtures.loc[away, 'Points'] += 1
            fixtures.loc[home, 'Total_Prob'] += draw_prob
            fixtures.loc[away, 'Total_Prob'] += draw_prob
        elif max(home_win_prob,away_win_prob,draw_prob) == home_win_prob:
            points = 3
            fixtures.loc[home, 'Points'] += 3
            fixtures.loc[home, 'Total_Prob'] += home_win_prob
            fixtures.loc[away, 'Total_Prob'] += away_win_prob
            print("{} wins with a probability of {:.2f}%".format(home, home_win_prob))
    print()

"""
Knockout stages and finals
"""
#Function to predict result
def knockout(home,away):
    print("{} vs. {}: ".format(home, away), end='')
    match = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]), columns=['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff'])
    match['host'] = 1 if home=='Brazil' or away=='Brazil' else 0
    match['impt'] = 1
    match['home_pts_diff'] = current_ranking.loc[home,'pts_diff']
    match['away_pts_diff'] = current_ranking.loc[away,'pts_diff']
    match['rank_diff'] = current_ranking.loc[home,'rank'] - current_ranking.loc[away,'rank']
    match['weighted_diff'] = current_ranking.loc[home,'mean_weighted'] - current_ranking.loc[away,'mean_weighted']
    base_home_goals = max(fixtures.loc[home,'avg score'],fixtures.loc[away,'avg conceded'])
    base_away_goals = max(fixtures.loc[home,'avg conceded'],fixtures.loc[away,'avg score'])
    home_diff_in_countries = (0.35*(fixtures.loc[home,'SPI']-fixtures.loc[away,'SPI']) +
                              0.25*(fixtures.loc[home,'Potential']-fixtures.loc[away,'Potential'])+
                              0.20*(fixtures.loc[home,'Part.']-fixtures.loc[away,'Part.'])-
                              0.05*(fixtures.loc[home,'Age']-fixtures.loc[away,'Age'])+
                              0.05*(fixtures.loc[home,'Height']-fixtures.loc[away,'Height'])+
                              0.10*(fixtures.loc[home,'Overall']-fixtures.loc[away,'Overall']))
    
    away_diff_in_countries = (0.4*(fixtures.loc[away,'SPI']-fixtures.loc[home,'SPI']) +
                              0.25*(fixtures.loc[away,'Potential']-fixtures.loc[home,'Potential'])+
                              0.25*(fixtures.loc[away,'Part.']-fixtures.loc[home,'Part.'])-
                              0.05*(fixtures.loc[away,'Age']-fixtures.loc[home,'Age'])+
                              0.05*(fixtures.loc[away,'Height']-fixtures.loc[home,'Height'])+
                              0.20*(fixtures.loc[away,'Overall']-fixtures.loc[home,'Overall']))
    
    mean_home_goals = max(0,base_home_goals + home_diff_in_countries)
    mean_away_goals = max(0,base_away_goals + away_diff_in_countries)
    home_prob_goals = list()
    away_prob_goals = list()
    home_prob_goals_rfmodel = list(rf_home.predict_proba(match)[0])
    away_prob_goals_rfmodel = list(rf_away.predict_proba(match)[0])
    for i in range(7):
        home_prob_goals.append(0.7*poisson.pmf(i,mean_home_goals) + 0.3*home_prob_goals_rfmodel[i])
        away_prob_goals.append(0.7*poisson.pmf(i,mean_away_goals) + 0.3*away_prob_goals_rfmodel[i])
    
    home_goals = np.argmax(home_prob_goals)
    away_goals = np.argmax(away_prob_goals)
    
    if home_goals>away_goals:
        print("{} wins {} with score of {}:{}".format(home,away,str(home_goals),str(away_goals)),end='')
        winners.append(home)
        losers.append(away)
    elif home_goals<away_goals:
        print("{} wins {} with score of {}:{}".format(away,home,str(away_goals),str(home_goals)),end='')
        winners.append(away)
        losers.append(home)
    else:
        team = [home,away]
        win = random.choice(team)
        team.remove(win)
        loser = team[0]
        print("{} draws with {} with a score of {}:{} after Extra-Time and {} wins the Penalty Shootout".format(home,away,str(home_goals),str(away_goals),win),end='')
        winners.append(win)
        losers.append(loser)
    print()

#home='Brazil'
#away = 'Colombia'

#QuarterFinals
features = result.loc[0:,['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff']]
labels_home = result.loc[:,['home_score']]
labels_away = result.loc[:,['away_score']]

sc = StandardScaler()
features = sc.fit_transform(features)
features = pd.DataFrame(features,columns=['host','impt','home_pts_diff','away_pts_diff','rank_diff','weighted_diff'])

#rf_home = RandomForestClassifier(n_estimators=100,max_depth= 8, max_leaf_nodes=110, min_samples_leaf= 2, min_samples_split= 2, n_jobs=-1)
#rf_home.fit(features,labels_home)

#rf_away = RandomForestClassifier(n_estimators=100,max_depth= 7, max_leaf_nodes=90, min_samples_leaf= 4, min_samples_split= 2, n_jobs=-1)
#rf_away.fit(features,labels_away)

xgb_home = xgb.XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.01, n_jobs=-1)
xgb_home.fit(features,labels_home)

xgb_away = xgb.XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.01, n_jobs=-1)
xgb_away.fit(features,labels_away)

pairing = [0,7,3,4,1,5,2,6]

fixtures = fixtures.sort_values(['Points','Group'],ascending=[False,True]).reset_index()
Finals = fixtures.groupby('Group').nth([0,1]).reset_index().set_index('Team')
Finals.sort_values(['Points','Group'],ascending=[False,True],inplace=True)
thirdPlaced=fixtures.groupby('Group').nth([2]).sort_values(['Points','Total_Prob'],ascending=[False,False]).reset_index().set_index('Team')
Finals=Finals.append(thirdPlaced.iloc[:2,:])
Finals.reset_index(inplace=True)
Finals = Finals.loc[pairing].set_index('Team')
fixtures.set_index('Team',inplace=True)

finals = ['Quarter-Finals','Semi-Finals']

for f in finals:
    print("___Starting of the {}___".format(f))
    iterations = int(len(Finals) / 2)
    winners = list()
    losers = list()
    for i in range(iterations):
        home = Finals.index[i*2]
        away = Finals.index[i*2+1]        
        knockout(home,away)
    Finals = Finals.loc[winners]    
    if len(winners)>2:
        print('\nSemi-Finalists: ' + winners[0] + ',' + winners[1] + ',' + winners[2] + ',' + winners[3] + '\n')
    else:
        print('\nFinalists: ' + winners[0] + ',' + winners[1] + '\n')

finalists = winners


#Third Place
print("___Third Place Playoff___")
home = losers[0]
away = losers[1]
winners = list()
losers = list()
knockout(home,away)
third = winners[0]
fourth = losers[0]

#Final
print("\n___Copa America Final___")
home = finalists[0]
away = finalists[1]
knockout(home,away)
champion = winners[1]
runners_up = losers[1]

print("\n{} wins the Copa America 2019.\n".format(champion))
print("{} are Runners-up.\n".format(runners_up))
print("{} secures Third place.\n".format(third))
print("{} secures Fourth place after losing from {} in the Playoff\n".format(fourth,third))


"""
result['home_pts_diff'] = 0
result['away_pts_diff'] = 0

#result['home_pts_diff'] = np.where(((result['home_team']==ranking['country_full']) and (result['date'].dt.month==ranking['date'].dt.month) and (result['date'].dt.year==ranking['date'].dt.year)),ranking['pts_diff'],0)

result['home_rank'] = np.where(result['home_team']==ranking['country_full'],ranking['rank'],np.nan)

res = result.groupby(['home_team'])
rank = ranking.groupby('country_full')
result['home_pts_diff'] = ranking[(res['country_full']==rank['country_full']) & (res['date'].dt.month==rank['date'].dt.month) & (res['date'].dt.year==rank['date'].dt.year)]['pts_diff']


features=ranking.iloc[0: ,[0,2,6]]

features1 = dict()
features2 = dict()
labels = dict()

for country in countries:
    features1[country]=ranking[(ranking['country_full']==country)][['rank_date','country_full','pts_diff']]
    features2[country]=result[(result['home_team']==country)][['away_team','tournament','neutral']]
    labels[country]=result[(result['home_team']==country)][['home_score','away_score']]
    
#result['date'] = result['date'].dt.date
"""
