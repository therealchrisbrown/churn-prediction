import numpy as np
import pandas as pd

# Datenvisualisierungen
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
pd.options.display.max_rows = None
pd.options.display.max_columns = None
from decouple import config

# read data
df = pd.read_csv(config('SOURCE'))

#Was für Daten liegen vor?
print(df.shape)

#Gibt es fehlende Werte?
print(df.isnull().sum())

#Eindeutige Anzahl für jedes Attribut ermitteln
print(df.nunique())

#Erste Datenbereinigung - Ausschluss (RowNumer; CustomerId; Surname)
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
print(df.head())
print(df.dtypes)

#Visualisierung Anteil der abgewanderten und gebliebenden Kunden
labels = 'Exited', 'Retained'
sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')
plt.title("Propotion of customer churned and retained", size=20)
plt.show()

#Überprüfung der ketagorialen Attribute mit dem Attribut Status ('Exited')
fig, axarr = plt.subplots(2, 2, figsize=(20,12))
sns.countplot(x='Geography',hue='Exited',data=df,ax=axarr[0][0])
sns.countplot(x='Gender',hue='Exited',data=df,ax=axarr[0][1])
sns.countplot(x='HasCrCard',hue='Exited',data=df,ax=axarr[1][0])
sns.countplot(x='IsActiveMember',hue='Exited',data=df,ax=axarr[1][1])
plt.show()

#Restliche Beziehung der Attribute mit dem Attribut Status ('Exited')
fig, axarr = plt.subplots(3,2,figsize=(20,12))
sns.boxplot(y='CreditScore',x='Exited',hue='Exited',data=df,ax=axarr[0][0])
sns.boxplot(y='Age',x='Exited',hue='Exited',data=df,ax=axarr[0][1])
sns.boxplot(y='Tenure',x='Exited',hue='Exited',data=df,ax=axarr[1][0])
sns.boxplot(y='Balance',x='Exited',hue='Exited',data=df,ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x='Exited',hue='Exited',data=df,ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x='Exited',hue='Exited',data=df,ax=axarr[2][1])
plt.show()

#Aufsplittung in test & train (80:20)
df_train = df.sample(frac=0.8, random_state=200)
df_test = df.drop(df_train.index)
#print(len(df_train))
#print(len(df_test))

#BalanceSalaryRatio definieren
df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
##sns.boxplot(y='BalanceSalaryRatio',x='Exited',hue='Exited',data=df_train)
##plt.ylim(-1,5)
#plt.show()

#Tenure = 'Funktion' des Attributs 'Alters' -> 'TenureByAge'
df_train['TenureByAge'] = df_train.Tenure/(df_train.Age)
##sns.boxplot(y='TenureByAge',x='Exited',hue='Exited',data=df_train)
##plt.ylim(-1,1)
#plt.show()

#Variable zur Erfassung der Kreditwürdigkeit in Abhängigkeit vom Alter (Kreditverhalten im Erwachsenenalter berücksichtigen)
df_train['CreditScoreGivenAge'] =df_train.CreditScore/(df_train.Age)
#print(df_train.head())

#Spalten nach Data Type anordnen
continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember', 'Geography', 'Gender']
df_train = df_train[['Exited'] + continuous_vars + cat_vars]
#print(df_train.head())

#Alle Attribute, die bisher 0/1 hatten auf 0/-1 ändern -> negative Beziehung erfassen
df_train.loc[df_train.HasCrCard == 0, 'HasCrCard'] = -1
df_train.loc[df_train.IsActiveMember == 0, 'IsActiveMember'] =-1
#print(df_train.head())

#Gleiche für die kategorialen Attribute 'Geography' & 'Gender'
lst = ['Geography', 'Gender']
remove = list()
for i in lst: 
        if(df_train[i].dtype == np.str or df_train[i].dtype == np.object):
                for j in df_train[i].unique():
                        df_train[i+'_'+j] = np.where(df_train[i] == j, 1, -1)
                remove.append(i)
df_train = df_train.drop(remove, axis=1)
#print(df_train.head())

#minMax scaling the continuous variables
minVec = df_train[continuous_vars].min().copy()
maxVec = df_train[continuous_vars].max().copy()
df_train[continuous_vars] = (df_train[continuous_vars]-minVec)/(maxVec-minVec)
#print(df_train.head())

#Datenaufbereitung für Pipeline Test-Daten
def DfPrepPipeline(df_predict,df_train_Cols,minVec,maxVec):
        #Neue Variablen hinzufügen
        df_predict['BalanceSalaryRatio'] = df_predict.Balance/df_predict.EstimatedSalary
        df_predict['TenureByAge'] = df_predict.Tenure/(df_predict.Age - 18)
        df_predict['CreditScoreGivenAge'] = df_predict.CreditScore/(df_predict.Age - 18)

        #Neuordnung der Spalten
        continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreGivenAge']
        cat_vars = ['HasCrCard', 'IsActiveMember', "Geography", "Gender"]
        df_predict = df_predict[['Exited'] + continuous_vars + cat_vars]

        #Änderung auf negative Beziehungen
        df_predict.loc[df_predict.HasCrCard == 0, 'HasCrCard'] = -1
        df_predict.loc[df_predict.IsActiveMember == 0, 'IsActiveMember'] = -1
        
        #Gleiche für die kategorialen Attribute 'Geography' & 'Gender'
        lst = ['Geography', 'Gender']
        remove = list()
        for i in lst: 
                if(df_predict[i].dtype == np.str or df_predict[i].dtype == np.object):
                        for j in df_predict[i].unique():
                                df_predict[i+'_'+j] = np.where(df_predict[i] == j, 1, -1)
                        remove.append(i)
        df_predict = df_predict.drop(remove, axis=1)

        #Sicherstellung, dass alle Daten auch in den nachfolgenden Daten erscheinen
        L = list(set(df_train_Cols) - set(df_predict.columns))
        for l in L:
                df_predict[str(1)] = -1
        #minMax scaling continuous variables
        df_predict[continuous_vars] = (df_predict[continuous_vars] - minVec)/(maxVec - minVec)

        #Sicherstellung, dass die Daten wie beim Trainigsdatensatz sortiert sind
        df_predict =df_predict[df_train_Cols]
        return df_predict

### MODEL FITTING UND AUSWAHL
#### ___ wurde über Dataaiku herausgefunden

#sklearn import
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

#Fitting models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Funktion fürs Scoring
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#Herausfinden: best model score & params
def best_model(model):
        print(model.best_score_)
        print(model.best_params_)
        print(model.best_estimator_)
def get_auc_scores(y_actual, method,method2):
        auc_score = roc_auc_score(y_actual, method2);
        fpr_df, tpr_df, _ = roc_curve(y_actual, method2);
        return (auc_score, fpr_df, tpr_df)


####___ MODEL FITTING ____


#FIT Logistic regression
##param_grid = {'C': [0.1,0.5,1,10,50,100], 'max_iter': [250], 'fit_intercept':[True],'intercept_scaling':[1],
##              'penalty':['l2'], 'tol':[0.00001,0.0001,0.000001]}
##log_primal_Grid = GridSearchCV(LogisticRegression(solver='lbfgs'),param_grid, cv=10, refit=True, verbose=0)
##log_primal_Grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
##print(best_model(log_primal_Grid))

#FIT logistic regression (2 polynomial kernel)
##param_grid = {'C': [0.1,10,50], 'max_iter': [300,500], 'fit_intercept':[True],'intercept_scaling':[1],'penalty':['l2'],
##              'tol':[0.0001,0.000001]}
##poly2 = PolynomialFeatures(degree=2)
##df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
##log_pol2_Grid = GridSearchCV(LogisticRegression(solver = 'liblinear'),param_grid, cv=5, refit=True, verbose=0)
##log_pol2_Grid.fit(df_train_pol2,df_train.Exited)
##print(best_model(log_pol2_Grid))

#FIT SVM (RBF Kernel)
##param_grid = {'C': [0.5,100,150], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['rbf']}
##SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
##SVM_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
##print(best_model(SVM_grid))

#FIT SVM (pol kernel)
##param_grid = {'C': [0.5,1,10,50,100], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['poly'],'degree':[2,3] }
##SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
##SVM_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
##print(best_model(SVM_grid))

#FIT random forest
##param_grid = {'max_depth': [3, 5, 6, 7, 8], 'max_features': [2,4,6,7,8,9],'n_estimators':[50,100],'min_samples_split': [3, 5, 6, 7]}
##RanFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, refit=True, verbose=0)
##RanFor_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
##print(best_model(RanFor_grid))

#FIT Extreme Gradient boost
##param_grid = {'max_depth': [5,6,7,8], 'gamma': [0.01,0.001,0.001],'min_child_weight':[1,5,10], 'learning_rate': [0.05,0.1, 0.2, 0.3], 'n_estimators':[5,10,20,100]}
##xgb_grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, refit=True, verbose=0)
##xgb_grid.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
##print(best_model(xgb_grid))


####___ FIT BEST MODELS___


#FIT logistic regression
log_primal = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=250, multi_class='ovr',n_jobs=None, 
                                penalty='l2', random_state=None, solver='lbfgs',tol=1e-05, verbose=0, warm_start=False)
log_primal.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)

#FIT logistic regression (pol 2 kernel)
poly2 = PolynomialFeatures(degree=2)
df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
log_pol2 = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=300, multi_class='ovr', n_jobs=None, 
                              penalty='l2', random_state=None, solver='liblinear',tol=0.0001, verbose=0, warm_start=False)
log_pol2.fit(df_train_pol2,df_train.Exited)

#FIT SVM (RBF Kernel)
SVM_RBF = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf', max_iter=-1, probability=True, 
              random_state=None, shrinking=True,tol=0.001, verbose=False)
SVM_RBF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)

#FIT SVM (Pol Kernel)
SVM_POL = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='poly',  max_iter=-1,
              probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
SVM_POL.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)

#FIT Random Forest
RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,warm_start=False)
RF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)

### Extreme Gradient Boost entfällt -> Fehler!

####___REVIEW -> welches Model hat die beste Accuracy?


print(classification_report(df_train.Exited, log_primal.predict(df_train.loc[:, df_train.columns != 'Exited'])))
print(classification_report(df_train.Exited,  log_pol2.predict(df_train_pol2)))
print(classification_report(df_train.Exited,  SVM_RBF.predict(df_train.loc[:, df_train.columns != 'Exited'])))
print(classification_report(df_train.Exited,  SVM_POL.predict(df_train.loc[:, df_train.columns != 'Exited'])))
print(classification_report(df_train.Exited,  RF.predict(df_train.loc[:, df_train.columns != 'Exited'])))


### PLOTTING
##y = df_train.Exited
##X = df_train.loc[:, df_train.columns != 'Exited']
##X_pol2 = df_train_pol2
##auc_log_primal, fpr_log_primal, tpr_log_primal = get_auc_scores(y, log_primal.predict(X),log_primal.predict_proba(X)[:,1])
##auc_log_pol2, fpr_log_pol2, tpr_log_pol2 = get_auc_scores(y, log_pol2.predict(X_pol2),log_pol2.predict_proba(X_pol2)[:,1])
##auc_SVM_RBF, fpr_SVM_RBF, tpr_SVM_RBF = get_auc_scores(y, SVM_RBF.predict(X),SVM_RBF.predict_proba(X)[:,1])
##auc_SVM_POL, fpr_SVM_POL, tpr_SVM_POL = get_auc_scores(y, SVM_POL.predict(X),SVM_POL.predict_proba(X)[:,1])
##auc_RF, fpr_RF, tpr_RF = get_auc_scores(y, RF.predict(X),RF.predict_proba(X)[:,1])

##plt.figure(figsize = (12,6), linewidth= 1)
##plt.plot(fpr_log_primal, tpr_log_primal, label = 'log primal Score: ' + str(round(auc_log_primal, 5)))
##plt.plot(fpr_log_pol2, tpr_log_pol2, label = 'log pol2 score: ' + str(round(auc_log_pol2, 5)))
##plt.plot(fpr_SVM_RBF, tpr_SVM_RBF, label = 'SVM RBF Score: ' + str(round(auc_SVM_RBF, 5)))
##plt.plot(fpr_SVM_POL, tpr_SVM_POL, label = 'SVM POL Score: ' + str(round(auc_SVM_POL, 5)))
##plt.plot(fpr_RF, tpr_RF, label = 'RF score: ' + str(round(auc_RF, 5)))
##plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
##plt.xlabel('False positive rate')
##plt.ylabel('True positive rate')
##plt.title('ROC Curve')
##plt.legend(loc='best')
###plt.savefig('roc_results_ratios.png')
##plt.show()


####___DEPLOY auf TESTDATEN____


df_test = DfPrepPipeline(df_test,df_train.columns,minVec,maxVec)
df_test = df_test.mask(np.isinf(df_test))
df_test = df_test.dropna()
print(df_test.shape)

print(classification_report(df_test.Exited,  RF.predict(df_test.loc[:, df_test.columns != 'Exited'])))

auc_RF_test, fpr_RF_test, tpr_RF_test = get_auc_scores(df_test.Exited, RF.predict(df_test.loc[:, df_test.columns != 'Exited']),
                                                       RF.predict_proba(df_test.loc[:, df_test.columns != 'Exited'])[:,1])
plt.figure(figsize = (12,6), linewidth= 1)
plt.plot(fpr_RF_test, tpr_RF_test, label = 'RF score: ' + str(round(auc_RF_test, 5)))
plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
#plt.savefig('roc_results_ratios.png')
plt.show()
