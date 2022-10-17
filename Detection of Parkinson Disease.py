from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
d=read_csv('parkinsons1.csv')
array=d.values
f=array[:,1:23]
Y=array[:,23]
scaler=StandardScaler().fit(f)
X=scaler.transform(f)
Y=Y.astype('int')
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=7)
"""models=[]
models.append(('KNN',KNeighborsClassifier()))
models.append(('SVC',SVC()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('NB',GaussianNB()))
#models.append(('Logistic',LogisticRegression()))
for name,model in models:
    kfold=ShuffleSplit(n_splits=10,test_size=0.2,random_state=7)
    scoring='accuracy'
    cv=cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    #print(f"{name} : {cv.mean()*100.0}")
#LinearDiscriminantAnalysis and DecisionTreeClassifier and KNN are three of the Classifiers having highest Accuracies
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
pipeline=[]
pipeline.append(('ScaledKNN',Pipeline([('StandardScale',StandardScaler()),('KNN',KNeighborsClassifier(n_neighbors=1))])))
pipeline.append(('ScaledSVC',Pipeline([('StandardScale',StandardScaler()),('SVC',SVC())])))
pipeline.append(('ScaledCART',Pipeline([('StandardScale',StandardScaler()),('CART',DecisionTreeClassifier())])))
pipeline.append(('ScaledLDA',Pipeline([('StandardScale',StandardScaler()),('LDA',LinearDiscriminantAnalysis())])))
pipeline.append(('ScaledNB',Pipeline([('StandardScale',StandardScaler()),('NB',GaussianNB())])))
for name,model in pipeline:
  kfold=ShuffleSplit(n_splits=10,test_size=0.2,random_state=7)
  scoring='accuracy'
  cv=cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
  print(f"{name} : {cv.mean()*100.0}")
#ScaledSVC and ScaledKNN are two of the Classifiers having highest Accuracies
#Tuning SVC
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold=ShuffleSplit(n_splits=10,test_size=0.2,random_state=7)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#TuningKNN
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier()
kfold=ShuffleSplit(n_splits=10,test_size=0.2,random_state=7)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#As we can see that KNN has an accuracy of 91.875% which is almost 92. We can see that the optimal configuration is K=1.
ensembles=[]
ensembles.append(('ADA',AdaBoostClassifier()))
ensembles.append(('GBA',GradientBoostingClassifier()))
ensembles.append(('RF',RandomForestClassifier()))
ensembles.append(('ET',ExtraTreesClassifier()))
for name,model in ensembles:
  kfold=ShuffleSplit(n_splits=10,test_size=0.2,random_state=7)
  scoring='accuracy'
  cv=cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
  print(f"{name} : {cv.mean()*100.0}")"""
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model=KNeighborsClassifier(n_neighbors=1)
model.fit(rescaledX,Y_train)
rescaledValidationX = scaler.transform(X_test)
predictions = model.predict(rescaledValidationX)
print("Accuracy of the Model is: ",(accuracy_score(Y_test, predictions))*100,"%")
matrix=DataFrame(confusion_matrix(Y_test,predictions),columns=['Predicted Healthy', 'Predicted Parkinsons'],index=['True Healthy', 'True Parkinsons'])
print(matrix)
print(classification_report(Y_test, predictions))