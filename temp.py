import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
 
from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import RobustScaler , StandardScaler
from sklearn.linear_model import LinearRegression, Ridge , Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



import xgboost as xgb


import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()



column_name= ["MPG", "Cylinders", "Dispacement", "Horsepower", "Weight", 
              "Accleration", "Model Year", "Origin"]


data= pd.read_csv("C:/Users/AYGENN/OneDrive/Masaüstü/staj/auto-mpg.data",
                  names=column_name, na_values="?", comment="\t", sep= " ",
                  skipinitialspace= True)

data= data.rename (columns = {"MPG":"target"})

print(data.head())
print("Data shape: " , data.shape)

data.info()


describe= data.describe()



print(data.isna().sum())


data["Horsepower"] = data ["Horsepower"].fillna(data["Horsepower"].mean())

print(data.isna().sum())


sns.displot(data.Horsepower)


corr_matrix = data.corr()
sns.clustermap(corr_matrix,annot=True, fmt=".2f")

plt.title("Correlation btw features")

plt.show()

threshold = 0.75

filtre = np.abs (corr_matrix["target"])>threshold
corr_features =corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(),annot=True, fmt="0.2f")
plt.title("corealtion btw feauteres 2")
plt.show()


sns.pairplot(data, diag_kind="kde", markers="+")
plt.show()

plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_counts())


for c  in data.columns:
    
    plt.figure()
    sns.boxplot(x=c, data=data, orient="v")
    
    
    
    
    
thr = 2

horsepower_desc = describe["Horsepower"]
q3_hp=horsepower_desc[6]
q1_hp=horsepower_desc[4]
IQR_hp=q3_hp - q1_hp
top_limit_hp=q3_hp + thr*IQR_hp
bottom_limit_hp=q1_hp - thr*IQR_hp

filter_hp_bottom=bottom_limit_hp< data["Horsepower"]
filter_hp_top=data["Horsepower"]< top_limit_hp

filter_hp=filter_hp_bottom & filter_hp_top

data= data[filter_hp]

    
thr = 2
accleration_desc = describe["Accleration"]
q3_acc=accleration_desc[6]
q1_acc=accleration_desc[4]
IQR_acc=q3_acc - q1_acc
top_limit_acc=q3_acc + thr*IQR_acc
bottom_limit_acc=q1_acc - thr*IQR_acc

filter_acc_bottom=bottom_limit_acc< data["Accleration"]
filter_acc_top=data["Accleration"]< top_limit_acc

filter_acc=filter_acc_bottom & filter_acc_top

data= data[filter_acc]



sns.distplot(data.target, fit=norm)

(mu, sigma)=norm.fit(data ["target"])
print("mu: {}, sigma= {}".format(mu, sigma))


plt.figure()
stats.probplot(data["target"], plot=plt)

plt.show()


data["target"]=np.log1p(data["target"])
sns.distplot(data.target, fit=norm)
plt.figure()


(mu, sigma)=norm.fit(data ["target"])
print("mu: {}, sigma= {}".format(mu, sigma))

plt.figure()
stats.probplot(data["target"], plot=plt)

plt.show()

skewed_feats=data.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness=pd.DataFrame(skewed_feats, columns =["skeewed"])



data["Cylinder"]= data["Cylinders"].astype(str)
data["Origin"]= data["Origin"].astype(str)
data =pd.get_dummies(data)


x=data.drop(["target"], axis=1)

y=data.target
test_size=0.9
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=test_size, random_state=42)




scaler= StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


lr= LinearRegression()

lr.fit(X_train, Y_train)

print("LR coef: ", lr.coef_)

y_predection_dummy =lr.predict(X_test)

mse = mean_squared_error(Y_test, y_predection_dummy)
print("Linear Regression MSE: ", mse)


ridge= Ridge(random_state=42, max_iter=10000)

alphas=np.logspace(-4, -0.5, 30)

tuned_parameters= [{'alpha':alphas}]
n_folds = 5

clf= GridSearchCV(ridge, tuned_parameters, cv=n_folds, scoring="neg_mean_squared_error",refit=True)

clf.fit(X_train, Y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std=clf.cv_results_["std_test_score"]

print("Ridge Coef: ",clf.best_estimator_.coef_ )

ridge= clf.best_estimator_

print("Ridge Best Estimator:", ridge)

y_predicted_dummy=clf.predict(X_test)

mse=mean_squared_error(Y_test, y_predicted_dummy)

print("Ridge mse: ", mse)

print("--------------------")

plt.figure()

plt.semilogx(alphas, scores)

plt.xlabel("alpha")








lasso= Lasso(random_state=42, max_iter=10000)

alphas=np.logspace(-4, -0.5, 30)

tuned_parameters= [{'alpha':alphas}]
n_folds = 5

clf= GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring="neg_mean_squared_error",refit=True)

clf.fit(X_train, Y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std=clf.cv_results_["std_test_score"]

print("Lasso Coef: ",clf.best_estimator_.coef_ )

lasso= clf.best_estimator_

print("Lasso Best Estimator:", lasso)

y_predicted_dummy=clf.predict(X_test)

mse=mean_squared_error(Y_test, y_predicted_dummy)

print("Lasso mse: ", mse)

print("--------------------")

plt.figure()

plt.semilogx(alphas, scores)

plt.xlabel("alpha")

plt.ylabel("score")
plt.title("Lasso")

plt.ylabel("score")
plt.title("Lasso")



parametersGrid= {"alpha" : alphas,
                 "l1_ratio": np.arange(0.0, 1.0, 0.05)}

eNet = ElasticNet(random_state=42, max_iter=10000)
clf= GridSearchCV(eNet, tuned_parameters, cv=n_folds, scoring="neg_mean_squared_error",refit=True)

clf.fit(X_train, Y_train)


print("ElasticNet Coef: ",clf.best_estimator_.coef_ )

print("ElasticNet Best Estimator:", clf.best_estimator_)

y_predicted_dummy = clf.predict(X_test)
mse=mean_squared_error(Y_test, y_predicted_dummy)
print("ElasticNet mse: ", mse)



model_xgb=xgb.XGBRFRegressor(objective='reg:linear', max_depth=5, min_child_weight=4, subsample=0.7, n_estimators=1000, learning_rate=0.07)

model_xgb.fit(X_train, Y_train)

y_predicted_dummy= model_xgb.predict(X_test)
mse=mean_squared_error(Y_test, y_predicted_dummy)
print("XGBRegression MSE:", mse)




































