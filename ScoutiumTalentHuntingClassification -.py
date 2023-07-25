################################   Scoutium Talent Hunting Classification   ################################


# Business Problem : Prediction of class of football players (average or highlighted) based on features of players


# Dataset :

# Scouts evaluate football players according to some features in matches.
# The data set consists of information about the evaluated football players, the features scored in the match and their scores.
# attributes: The points of feature given by users for each player  in a match.(bağımsız değişkenler)
# potential_labels: Potential tags that contain users' final opinions about the players in each match (target variable)

# task_response_id: The set of a scout's assessments of all players on a team's roster in a match.
# match_id: The id of the relevant match
# evaluator_id: The id of relevant scout
# player_id: The id of relevant player
# position_id: The id of the position played by the relevant player in the match.

# 1. Goalkeeper
# 2. Centre-back
# 3. Right-back
# 4. Left-back
# 5. Defensive midfielder
# 6. Central midfielder
# 7. Right-winger
# 8. Left-winger
# 9. Central attacking midfielder
# 10. Striker

# analysis_id: A set containing the scout's evaluations of a player's features in a match
# attribute_id: The id of the evaluated player attributes
# attribute_value: Bir yetenek avcısı tarafından bir oyuncunun niteliğine verilen bir puan
# potential_label: Potential tags that contain users' final opinions about the players in each match



import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

### TASK 1 : Data preparation

df = pd.read_csv(r"C:\Users\burcu\OneDrive\Masaüstü\DS Miiul\6.Machine Learning\ödev\son hafta - Gözetimsiz\Scoutium\scoutium_attributes.csv", sep=";")

df2 = pd.read_csv(r"C:\Users\burcu\OneDrive\Masaüstü\DS Miiul\6.Machine Learning\ödev\son hafta - Gözetimsiz\Scoutium\scoutium_potential_labels.csv", sep=";")

# Merging csv files

dff = pd.merge(df, df2, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])
dff.head()


# Removal of the Keeper (1) class in position_id
dff = dff[dff["position_id"] != 1]


# Removal of below_average class in potential_label from dataset beceause below average class is rare
dff = dff[dff["potential_label"] != "below_average"]


# Attribute value - Attribute_id Pivot table
pt = pd.pivot_table(dff, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])
pt

# Correcting indexes and converting columns to strings
pt = pt.reset_index(drop=False)
pt.columns = pt.columns.map(str)


# “num_cols” : A new list includes numeric variable columns
num_cols = pt.columns[3:]


### TASK 2 : EDA
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(pt)


# Analysis of categorical variables

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in ["position_id","potential_label"]:
    cat_summary(pt, col)



# Analysis of numerical variables

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(pt, col, plot=True)


# Analysis of numerical variables according to target variable

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(pt, "potential_label", col)


# Correlation
pt[num_cols].corr()

# Correlation Matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(pt[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()



### TASK 3 : Feature Extraction

pt["min"] = pt[num_cols].min(axis=1)
pt["max"] = pt[num_cols].max(axis=1)
pt["sum"] = pt[num_cols].sum(axis=1)
pt["mean"] = pt[num_cols].mean(axis=1)
pt["median"] = pt[num_cols].median(axis=1)


pt["mentality"] = pt["position_id"].apply(lambda x: "defender" if (x == 2) | (x == 5) | (x == 3) | (x == 4) else "attacker")

for i in pt.columns[3:-6]:
    threshold = pt[i].mean() + pt[i].std()

    lst = pt[i].apply(lambda x: 0 if x < threshold else 1)
    pt[str(i) + "_FLAG"] = lst

flagCols = [col for col in pt.columns if "_FLAG" in col]

pt["counts"] = pt[flagCols].sum(axis=1)

pt["countRatio"] = pt["counts"] / len(flagCols)

pt.head()


### TASK 4 : Label Encoder and Standard Scaler

#Applying label encoder for binary cols

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


labelEncoderCols = ["potential_label","mentality"]

for col in labelEncoderCols:
    pt = label_encoder(pt, col)


# Applying standard scaler for scaling data in "num_cols" variables

pt.head()
lst = ["counts", "countRatio","min","max","sum","mean","median"]
num_cols = list(num_cols)

for i in lst:
    num_cols.append(i)

scaler = StandardScaler()
pt[num_cols] = scaler.fit_transform(pt[num_cols])

pt.head()


### TASK 5 : A machine learning model that predicts potential label of football players with minimal error

y = pt["potential_label"]
X = pt.drop(["potential_label", "player_id"], axis=1)


models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   #("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   #('CatBoost', CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier())]



for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))

"""

LR
    roc_auc score:0.8650216450216451
  f1 score:0.5675324675324676
 precision score:0.6983333333333335
 recall score:0.5266666666666666
  accuracy score:0.8558201058201057



KNN
  roc_auc score:0.7331349206349207
 f1 score:0.44058441558441563
  precision score:0.78
r recall score:0.32666666666666666
  accuracy score:0.8485449735449734
  
RF
roc_auc score:0.8989393939393938
f1 score:0.5531746031746031
precision score:0.975
recall score:0.45666666666666667
accuracy score:0.8673280423280424

GBM
roc_auc score:0.8552813852813854
f1 score:0.521955266955267
precision score:0.6716666666666666
recall score:0.45666666666666667
accuracy score:0.8488095238095239

XGBoost
roc_auc score:0.8772077922077921
f1 score:0.65997113997114
precision score:0.7905555555555556
recall score:0.61
accuracy score:0.8747354497354498

LightGBM
roc_auc score:0.8934415584415584
f1 score:0.6567427017427018
precision score:0.8280952380952382
recall score:0.5799999999999998
accuracy score:0.8855820105820106
"""



### Task 6:  Hyperparameter Optimization

lgbm_model = LGBMClassifier(random_state=46)

#rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)



final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse

"""
rmse
0.3364668388277351

"""



# Feature importance

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)


