import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/hitters.csv")
df.head()

def check_df(dataFrame, head=5):
    print("####### Shape #######")
    print(dataFrame.shape)
    print("####### Type ########")
    print(dataFrame.dtypes)
    print("####### Head ########")
    print(dataFrame.head(head))
    print("####### Tail ########")
    print(dataFrame.tail(head))
    print("####### NA ########")
    print(dataFrame.isnull().sum())
    print("####### Quantiles ########")
    print(dataFrame.describe().T)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in df:
    print(f'{col} : {df[col].nunique()}')

# Kategorik ve numeric değişkenlerin analizi

def cat_summary(dataframe, col_name, plot=True):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),  #değişkende hangi degerden kacar adet var?
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot= False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("***********************************")

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols:
    num_summary(df, col, True)

df[num_cols].plot(kind='box')
plt.xticks(rotation=30, horizontalalignment='right')

# Analysis of target variable

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

# Analysis of Correlation

df[num_cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(25,10))
sns.heatmap(df[num_cols].corr(), annot=True, linewidths=.5, ax=ax)
plt.show()

# correlation with the final state of the variables
plt.figure(figsize=(45,45))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(df[num_cols].corr(), mask=mask, cmap='coolwarm', vmax=.3, center=0,
            square=True, linewidths=.5,annot=True)
plt.show(block=True)



def find_correlation(dataframe, numeric_cols, corr_limit = 0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "Salary":
            pass
        else:
            correlation = dataframe[[col, "Salary"]].corr().loc[col, "Salary"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df, num_cols)

#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# Outliers

sns.boxplot(x=df["Salary"], data=df)
plt.show(block=True)

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))


for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


# Missing Values

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)
df.columns
df1 = df.copy()
df1.head()
cat_cols, num_cols, cat_but_car = grab_col_names(df1)
df1.columns
df1 = pd.get_dummies(df1[cat_cols+num_cols], drop_first=True)
scaler = RobustScaler()
df1 = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)
imputer = KNNImputer(n_neighbors=5)
df1 = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)
df1 = pd.DataFrame(scaler.inverse_transform(df1), columns=df1.columns)

# Feature Extraction

new_num_cols=[col for col in num_cols if col!="Salary"]
df1[new_num_cols]=df1[new_num_cols]+0.0000000001

df1["Hits_Success"] = (df1["Hits"] / df1["AtBat"]) * 100
df1['NEW_RBI'] = df1['RBI'] / df1['CRBI']
df1['NEW_Walks'] = df1['Walks'] / df1['CWalks']
df1['NEW_PutOuts'] = df1['PutOuts'] * df1['Years']
df1['NEW_Hits'] = df1['Hits'] / df1['CHits'] + df1['Hits']
df1["NEW_CRBI*CATBAT"] = df1['CRBI'] * df1['CAtBat']
df1["NEW_Chits"] = df1["CHits"] / df1["Years"]
df1["NEW_CHmRun"] = df1["CHmRun"] * df1["Years"]
df1["NEW_CRuns"] = df1["CRuns"] / df1["Years"]
df1["NEW_Chits"] = df1["CHits"] * df1["Years"]
df1["NEW_RW"] = df1["RBI"] * df1["Walks"]
df1["NEW_RBWALK"] = df1["RBI"] / df1["Walks"]
df1["NEW_CH_CB"] = df1["CHits"] / df1["CAtBat"]
df1["NEW_CHm_CAT"] = df1["CHmRun"] / df1["CAtBat"]
df1['NEW_Diff_Atbat'] = df1['AtBat'] - (df1['CAtBat'] / df1['Years'])
df1['NEW_Diff_Hits'] = df1['Hits'] - (df1['CHits'] / df1['Years'])
df1['NEW_Diff_HmRun'] = df1['HmRun'] - (df1['CHmRun'] / df1['Years'])
df1['NEW_Diff_Runs'] = df1['Runs'] - (df1['CRuns'] / df1['Years'])
df1['NEW_Diff_RBI'] = df1['RBI'] - (df1['CRBI'] / df1['Years'])
df1['NEW_Diff_Walks'] = df1['Walks'] - (df1['CWalks'] / df1['Years'])


#############################################
#  Feature Scaling (Özellik Ölçeklendirme)

cat_cols, num_cols, cat_but_car = grab_col_names(df1)

num_cols = [col for col in num_cols if col not in ["Salary"]]
scaler = StandardScaler()
df1[num_cols] = scaler.fit_transform(df1[num_cols])
df1.head()


# Correlation Analysis
fig, ax = plt.subplots(figsize=(25,10))
sns.heatmap(df1.corr(), annot=True, linewidths=.5, ax=ax)
plt.show()


#############################################
#               MODELING                    #
#############################################
df1.isnull().sum().sum()

y = df1["Salary"]
X = df1.drop(["Salary"], axis=1)

X.shape
y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=46)

linreg = LinearRegression()
model = linreg.fit(X_train, y_train)
y_pred = model.predict(X_train)

# RMSE, modelin gerçek değerler ile tahmin edilen değerler arasındaki farkların karelerinin ortalamasının kareköküdür
lin_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
print("LINEAR REGRESSION TRAIN RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_train, y_pred))))

# R-kare, bağımlı değişkenin varyansının bağımsız değişkenler tarafından açıklanan yüzdesidir.
lin_train_2 = linreg.score(X_train, y_train)
print("LINEAR REGRESSION TRAIN R-SQUARED:", "{:,.3f}".format(linreg.score(X_train, y_train)))

y_pred = model.predict(X_test)
lin_test_rmse =np.sqrt(mean_squared_error(y_test, y_pred))
print("LINEAR REGRESSION TEST RMSE:", "{:,.2f}".format(np.sqrt(mean_squared_error(y_test,y_pred))))

lin_test_r2 = linreg.score(X_test,y_test)
print("LINEAR REGRESSION TEST R-SQUARED:", "{:,.3f}".format(linreg.score(X_test,y_test)))


# Test part regplot:
g = sns.regplot(x=y_test, y=y_pred, scatter_kws={'color': 'b', 's': 5},
                ci=False, color="r")
g.set_title(f"Test Model R2: = {linreg.score(X_test, y_test):.3f}")
g.set_ylabel("Predicted Salary")
g.set_xlabel("Salary")
plt.xlim(-5, 2700)
plt.ylim(bottom=0)
plt.show(block=True)

print("LINEAR REGRESSION CROSS_VAL_SCORE:", "{:,.3f}".format(np.mean(np.sqrt(-cross_val_score(model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))))

#OLS for Linear Regression

import statsmodels.api as sm

#Adding a constant to the model (necessary for statsmodels)
X_train_sm = sm.add_constant(X_train)
# Fitting the model using statsmodels
model_sm = sm.OLS(y_train, X_train_sm).fit()
# Getting the summary of the regression model
model_summary = model_sm.summary()
# return model_summary

