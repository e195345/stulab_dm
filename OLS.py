#ライブラリのインポート
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split

# データ読み込み
df = pd.read_csv("./1year.csv", skiprows=1).dropna(how='any').reset_index(drop=True)

# 前処理
previous_day_temperature = 0.0
for i in range(len(df)):
    if i == 0:
        previous_day_temperature = df.at[i, '気温']
        df.at[i, '気温'] = np.nan
    else:
        temperature_difference = df.at[i, '気温'] - previous_day_temperature
        previous_day_temperature = df.at[i, '気温']
        df.at[i, '気温'] = temperature_difference

df = df.dropna(how='any').reset_index(drop=True)

X = df[['気温', '降水量', '雲量']]
Y = df['label']

#トレーニングデータとテストデータに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 0)

#線形回帰モデルの構築
#lr =  sm.OLS(formula,X,sbbset=None,drop_cols=None)
lr_model_1 = sm.OLS(Y,X)
lr_model_2=sm.OLS(Y_train,X_train)
lr_model_3=sm.OLS(Y_test,X_test)
lr_model_3=sm.OLS(a,)
#学習
res1 = lr_model_1.fit()
res2=lr_model_2.fit()
res3=lr_model_3.fit()

#結果
print(res2.summary())

#精度
print("1")
#print('Parameters: ', res1.params)
print('R2: ', res1.rsquared)
print("train")
#print('Parameters: ', res2.params)
print('R2: ', res2.rsquared)
print("test")
#print('Parameters: ', res3.params)
print('R2: ', res3.rsquared)
print("精度 : ")

