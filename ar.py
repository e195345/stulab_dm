#ライブラリのインポート
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.api import AR
from statsmodels.tsa import ar_model

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
df['change'] = Y.pct_change()
"""
for i in range(20):
    model = ar_model.AR(df['change'][1:])
    maxlag = i+1
    results = model.fit(maxlag=maxlag)
    print(f'lag = {i+1}, aic : {results.aic}')
#この結果より適切であるのは lag=1
#したがって AR(1)
"""
model = ar_model.AR(df['change'][1:])
result1 = model.fit(maxlag=1)

# 残差
resid1 = result1.resid
#print(model.select_order(15).summary())

#トレーニングデータとテストデータに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 0)

#VARモデルの構築
ar = AR(Y_train)
#print(model.select_order(15).summary())

#学習
res = ar.fit(maxlags=15, ic='aic')

#予測
#ar_predict = ar.predict(0,100)

#結果
print(res.summary())

#精度
#print("精度 : ")

