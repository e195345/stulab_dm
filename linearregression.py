#ライブラリのインポート
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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

#線形回帰モデルの構築 学習
lr = LinearRegression(fit_intercept=False)
lr.fit(X_train, Y_train)
print("説明変数の係数")
print('coefficient = ', lr.coef_[0]) # 説明変数の係数

#予測
Y_pred = lr.predict(X_test) # 検証データを用いて目的変数を予測

#平均２乗誤差
from sklearn.metrics import mean_squared_error
Y_train_pred = lr.predict(X_train) # 学習データに対する目的変数を予測
print("平均２乗誤差")
print('MSE train data: ', mean_squared_error(Y_train, Y_train_pred)) # 学習データを用いたときの平均二乗誤差を出力
print('MSE test data: ', mean_squared_error(Y_test, Y_pred))         # 検証データを用いたときの平均二乗誤差を出力

#決定係数
from sklearn.metrics import r2_score
print("決定係数")
print('r^2 train data: ', r2_score(Y_train, Y_train_pred))
print('r^2 test data: ', r2_score(Y_test, Y_pred))

#精度
print('トレーニングデータの精度 :', lr.score(X_train, Y_train))
print('テストデータの精度 : ', lr.score(X_test, Y_test))
