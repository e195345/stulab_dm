import pandas as pd
import math
from sklearn import metrics, preprocessing
from sklearn.svm import LinearSVC
import numpy as np
import collections
import seaborn as sns
import matplotlib.pyplot as plt

# データ読み込み
temperature_df = pd.read_csv("temperature.csv", skiprows=5, header=None, names=['年月日時', '気温', '気温_品質'])
rain_df = pd.read_csv("rain.csv", skiprows=5, header=None, names=['年月日時', '降水量', '降水量_品質', '降水量_情報'])
weather_df = pd.read_csv("weather.csv", skiprows=5, header=None, names=['年月日時', 'label', '天気_品質'])
cloud_df = pd.read_csv("cloud.csv", skiprows=5, header=None, names=['年月日時', '雲量', '雲量_品質'])

df_t_r = pd.merge(temperature_df, rain_df, on='年月日時')
df_t_r_c = pd.merge(df_t_r, cloud_df, on='年月日時')
df = pd.merge(df_t_r_c, weather_df, on='年月日時')

weather_num = 1.0
cloud_num = 9.5

# nanの処理
'''
for i in range(len(df)):
    if math.isnan(df.at[i, 'label']):
        df.at[i, 'label'] = weather_num
    else:
        weather_num = df.at[i, 'label']

for i in range(len(df)):
    if math.isnan(df.at[i, '雲量']):
        df.at[i, '雲量'] = cloud_num
    else:
        cloud_num = df.at[i, '雲量']
'''

rain_num = 2.0
for i in range(len(df)):
    if (df.at[i, '降水量'] < rain_num) & (df.at[i, 'label'] == 10.0):
        df.at[i, '降水量'] = rain_num

# 気温の差分で学習
previous_day_temperature = 0.0
for i in range(len(df)):
    if i == 0:
        previous_day_temperature = df.at[i, '気温']
        df.at[i, '気温'] = np.nan
    else:
        temperature_difference = df.at[i, '気温'] - previous_day_temperature
        previous_day_temperature = df.at[i, '気温']
        df.at[i, '気温'] = temperature_difference

df = df[['気温', '降水量', '雲量', 'label']]
df = df.dropna(how='any')

X = df[['気温', '降水量', '雲量']]
Y = df['label']

sc = preprocessing.StandardScaler()
sc.fit(X)
X_normal = pd.DataFrame(sc.transform(X))
X_normal.columns = ['気温', '降水量', '雲量']

#
half = int(len(df)*0.8)
X_train, X_test = X_normal[:half], X_normal[half:]
Y_train, Y_test = Y[:half], Y[half:]

clf_result = LinearSVC(max_iter=1000000, C=5.0)
clf_result.fit(X_train, Y_train)

# 正答率
pre = clf_result.predict(X_test)
ac_score = metrics.accuracy_score(Y_test, pre)
print('正答率：{:.3f}%'.format(ac_score*100))

# 混合行列
Mixed_matrix = metrics.confusion_matrix(Y_test, pre)
print("標準化")
print(Mixed_matrix)



