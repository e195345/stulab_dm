import pandas as pd
import math
from sklearn import metrics, preprocessing
from sklearn.svm import LinearSVC

# データ読み込み
temperature_df = pd.read_csv("temp.csv", skiprows=3)
precipitation_df = pd.read_csv("precipitation.csv", skiprows=3)
weather_df = pd.read_csv("weather.csv", skiprows=3)
cloud_cover_df = pd.read_csv("cloud.csv", skiprows=3)

temperature_df.columns = [u'年月日時', u'気温', u'気温_品質']
precipitation_df.columns = [u'年月日時', u'降水量', u'降水量_品質', u'降水量_品質']
weather_df.columns = [u'年月日時', u'label', u'天気_品質']
cloud_cover_df.columns = [u'年月日時', u'雲量', u'雲量_品質']

df_t_p = pd.merge(temperature_df, precipitation_df, on='年月日時')
df_t_p_c = pd.merge(df_t_p, cloud_cover_df, on='年月日時')
df = pd.merge(df_t_p_c, weather_df, on='年月日時')

weather_num = 1.0
cloud_num = 1.0

# nanの処理
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

clf_result = LinearSVC(max_iter=1000000)
clf_result.fit(X_train, Y_train)

# 正答率
pre = clf_result.predict(X_test)
ac_score = metrics.accuracy_score(Y_test, pre)
print('{:.3f}%'.format(ac_score))

# 混合行列
Mixed_matrix = metrics.confusion_matrix(Y_test, pre)
print("標準化")
print(Mixed_matrix)


