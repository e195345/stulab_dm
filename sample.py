import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import GridSearchCV

parameters = [{'max_iter': 1000000, 'C': np.logspace(0, 10, 100), 'penalty': 'l1', 'dual': False, 'multi_class': {'ovr', 'crammer_singer'},
               'class_weight': {'None', 'balanced'}},
              {'max_iter': 1000000, 'C': np.logspace(0, 10, 100), 'multi_class': {'ovr', 'crammer_singer'}, 'loss': 'hinge', 'class_weight':
                  {'None', 'balanced'}}]


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

sc = preprocessing.StandardScaler()
sc.fit(X)
X_normal = pd.DataFrame(sc.transform(X))
X_normal.columns = ['気温', '降水量', '雲量']

# トレーニングデータとテストデータに分ける
partition = int(len(df)*0.8)
X_train, X_test = X_normal[:partition], X_normal[partition:]
Y_train, Y_test = Y[:partition], Y[partition:]


clf = GridSearchCV(LinearSVC(), parameters, n_jobs=-1, cv=5)
clf.fit(X_train, Y_train)
print("ベストパラメータ")
print(clf.best_estimator_)
print("--------通常-------")
clf_result = LinearSVC(max_iter=1000000, C=5.0, penalty='l1', dual=False)
clf_result.fit(X_train, Y_train)

# 正答率
pre = clf_result.predict(X_test)
ac_score = metrics.accuracy_score(Y_test, pre)
print('正答率：{:.3f}%'.format(ac_score*100))

# 混合行列
Mixed_matrix = metrics.confusion_matrix(Y_test, pre)
print("標準化")
print(Mixed_matrix)
