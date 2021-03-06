import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.svm import LinearSVC
import numpy as np

if __name__ == "__main__":
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

    label = np.nan
    i = len(df)-1
    while i > 0:
        if i == len(df)-1:
            label = df.at[i, '天気']
            df.at[i, '天気'] = np.nan
        else:
            hokan = df.at[i, '天気']
            df.at[i, '天気'] = label
            label = hokan
        i -= 1

    df = df.dropna(how='any').reset_index(drop=True)

    X = df[['気温', '降水量', '雲量', '湿度']]
    Y = df['天気']

    sc = preprocessing.StandardScaler()
    sc.fit(X)
    X_normal = pd.DataFrame(sc.transform(X))
    X_normal.columns = ['気温', '降水量', '雲量', '湿度']

    # トレーニングデータとテストデータに分ける
    partition = int(len(df)*0.8)
    X_train, X_test = X_normal[:partition], X_normal[partition:]
    Y_train, Y_test = Y[:partition], Y[partition:]

    # 学習させる
    clf_result = LinearSVC(intercept_scaling=1.0, max_iter=10000000)
    clf_result.fit(X_train, Y_train)
    # 正答率
    pre = clf_result.predict(X_test)
    ac_score = metrics.accuracy_score(Y_test, pre)
    print('正答率：{:.3f}%'.format(ac_score*100))

    # 混合行列
    Mixed_matrix = metrics.confusion_matrix(Y_test, pre)
    print(Mixed_matrix)
