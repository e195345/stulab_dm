import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    '''
    LinearSVCのパラメータ一覧
    penalty{‘l1’, ‘l2’}, default=’l2’
    loss{‘hinge’, ‘squared_hinge’}, default=’squared_hinge’
    dual bool, default=True
    tol float, default=1e-4
    C float, default=1.0
    multi_class{‘ovr’, ‘crammer_singer’}, default=’ovr’
    fit_intercept bool, default=True
    intercept_scaling float, default=1
    class_weight dict or ‘balanced’, default=None
    verbose int, default=0
    random_state int, RandomState instance or None, default=None
    max_iter int, default=1000
    '''
    parameters = [{'max_iter': [100000000], 'C': np.logspace(0, 1, 10), 'penalty': ['l2'], 'dual': [True], 'multi_class': ['ovr'], 'intercept_scaling': np.logspace(0, 1, 10)},
                  {'max_iter': [100000000], 'C': np.logspace(0, 1, 10), 'penalty': ['l1'], 'dual': [False], 'multi_class': ['ovr']},
                  {'max_iter': [100000000], 'C': np.logspace(0, 1, 10), 'loss': ['hinge']}
                  ]
    df = pd.read_csv("./1year.csv", skiprows=1).dropna(how='any').reset_index(drop=True)
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

    X = df[['気温', '降水量', '雲量', '湿度']]
    Y = df['label']

    sc = preprocessing.StandardScaler()
    sc.fit(X)
    X_normal = pd.DataFrame(sc.transform(X))
    X_normal.columns = ['気温', '降水量', '雲量', '湿度']

    partition = int(len(df)*0.8)
    X_train, X_test = X_normal[:partition], X_normal[partition:]
    Y_train, Y_test = Y[:partition], Y[partition:]

    # パラメータの最適化を行う
    clf = GridSearchCV(LinearSVC(), parameters, n_jobs=-1, cv=5)
    clf.fit(X_train, Y_train)
    print("ベストパラメータ")
    print(clf.best_estimator_)
