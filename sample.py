import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.svm import LinearSVC


df = pd.read_csv("data.csv", skiprows=3)
# 年月日時,気温(℃),気温(℃),天気,天気,降水量(mm),降水量(mm),降水量(mm),雲量(10分比),雲量(10分比)
df.columns = [u'年月日時', u'気温', u'気温.品質', u'label', u'label.品質', u'降水量', u'降水量.情報', u'降水量.品質', u'雲量', u'雲量.品質']
df = df.dropna(how='any')
X = df[['気温', '降水量']]
print(X.describe())
Y = df['label']

sc = preprocessing.StandardScaler()
sc.fit(X)
X_normal = pd.DataFrame(sc.transform(X))
X_normal.columns = ['気温', '降水量']

half = 150
print(half)

X_train, X_test = X_normal[:half], X_normal[half:]
Y_train, Y_test = Y[:half], Y[half:]

print(X.shape)
clf_result = LinearSVC(max_iter=1000000)
print(X.shape)
clf_result.fit(X_train, Y_train)
print(X.shape)

pre = clf_result.predict(X_test)
ac_score = metrics.accuracy_score(Y_test, pre)
print(ac_score)

Mixed_matrix = metrics.confusion_matrix(Y_test, pre)
print("標準化")
print(Mixed_matrix)

