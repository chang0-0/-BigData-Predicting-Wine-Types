import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import linalg
from scipy import stats
from sklearn.linear_model import LinearRegression
# 자동으로 표준화를 해주는 함수가 있음.
from sklearn.preprocessing import StandardScaler
import random
standardScaler = StandardScaler()

wine = pd.read_csv('wine.csv', sep=',', header=0)

wine.columns = wine.columns.str.replace(' ','_').str.replace("\'", "")

wine['type'] = np.where(wine['class'] == 0.0, 'red', 'white')
print("================================================= wine head 위에서 5줄까지 출력 =================================================")
print(wine.head())
print("================================================= wine info wine.csv 데이터 타입 표현 =================================================")
print(wine.info())

# unstack 함수를 추가하여 그 결과를 가로 방향으로 재구조화한다.
print("================================================= wine describe =================================================")
print(wine.describe())

print(pd.unique(wine['class'])) #0 1
wine_input = wine[['alcohol', 'sugar', 'pH']].to_numpy()
print(wine_input[:5])

wine_target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, random_state=40)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)



from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

print(kn.classes_)
print(kn.predict(test_scaled[:5]))


proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])

# 로지스틱회귀 분석 모델 그래프 그림
# z = np.arange(-5, 5, 0.1)
# phi = 1 / (1 + np.exp(-z))

# plt.plot(z, phi)
# plt.xlabel('z')
# plt.ylabel('phi')
# plt.show()

wine_indexes = (train_target == 0) | (train_target == 1)
train_wine = train_scaled[wine_indexes]
target_wine = train_target[wine_indexes]

from sklearn.linear_model import LogisticRegression

logit_model = LogisticRegression()
logit_model.fit(train_wine, target_wine)

print(logit_model.predict(train_wine[:5]))
print(logit_model.predict_proba(train_wine[:5]))
print(logit_model.classes_)
print(logit_model.coef_, logit_model.intercept_)

decisions = logit_model.decision_function(train_wine[:5])
print(decisions)

