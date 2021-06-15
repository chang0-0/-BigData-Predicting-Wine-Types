import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import linalg
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols, glm
from sklearn.linear_model import LinearRegression
# 자동으로 표준화를 해주는 함수가 있음.
from sklearn.preprocessing import StandardScaler
import random
standardScaler = StandardScaler()

wine = pd.read_csv('wine.csv', sep=',', header=0)

wine.columns = wine.columns.str.replace(' ','_').str.replace("\'", "")

wine['type'] = np.where(wine['class'] == 0.0, 'red', 'white')
print("================================================= wine head  =================================================")
print(wine.head())
print("================================================= wine info =================================================")
print(wine.info())

print("================================================= wine describe =================================================")
print(wine.describe())

print(pd.unique(wine['class'])) #0 1
wine_input = wine[['alcohol', 'sugar', 'pH']].to_numpy()
print(wine_input[:5])

wine_target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# print("사이킷런 라이브러리 표준화")
# train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target)
# ss = StandardScaler()
# train_scaled = ss.fit_transform(train_input)
# test_scaled = ss.transform(test_input)

# #표준화 완료.
# print(test_scaled)
# print(wine_target)

# wine_indexes = (train_target == 0) | (train_target == 1)
# train_wine = train_scaled[wine_indexes]
# target_wine = train_target[wine_indexes]

# from sklearn.linear_model import LogisticRegression
# logit = LogisticRegression()
# logit.fit(train_scaled, target_wine)

print("================================================= 통계량 출력 =================================================")
print(wine.groupby(['class'])[['alcohol', 'sugar', 'pH']].agg(['count', 'mean', 'std', 'min', 'max']))

dependent_variable = wine['class']
independent_variables = wine[['alcohol', 'sugar', 'pH']]
independent_variable_standardized = (independent_variables - independent_variables.mean()) / independent_variables.std()
independent_variable_constand = sm.add_constant(independent_variable_standardized, prepend=True)
logit_model = sm.Logit(dependent_variable, independent_variable_constand).fit()
print(logit_model.params)
print(logit_model.bse)

print("\nCoefficients:\n%s" % logit_model.params)

# print(logit_model.predict(train_wine[:5]))
# print(logit_model.predict_proba(train_wine[:5]))
# print(logit_model.classes_)
# print("계수와 절편 출력")
# print(logit_model.coef_, logit_model.intercept_)

# 추가문제. 라이브러리를 사용해서 출력된값으로 Logist Regression Result를 출력해보자.
# 라이브러리 사용 결과값. 
print("y = 1.7963 + 0.5348*alcohol + 1.6700*sugar - 0.7105)*pH")

def inverse_logit(model_formula):
    from math import exp
    return (1.0 / (1.0 + exp(-model_formula)))

new_value = float(logit_model.params[0]) + \
        float(logit_model.params[1])*float(9.0) + \
        float(logit_model.params[2])*float(1.1) + \
        float(logit_model.params[3])*float(3.0)

print("\n 새로운 값을 넣어서 예측한 결과 %.2f \n" % new_value)

print(independent_variables.columns)
new_observations = wine.loc[wine.index.isin(range(10)), independent_variables.columns]
new_observations_with_constant = sm.add_constant(new_observations, prepend=True)
y_predicted = logit_model.predict(new_observations_with_constant)
print(y_predicted)

output_variable = wine['class']
vars_to_keep = wine[['alcohol', 'sugar', 'pH']]
inputs_standardized = (vars_to_keep - vars_to_keep.mean()) / vars_to_keep.std()
input_variables = sm.add_constant(inputs_standardized, prepend=True)
logit_model = sm.Logit(output_variable, input_variables).fit()
print(logit_model.summary())
print(logit_model.params)
print(logit_model.bse)

# 로지스틱회귀 분석 모델 그래프 그림
# z = np.arange(-5, 5, 0.1)
# phi = 1 / (1 + np.exp(-z))

# plt.plot(z, phi)
# plt.xlabel('z')
# plt.ylabel('phi')
# plt.show()


