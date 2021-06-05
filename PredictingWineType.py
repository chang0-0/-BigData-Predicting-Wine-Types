import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols, glm
from scipy import linalg
from scipy import stats
from sklearn.linear_model import LinearRegression

wine = pd.read_csv('wine.csv', sep=',', header=0)
wine.columns = wine.columns.str.replace(' ','_')

#'type' 이진 값 예측 분류 0 OR 1 
# 0 = red_wine, 1 = white_wine (값 이진 분류 로지스틱 회귀 모델 사용) 
wine['type'] = np.where(wine['class'] == 0, 'red', 'white')
print("================================================= wine head 위에서 5줄까지 출력 =================================================")
print(wine.head())

print("================================================= wine info wine.csv 데이터 타입 표현 =================================================")
print(wine.info())

# unstack 함수를 추가하여 그 결과를 가로 방향으로 재구조화한다.
print("================================================= wine describe =================================================")
print(wine.describe())


# groupby함수는 type열의 두 값, 즉 레드와 화이트를 기준으로 데이터셋을 그룹화한다
# 이때 대괄호를 사용한다. 
# describe함수는 quality 열의 요약통계를 두 그룹으로 구분하여 세로방향으로 출력한다.
# unstack함수를 추가하여 결과를 가로 방향으로 재구조화 한다. 
print("================================================= wine groupby =================================================")
print(wine.groupby('type')[['class']].describe().unstack('type'))

#와인 종류에 따른 품질의 분포 확인하기
red_wine = wine.loc[wine['type']=='red', 'alcohol']
white_wine = wine.loc[wine['type']=='white', 'alcohol']

#와인 종류에 따라 당도, 산성도, 도수의 차이 검정
print("=================================================와인 종류에 따른 당도, 산성도, 도수의 차이 검정=================================================")
print(wine.groupby(['type'])[['sugar','pH', 'alcohol']].agg(['std', 'mean']))
tstat, pvalue, df = sm.stats.ttest_ind(red_wine, white_wine)
print('tstat: %.3f pvalue: %.4f' % (tstat, pvalue))

print("================================================wine corr=====================================================")
#와인 테이블 상관관계 분석  (모든 변수 쌍 사이의 상관계수 구함)
print(wine.corr())

#groupby 및 agg함수를 사용하여 그룹별 품질의 평균과 표준편차를 구했다.

#상관 관계 분석중 유의미 한 부분 vars 변수로 선택
g = sns.pairplot(wine, kind='reg', plot_kws={"ci": False, "x_jitter": 0.25, "y_jitter":0.25}, \
	hue='type', diag_kind='hist', diag_kws={"bins":10, "alpha":1.0}, palette=dict(red="red", white="white"), \
		markers=["o", "s"], vars=['class', 'alcohol', 'sugar', 'pH'])
print(g)
plt.suptitle('Histograms and Scatter Plots of Type, Alcohol, and Sugar', fontsize=14, horizontalalignment='center', verticalalignment='top', x=0.5, y=0.999)
plt.show()

#그룹별 기술통게 구하기
print("================================================그룹별 기술 통계 구하기=====================================================")
print(wine.groupby(['type'])[['sugar','pH', 'alcohol']].agg(['count', 'mean', 'std']))

# 로지스틱 회귀 분석에서는 회귀식 대신 독립변수와 종속변수를 따로 할당한다.
# 2번 계수와 절편이 포함된 통계표 
print("===============================================계수와 절편이 포함된 통계표<logit model>=============================================== ")
dependent_variable = wine['class']
independent_variables = wine[['sugar', 'pH', 'alcohol']]
independent_variables_with_constant = sm.add_constant(independent_variables, prepend=True)
logit_model = sm.Logit(dependent_variable, independent_variables_with_constant).fit()
#logit_model = smf.glm(output_variable, input_variables, family=sm.families.Binomial()).fit()
print(logit_model.summary())


# 계수와 절편값을 통한 선형함수식.



# 와인 데이터셋의 quality를 종속변수로 생성
dependent_variable = wine['class']
independent_variable = wine[wine.columns.difference(['class'])]


#계수 (기울기): coef / 절편 intercept == const



#4. 새로운 테스트 값 입력 wine type 예측
print("======================================= 값 예측하기 =============================================")
new_observations = wine.loc[wine.index.isin(range(10)), independent_variables.columns]
new_observations_with_constant = sm.add_constant(new_observations, prepend=True)
y_predicted = logit_model.predict(new_observations_with_constant)
y_predicted_rounded = [round(score, 2) for score in y_predicted]
print(y_predicted_rounded)
print(y_predicted_rounded.coef)
# [0.23, 0.64, 0.55, 0.62, 0.23, 0.22, 0.4, 0.34, 0.38, 0.81]


'''
계수와 절편 -> 선형함수식 -> 선형회귀모형 -> 선형회귀분석
z > 0 z의 값이 양수(양성) 


wine.csv 파일에는 네 개의 특성(속성)이 있다. alcohol, sugar, pH, class가 있고, 6497개의 데이터가 있다. class의 값이 0이면 레드 와인, 1이면 화이트 와인이다.
alcohol, sugar, pH의 입력값에 따라 레드 와인인지 화이트 와인인지 예측(이진 분류)하려고 한다.
다음 요구사항을 만족하도록 프로그래밍 하시오. 라이브러리 사용은 제한이 없고 특성값(입력값)의 표준화가 필요하다. 로지스틱 회귀분석을 사용한다.

1) head(), info(), describe()의 결과를 print 하시오.
2) 계수와 절편 등이 포함된 통계표를 print 하시오.
3) 출력된 계수와 절편 값을 이용하여 선형함수식을 만들어 print 하시오.
4) 새로운 테스트 값을 입력해서 레드 와인인지 화이트 와인인지 예측하고 print하시오.
5) 예측시에 확률(로지스틱 함수의 출력값)을 print 하시오.

'''