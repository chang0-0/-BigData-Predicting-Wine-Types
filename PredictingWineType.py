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
wine['type'] = np.where(wine['class'] == 0., 'red', 'white')
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
print("=================================================와인 종류에 따른 당도의 차이 검정=================================================")
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
plt.suptitle("dsf")
plt.show()








my_formula = 'class ~ alcohol + pH + sugar'
lm = ols(my_formula, data=wine).fit()
print(lm.summary())


#계수 (기울기): coef / 절편 intercept


'''
wine.csv 파일에는 네 개의 특성(속성)이 있다. alcohol, sugar, pH, class가 있고, 6497개의 데이터가 있다. class의 값이 0이면 레드 와인, 1이면 화이트 와인이다.
alcohol, sugar, pH의 입력값에 따라 레드 와인인지 화이트 와인인지 예측(이진 분류)하려고 한다.
다음 요구사항을 만족하도록 프로그래밍 하시오. 라이브러리 사용은 제한이 없고 특성값(입력값)의 표준화가 필요하다. 로지스틱 회귀분석을 사용한다.

1) head(), info(), describe()의 결과를 print 하시오.
2) 계수와 절편 등이 포함된 통계표를 print 하시오.
3) 출력된 계수와 절편 값을 이용하여 선형함수식을 만들어 print 하시오.
4) 새로운 테스트 값을 입력해서 레드 와인인지 화이트 와인인지 예측하고 print하시오.
5) 예측시에 확률(로지스틱 함수의 출력값)을 print 하시오.'''