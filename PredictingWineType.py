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


#헷갈리지 말기 => 계수 (기울기): coef / 절편 intercept == const

wine = pd.read_csv('wine.csv', sep=',', header=0)

wine.columns = wine.columns.str.replace(' ','_').str.replace("\'", "")

#'type' 이진 값 예측 분류 0 OR 1 
# 0 = red_wine, 1 = white_wine (값 이진 분류 로지스틱 회귀 모델 사용) 
wine['type'] = np.where(wine['class'] == 0.0, 'red', 'white')
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
print(wine.groupby(['type'])[['class','alcohol','sugar', 'pH']].agg(['std', 'mean']))
tstat, pvalue, df = sm.stats.ttest_ind(red_wine, white_wine)
print('tstat: %.3f pvalue: %.4f' % (tstat, pvalue))

print("================================================wine corr=====================================================")
#와인 테이블 상관관계 분석  (모든 변수 쌍 사이의 상관계수 구함)
print(wine.corr())

#groupby 및 agg함수를 사용하여 그룹별 품질의 평균과 표준편차를 구했다.

#상관 관계 분석중 유의미 한 부분 vars 변수로 선택
#  g = sns.pairplot(wine, kind='reg', plot_kws={"ci": False, "x_jitter": 0.25, "y_jitter":0.25}, \
# 	hue='type', diag_kind='hist', diag_kws={"bins":10, "alpha":1.0}, palette=dict(red="red", white="white"), \
# 		markers=["o", "s"], vars=['class', 'alcohol', 'sugar', 'pH'])
# print(g)
# plt.suptitle('Histograms and Scatter Plots of Type, Alcohol, and Sugar', fontsize=14, horizontalalignment='center', verticalalignment='top', x=0.5, y=0.999)
# plt.show()

#그룹별 기술통게 구하기
print("================================================그룹별 기술 통계 구하기=====================================================")
print(wine.groupby(['type'])[['class','alcohol','sugar','pH']].agg(['count', 'mean', 'std']))

# 로지스틱 회귀 분석에서는 회귀식 대신 독립변수와 종속변수를 따로 할당한다.
# 2번 계수와 절편이 포함된 통계표 

print("===============================================계수와 절편이 포함된 통계표(표준화 작업 후, 상수항 추가)<logit model>=============================================== ")
# dependent_variable = wine['class']
# independent_variables = wine[['alcohol', 'sugar', 'pH']]
# independent_variables_standardized = (independent_variables - independent_variables.mean()) / independent_variables.std()
# independent_variables_standardized = sm.add_constant(independent_variables_standardized)
# print(independent_variables_standardized)
# logit_model = sm.Logit(dependent_variable, independent_variables_standardized).fit()

# 학습 전에 일단 표준화 부터 실행 상수행 기존 행 삭제. add_cosntant
# 'class, type'를 제외한 모든 column들을 feature로 삼으려면,
columns_selected = "+".join(wine.columns.difference(["class", "type"]))
my_formula = "class ~ " + columns_selected
from statsmodels.formula.api import ols, glm, logit

dependent_variable = wine['class']
independent_variables = wine[['alcohol', 'sugar', 'pH']]
print("출력 값 테스트")
print(wine.info())
print(independent_variables)

independent_variables_standardized = (independent_variables - independent_variables.mean()) / independent_variables.std()
independent_variables_with_constant = sm.add_constant(independent_variables_standardized, prepend=True)
wine_standardized = pd.concat([dependent_variable, independent_variables_with_constant], axis=1)

print(wine_standardized)


logit_model = sm.Logit(dependent_variable, independent_variables_with_constant).fit()


# 기존의 데이터값에서 표준화는 하지 않고 상수항만 추가한 상태에서 odds 값을추가함.
independent_variables_with_constant2 = sm.add_constant(independent_variables, prepend=True)
logit_model2 = sm.Logit(dependent_variable, independent_variables_with_constant2).fit()


print(logit_model.summary())
print("기존의 데이터에서 상수항만 추가하고 표준화는 진행하지 않은 로지스트모델 값.")
print(logit_model2.summary())


# 상수항을 추가하지않은 표준화 데이터를 만들어보자.
# independent_variables_standardized2 = (independent_variables - independent_variables.mean()) / independent_variables.std()
# wine_standardized2 = pd.concat([dependent_variable, independent_variables_standardized2], axis=1)

#결론은 표준화된 독립변수들 사이에서 전체 앞의 종속변수 와 axis=1 하나의 행으로 합친다는 의미를 가짐.

print("axis 테스트 \n")
print("종속 변수 출력\n" ,dependent_variable)
print("표준화 되기 전 독립변수 출력 \n", independent_variables)
print("표준화 된 독립변수 출력\n", independent_variables_standardized)
print("전체 표준화 axis 값 출력\n", wine_standardized)
print(independent_variables_with_constant)


# ⭐⭐⭐⭐ 상위 결과 값 logit(로지스트 회귀분석에서)cosnt 값이 출력되는데 Intercept 의 값을 출력시켜보자.⭐⭐⭐⭐
# 참고로 intercept 가 나와도 const 계수값은 똑같이 나온다는 점을 다른 고객이탈 데이터 결과값으로 알수 있음.
# 밑에 줄 안되니까 일단 넘어가는걸로 ols 도대체 왜 안먹히는 거임?
# logit_model2 = ols(formula=my_formula, data=wine_standardized2).fit()
# print(logit_model2.summary())

print("\nCoefficients:\n%s" % logit_model.params)
print("\nCoefficient Std Errors:\n%s" % logit_model.bse)

# 기존 customer방식으로 표준화를 진행할 예정 아무리 봐도 맞는거같음 (단, 상수항을 추가하지않았음.)
# 여기서 만약 표준화를 진행하지 않고 odds를 실행할경우 평균은 0.86 분산과 표준편차는 0.05이 나오게 된다.
# 표준화를 진행하고 odds를 진행하면 평균은 0, 분산과 표준편차는 1이 된다는것을 볼수 있는듯함.


# print("===============================================계수와 절편이 포함된 통계표(표준화 후)<logit model>=============================================== ")
# my_formula = 'class~alcohol + sugar + pH'
# from statsmodels.formula.api import ols, glm, logit
# from sklearn.preprocessing import StandardScaler
# standardScaler = StandardScaler()


# #표준화 작업
# # dependent_variable = wine['class']
# # independent_variables = wine[wine.columns.difference(['alcohol','sugar','pH'])]
# # independent_variables_standardized = (independent_variables - independent_variables.mean()) / independent_variables.std()
# # wine_standardized = pd.concat([dependent_variable, independent_variables_standardized], axis=1)
# # logit_model = sm.Logit(my_formula, data=wine_standardized).fit()

# # print(wine_standardized.describe())
# # print(logit_model.summary())

# output_variable = wine['class']
# vars_to_keep = wine[['alcohol', 'sugar', 'pH']]
# inputs_standardized = (vars_to_keep - vars_to_keep.mean()) / vars_to_keep.std()
# input_variables = sm.add_constant(inputs_standardized, prepend=True)
# logit_model = sm.Logit(output_variable, input_variables).fit()

# print(logit_model.summary())

# # 계수 출력 함수
# print("\nCoefficients:\n%s" % logit_model.params)
# # 계수 오류 출력
# print("\nCoefficient Std Errors:\n%s" % logit_model.bse)


# 3번 출력된 계수와 절편을 이용해 선형함수식을 print 하시오.
print("=============================================== 선형 함수식 ===============================================")
print("\n (Logit)y=(1.8023 + 0.5350*alcohol + 1.6808*sugar + (-0.7101)*pH")
print("\n (OLS) y=(0.7539 + 0.0816*alcohol +  0.1501*sugar + (-0.1115)*pH")


# 4번. 새로운 테스트 값을 입력해서 레드 와인인지 화이트 와인인지 예측하고 print 하시오.
print("==========================================odds 적용=========================================")
# 회귀계수 해석
# 회귀계수가 종속변수의 성공 확률에 미치는 영향을 파악하기 위해서는 모형 평가의 기준, 
# odds가 필요하다. 선형회귀모형의 절편과 마찬가지로 기준 오즈는 모든 독립변수가 0인 경우에
# 성공 확률에 미치는 영향을 나타낸다. 절편과 마찬가지로 기준 
# odds값은 그 자체로서는 그다지 의미가 없으므로 먼저 모든 독립변수를 평균으로 설정하여
# 모형을 평가해야 한다.

def inverse_logit(model_formula):
	from math import exp
	return (1.0 / (1.0 + exp(-model_formula)))*100.0

at_means = float(logit_model2.params[0]) + \
	float(logit_model2.params[1]) * float(wine['alcohol'].mean()) + \
	float(logit_model2.params[2]) * float(wine['sugar'].mean()) + \
	float(logit_model2.params[3]) * float(wine['pH'].mean())

print(wine['alcohol'].mean())
print(wine['sugar'].mean())
print(wine['pH'].mean())
print(at_means)
print("Probability of wine when independent variables are at their mean values: %.2f" % inverse_logit(at_means))

alcohol_mean = float(logit_model2.params[0]) + \
	float(logit_model2.params[1])*float(wine['alcohol'].mean()) + \
	float(logit_model2.params[2])*float(wine['sugar'].mean()) + \
	float(logit_model2.params[3])*float(wine['pH'].mean())
		
alcohol_mean_minus_one = float(logit_model2.params[0]) + \
		float(logit_model2.params[1])*float(wine['alcohol'].mean()-1.0) + \
		float(logit_model2.params[2])*float(wine['sugar'].mean()) + \
		float(logit_model2.params[3])*float(wine['pH'].mean())

print(alcohol_mean)
print(wine['alcohol'].mean()-1.0)
print(alcohol_mean_minus_one)
print("Probability of wine when sugar changes by 1: %.2f" % (inverse_logit(alcohol_mean) - inverse_logit(alcohol_mean_minus_one)))
# 알코올이 한단위 증가할경우 레드와인이 될 확률

# 와인 데이터셋의 quality를 종속변수로 생성
#4번 .새로운 테스트 값 입력 wine type 예측

print("=======================================기존 데이터 셋의 첫 10개 값을 가지고 새로운 관측값 데이터 셋을 만듦 =============================================")
new_observations = wine.loc[wine.index.isin(range(10)), independent_variables.columns]
new_observations_with_constant = sm.add_constant(new_observations, prepend=True)
y_predicted = logit_model2.predict(new_observations_with_constant)
y_predicted_rounded = [round(score, 2) for score in y_predicted]
print(y_predicted_rounded)


# 예측시에 확률 로지스틱 함수와 출력값 print

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


의문점 1. 과연 표준화를 진행해야 하는가?
//로지스틱 함수도 똑같이 범위가 클 경우 표준화를 진행해야함.
추가 의문점 그렇다면 customer_churn 파일도 표준화가 되어있는가? 
표준화 안되어있다고 함.( 근데 모르겠음)

의문점 2. 로지스틱 함수의 식과 선형함수의 식은 어떻게 다른가
로지스틱 함수의 역함수는 시그모이드 함수가 되는데 시그모이드 함수는 출력의 결과값이 0과 1로 변화시켜줌.
로지스틱함수에서는 필수적인 요소.

의문점 3. 지금나오는 예측값이 정상적인 값인가? (제대로 나오는게 아닌 것 같음)
아닌듯 

의문점 4. 식은 내가 직접 작성해야되는가?
 맞음


의문점 5. 예측시에 확률 (로지스틱 함수의 출력값은 무엇을 써야하는가?)
(고객 이탈 자료의 11.23)

의문점 6. customer_churn.py 파일에서는 왜 평균값을 사용했는가? (차이점 구별하기)
(odds의 활용값 때문에 평균값 활용으로 해당 독립변수가 1만큼 증가했을때 전체 예측 확률의 증가 정도를 파악할 수 있음)

의문점 7. 고객이탈 파일에서 표준화를 진행한다면 과연 값이 얼마나 변하는 가?


의문점 8. 상수항 추가한 표준화 계수와 상수항 추가하지않은 표준화 계수가 값이 다르지 않음 무슨차이?
(상수항을 추가하고 안하고 차이는 없는 것 같지만 odds의 출력값에서 확실한 변화가 생기는 것 같음)
prepend = True는 열 생성 가장앖에 추가하는 것이고 axis는 여러개의 열을 하나로 합하는 것

표준화 진행했음 근데 제대로 된건지 모르겠음
그리고 제대로 됬는지 확인이 안되는 이유
분명 수업시간에 custserv_calls 부분이 가장 높은 계수를 가진다고 했는데
표준화를 똑같이 진행하고 나니까 total_charge 부분이 계수가 가장 높아짐.



의문점 8. 과연 표준화를 진행했을 경우 값이 더 커지는 경우가 있는가?
가령, 고객이탈 자료에서 분명 고객센터에 전화한 횟수가 증가할경우 고객이 이탈할 경우가 높아지는데
그 이유를 확인할수 있는 첫번재 이유는 분명 계수가 가장높았음
그런데 표준화를 진행한후에 계수가 totalcharge가 더 커졌음 why?
( account_length의  최솟값과 최댓값의 차이가 매우 컸음)
min = 1.0 max = 243.0 이었는데 표준화 작업 이후
min = -2, max = 3.0
결론. 표준화 작업이 제대로 된게 맞는가?????

해결점 => churn 의 데이터를 표준화 작업을 실행했을 때 count 단위가 작아졌다
당연히 오류 일줄 알았는데 wine 표준화 작업을 실행했을때도 count가 작아지는걸 보니
표준화 작업을 실행했을 때 count도 영향이 있는게 확실함

고객이탈 파일에서 마지막줄에 있는 표준화 코드가 표준화가 아니라고 했는데 무슨 차이점인지 드디어 발견함.
기존에 상위에 있던 상수항을 추가해서 결과값을 뽑은 로지스트 회귀 파일은 
Logist 값에 Intercept 계수의 결과값이 출력되서 나온 반면,





'''