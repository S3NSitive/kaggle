"""
갓 세차를 구매한 운전자의 들뜬 마음에 찬물을 끼얹는 것은 다름 아닌 보험 청구서 내역이다.
당신이 안전한 운전자일수록, 보험 청구서로 인한 불편함은 더 크게 다가올 뿐이다. 도로에서 몇 년 동안 안전운전을 해온 당신이 그토록
많은 금액을 지불해야 한다는 것은 공평해보이지 않는다.
브라질에서 가장 큰 자동차 및 주택 보험회사인 포르토 세구로는 전적으로 동의한다. 자동차 보험 회사의 부정확한 보험 청구 예측 ㅁ델은
좋은 운전자에게 과다한 금액을 청구하고 나쁜 운전자에게 약소한 금액을 청구한다.
이번 경진대회에서, 여러분은 운전자가 내년에 자동차 보험 청구를 진행할 확률을 예측하는 모델을 개발하게 된다.
포르토 세구로 사는 지난 20년 난 기계 학습을 꾸준히 사용해 왔지만, 캐글 머신러닝 커뮤니티에서 새롭고, 더욱 강력한
기법이 발견되길 기대하고 있다. 보다 정확한 예측 모델은 운전자에게 합리적인 가격을 제공하고, 더 많은 운전자들이 자동차 보험의 혜택을
받을 수 있게 도와줄 것이다.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

print(train.shape)
print(train.columns)
print(train.describe())
