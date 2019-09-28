import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import MinMaxScaler

#skiprow 건너 띄는 것
df = pd.read_csv('./data/cansim-0800020-eng-6674700030567901031.csv',
                 skiprows = 6, skipfooter=9, engine='python'
                 )
print(df.head())
df['Adjustments'] = pd.to_datetime(df['Adjustments']) + MonthEnd(1)
df = df.set_index('Adjustments')
print(df.head())
plt.plot(df)
#plt.show()

split_date = pd.Timestamp('01-01-2011')
#처음부터 split_date까지
train = df.loc[:split_date, ['Unadjusted']]
#split_date부터 끝까지
test = df.loc[split_date:, ['Unadjusted']]
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])
plt.plot()
#plt.show()
sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)
train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)

#print(train_sc_df.head())
#0에서1의 범위를 정한 후(최고값, 최소값을 0부터 1사이의 갑승로)
# pandas shift를 통해 window 만들기
#shift는 이전 정보를 다음 row에서 다시 쓰기 위한 pandas 함수
# 과거값을 총 12개로 저장하며, timestamp는 12개가 된다.
# 이 작업의 이유는 과거 값 shift ~ 12를 통해 현재값 Scaled를 예측하는 것


for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['test_{}'.format(s)] = test_sc_df['Scaled'].shift(s)
#print(train_sc_df.head(13))
X_train = train_sc_df.dropna().drop('Scaled', axis=1)
y_train = train_sc_df.dropna()[['Scaled']]
X_test = train_sc_df.dropna().drop('Scaled', axis=1)
y_test = train_sc_df.dropna()[['Scaled']]
#최종 트레이닝 셋
print(X_train.head())
print(y_train.head())
#ndarray로 변환
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
print(X_train)
print(X_test)
print(y_train)
print(y_test)
print('#############')
X_train_t = X_train.reshape(X_train.shape[0], 12, 1)
X_test_t = X_train.reshape(X_test.shape[0], 12, 1)
print('최종 data')
print(X_train.shape)
print(X_train)
print(y_train.shape)
"""
시간에 따른 텐서(매트릭스)가 움직이면서 매 시간마다 가장 높은 확률을 찍는다
[[0.20091289 0.13173822 0.11139526 ... 0.0704258  0.         0.01402033]
 [0.03002688 0.20091289 0.13173822 ... 0.09531795 0.0704258  0.        ]
 [0.01999285 0.03002688 0.20091289 ... 0.16362761 0.09531795 0.0704258 ]
 ...
 [0.79916654 0.81439355 0.86398323 ... 0.92972161 0.71629034 0.77368724]
 [0.80210057 0.79916654 0.81439355 ... 0.59734863 0.92972161 0.71629034]
 [0.81482896 0.80210057 0.79916654 ... 0.53166512 0.59734863 0.92972
 이 값이 텐서
 매트릭스 구조

"""






