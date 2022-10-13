import librosa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
import random

from sklearn.preprocessing import PolynomialFeatures

train_Y = []
X , sr = librosa.load('./dataSet/wav/audio1.wav')

# convert X to 2d array
train_X = X.reshape(1, X.size)

for num in range(2,266):
  filename = "./dataSet/wav/audio" + str(num) + ".wav"
  data, sr = librosa.load(filename)
  train_X = np.append(train_X, data.reshape(1, data.size), axis=0)

result = pd.read_csv('./dataSet/result/SubjectiveResultsLiterature.csv', usecols=['y'])
train_Y = result.values.astype(float)
# print(X_2d)
# print(Y)

# LinearRegression
# create model
clf = linear_model.LinearRegression()
clf.fit(train_X, train_Y)
# error
print(clf.intercept_)

# testing
randomNum = random.randrange(265)
# predicted value
predictedValue = clf.predict([train_X[randomNum]])
print("testing value with audio", randomNum)
print("predicted value = ", predictedValue)
# expected value
expectedValue = train_Y[randomNum]
print("expected value = ", expectedValue)


# poly = PolynomialFeatures(6)
# train_poly_X = poly.fit_transform(train_X)

# model = linear_model.Ridge(alpha=1.0)
# model.fit(train_poly_X, train_Y)
# print("predicted value = ", model.predict([train_poly_X[randomNum]]))
# print("expected value = ", train_Y[randomNum])

# wine = pd.read_csv("./dataSet/result/winequality-red.csv", sep=";")
# wine.head
# wine_except_quality = wine.drop("quality", axis=1)
# H = wine_except_quality.values
# Y = wine['quality'].values
# # 予測モデルを作成
# clf.fit(H, Y)
# # 偏回帰係数
# print(pd.DataFrame({"Name":wine_except_quality.columns,
#                     "Coefficients":clf.coef_}).sort_values(by='Coefficients') )
# # 切片 (誤差)
# print(clf.intercept_)