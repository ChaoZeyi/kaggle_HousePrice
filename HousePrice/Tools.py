# -*- coding: utf-8 -*-
"""
    @ author cherry
    @ data 2017/11/26 16:22
"""
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import LabelEncoder
def readTrainData():
    train_data = pd.read_csv("../input/train.csv", sep=",")
    return train_data

def readTestData():
    test_data = pd.read_csv("../input/test.csv", sep=",")
    return test_data

def valuation(prediction, label):
    result = np.sqrt(mean_squared_error(prediction, label))
    print('RMSE误差是：{}'.format(result))

def predictionPerformance(model, train_features, train_lebels, validation_features, validation_labels):
    train_prediction = model.predict(train_features)
    valuation(train_prediction, train_lebels)

    validation_prediction = model.predict(validation_features)
    valuation(validation_prediction, validation_labels)

def category2num1(series):
    series.replace('Ex', '0',inplace=True)
    series.replace('Gd', '1',inplace=True)
    series.replace('TA', '2',inplace=True)
    series.replace('Fa', '3',inplace=True)
    series.replace('Po', '4',inplace=True)
    return series
def category2num2(series):
    series.replace('Ex', '0',inplace=True)
    series.replace('Gd', '1',inplace=True)
    series.replace('TA', '2',inplace=True)
    series.replace('Fa', '3',inplace=True)
    series.replace('Po', '4',inplace=True)
    series.replace('None', '5',inplace=True)
    return series
def category2num3(series):
    series.replace('Gd', '0',inplace=True)
    series.replace('Av', '1',inplace=True)
    series.replace('Mn', '2',inplace=True)
    series.replace('No', '3',inplace=True)
    series.replace('None', '4',inplace=True)
    return series
def category2num4(series):
    series.replace('GLQ', '0',inplace=True)
    series.replace('ALQ', '1',inplace=True)
    series.replace('BLQ', '2',inplace=True)
    series.replace('Rec', '3',inplace=True)
    series.replace('Lwq', '4',inplace=True)
    series.replace('Unf', '5',inplace=True)
    series.replace('None', '6',inplace=True)
    return series
def category2num5(series):
    series.replace('Fin', '0',inplace=True)
    series.replace('RFn', '1',inplace=True)
    series.replace('Unf', '2',inplace=True)
    series.replace('None', '3',inplace=True)
    return series
def numericStandard(series):
    series = (series - series.mean())/series.std()
    return series
s = pd.DataFrame([['1','3','4'],['a','b','c']])
s = pd.get_dummies(s)
print(s)