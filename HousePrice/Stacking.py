# -*- coding: utf-8 -*-
"""
    @author cherry
    @date 2017/11/28 17:15
"""
"""
分析数据特点，初步选择可能的特征
"""
from Tools import readTrainData, readTestData, category2num1, \
    category2num2, category2num3, category2num4, category2num5, numericStandard, valuation
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import xgboost as xgb
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
# 读取训练数据和测试数据
train_data = readTrainData()
test_data = readTestData()
# 训练目标值：价格，观察是否有为0的值
train_labels = np.log(train_data['SalePrice'])
train_data.drop('SalePrice', axis=1, inplace=True)
print(train_labels.describe())
# 价格的分布直方图
# train_labels.hist()
# plt.show()

# 观察各列为na的情况
train_isnull = train_data.isnull().sum()
# print(type(train_isnull))    Series
print(train_isnull[train_isnull > 0])
test_isnull = test_data.isnull().sum()
# print(type(train_isnull))    Series
print(test_isnull[test_isnull > 0])
# 确定去除的列：Alley、PoolQC、Fence、MiscFeature
features_toBeAbandoned = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
train_data.drop(features_toBeAbandoned, axis=1, inplace=True)
test_data.drop(features_toBeAbandoned, axis=1, inplace=True)
"""
##################################对具有缺失值的列进行填充###############################################################
"""
# 用所有相同邻居的住宅的距离中位数来填充
train_data['LotFrontage'] = train_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
test_data['LotFrontage'] = test_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
# Electrical 只有一个NA值，用出现次数最多的值来代替
train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])
"""
##############################################不能简单的用None和0来填充，没有实际意义#####################################
"""
"""train_whetherNull = train_data.isnull().sum()
train_isnull = train_whetherNull[train_whetherNull > 0]
# print(type(train_isnull))    Series
for idx in train_isnull.index:
    # print(idx)
    # print(type(train_data[idx]))
    # print(type(train_data[idx][0]))
    if(isinstance(train_data[idx][1], str)):
        train_data[idx] = train_data[idx].fillna('None')
    else:
        train_data[idx] = train_data[idx].fillna(0)

test_whetherNull = test_data.isnull().sum()
test_isnull = test_whetherNull[test_whetherNull > 0]
# print(type(train_isnull))    Series
for idx in test_isnull.index:
    if(isinstance(train_data[idx][2], str)):
        test_data[idx] = test_data[idx].fillna('None')
    else:
        test_data[idx] = test_data[idx].fillna(0)
train_isnull2 = train_data.isnull().sum()
# print(type(train_isnull))    Series
print(train_isnull2[train_isnull2 > 0])
test_isnull2 = test_data.isnull().sum()
# print(type(train_isnull))    Series
print(test_isnull2[test_isnull2 > 0])
print("no missing columns.....")
"""

# 地板类型和面积，这两者是一致的
train_data['MasVnrType'] = train_data['MasVnrType'].fillna('None')
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(0)
test_data['MasVnrType'] = test_data['MasVnrType'].fillna('None')
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(0)
# NA用None填充，表示没有地下室
for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    train_data[col] = train_data[col].fillna('None')
    test_data[col] = test_data[col].fillna('None')
for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    test_data[col] = test_data[col].fillna(0)
# NA用None填充，表示没有车库
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train_data[col] = train_data[col].fillna('None')
    test_data[col] = test_data[col].fillna('None')
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(0)
test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(0)
# GarageCars、GarageArea
for col in ['GarageCars','GarageArea']:
    test_data[col] = test_data[col].fillna(0)
# FireplaceQu NA值代表没有fireplcae
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna('None')
test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna('None')
# MSZoning、utilities、KitchenQual、Functional、SaleType
for col in ['MSZoning','Utilities','KitchenQual','Functional','SaleType']:
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
# Exterior1st、Exterior2nd
test_data['Exterior1st'] = test_data['Exterior1st'].fillna(test_data['Exterior1st'].mode()[0])
test_data['Exterior2nd'] = test_data['Exterior2nd'].fillna(test_data['Exterior1st'].mode()[0])


train_isnull2 = train_data.isnull().sum()
# print(type(train_isnull))    Series
print(train_isnull2[train_isnull2 > 0])
test_isnull2 = test_data.isnull().sum()
# print(type(train_isnull))    Series
print(test_isnull2[test_isnull2 > 0])

print("no missing data")
"""
##########################################选择特征######################################################################
"""
print("start to select features......")
# 特征与价格的相关性
#
# plt.scatter(train_data['GrLivArea'], train_data['SalePrice'])
# plt.show()
# corr = train_data.corr()
# sn.heatmap(corr)
# plt.show()
# 将训练数据和测试数据进行拼接，做相同的类型转换和编码处理
print("train_data size is : {}".format(train_data.shape))
print("test_data size is : {}".format(test_data.shape))
ntrain = train_data.shape[0]
ntest = test_data.shape[0]
all_data = pd.concat((train_data, test_data)).reset_index(drop=True)
print("all_data size is : {}".format(all_data.shape))
# 把表示类别的数值型数据转换为类别类型
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
# test_data['MSSubClass'] = test_data['MSSubClass'].astype(str)
# ExterQual ExterCond
all_data['ExterQual'] = category2num1(all_data['ExterQual'])
all_data['ExterCond'] = category2num1(all_data['ExterCond'])
all_data['BsmtQual'] = category2num2(all_data['BsmtQual'])
all_data['BsmtCond'] = category2num2(all_data['BsmtCond'])
all_data['BsmtExposure'] = category2num3(all_data['BsmtExposure'])
all_data['BsmtFinType1'] = category2num4(all_data['BsmtFinType1'])
all_data['BsmtFinType2'] = category2num4(all_data['BsmtFinType2'])
all_data['HeatingQC'] = category2num1(all_data['HeatingQC'])
all_data['KitchenQual'] = category2num4(all_data['KitchenQual'])
all_data['FireplaceQu'] = category2num2(all_data['FireplaceQu'])
all_data['GarageFinish'] = category2num5(all_data['GarageFinish'])
all_data['GarageQual'] = category2num2(all_data['GarageQual'])
all_data['GarageCond'] = category2num2(all_data['GarageCond'])

# test_data['ExterQual'] = category2num1(test_data['ExterQual'])
# test_data['ExterCond'] = category2num1(test_data['ExterCond'])
# test_data['BsmtQual'] = category2num2(test_data['BsmtQual'])
# test_data['BsmtCond'] = category2num2(test_data['BsmtCond'])
# test_data['BsmtExposure'] = category2num3(test_data['BsmtExposure'])
# test_data['BsmtFinType1'] = category2num4(test_data['BsmtFinType1'])
# test_data['BsmtFinType2'] = category2num4(test_data['BsmtFinType2'])
# test_data['HeatingQC'] = category2num1(test_data['HeatingQC'])
# test_data['KitchenQual'] = category2num4(test_data['KitchenQual'])
# test_data['FireplaceQu'] = category2num2(test_data['FireplaceQu'])
# test_data['GarageFinish'] = category2num5(test_data['GarageFinish'])
# test_data['GarageQual'] = category2num2(test_data['GarageQual'])
# test_data['GarageCond'] = category2num2(test_data['GarageCond'])
# 对于不同的类别采用不同的编码方式，存在明显大小关系的采用LabelEncoder 其他的采用one-hot编码
# LabelEncoder:OverallQual,OverallCond,YearBuilt,YearRemodAdd, ExterQual ExterCond BsmtQual BsmtCond BsmtExposure
# BsmtFinType1 BsmtFinType2,HeatingQC,CentralAir,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,Bedroom,Kitchen,KitchenQual
# TotRmsAbvGrd Fireplaces,FireplaceQu,GarageYrBlt,GarageFinish,GarageCars,MiscVal,MoSold,YrSold
for col in ['OverallQual','OverallCond','YearBuilt','YearRemodAdd', 'ExterQual','ExterCond', 'BsmtQual', 'BsmtCond',
            'BsmtExposure','BsmtFinType1', 'BsmtFinType2','HeatingQC','CentralAir','BsmtFullBath','BsmtHalfBath',
            'FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd' ,'Fireplaces','FireplaceQu',
            'GarageYrBlt','GarageFinish','GarageCars','MiscVal','MoSold','YrSold']:
    le = LabelEncoder()
    le.fit(all_data[col])
    all_data[col] = le.transform(all_data[col])
# one-hot编码:'MSSubClass', 'MSZoning','Street','Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,
# Condition1,Condition2,BldgType,HouseStyle,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,Foundation,Heating
# Electrical,Functional,GarageType,PavedDrive,SaleType,SaleCondition
for col in ['MSSubClass', 'MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
            'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
            'Exterior2nd','MasVnrType','Foundation','Heating','Electrical','Functional','GarageType','PavedDrive',
            'SaleType','SaleCondition']:
    all_data[col] = pd.get_dummies(all_data[col])
# 数值型，进行归一化：'LotFrontage','LotArea',MasVnrArea,BsmtFinSF1.,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,1stFlrSF,2ndFlrSF,
# LowQualFinSF,GrLivArea,GarageArea,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea
for col in ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF',
            '2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
            'ScreenPorch','PoolArea']:
    all_data[col] = numericStandard(all_data[col])

"""
#########################################################建立模型#######################################################
"""
x_train = all_data[:ntrain]
x_test = all_data[ntrain:]
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse= cross_val_score(model, x_train.values, train_labels.values, cv = kf.get_n_splits())
    return(rmse)
# 尝试多个模型
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                              nthread = -1)
score_xgb = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score_xgb.mean(), score_xgb.std()))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score_lgb = rmsle_cv(model_lgb)
print("Lightgbm score: {:.4f} ({:.4f})\n".format(score_lgb.mean(), score_lgb.std()))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score_krr = rmsle_cv(KRR)
print("KRR score: {:.4f} ({:.4f})\n".format(score_krr.mean(), score_krr.std()))

model_xgb.fit(x_train.values, train_labels)
model_xgb_prec = model_xgb.predict(x_train.values)
print(valuation(model_xgb_prec, train_labels))

model_lgb.fit(x_train.values, train_labels)
model_lgb_prec = model_lgb.predict(x_train.values)
print(valuation(model_lgb_prec, train_labels))

KRR.fit(x_train.values, train_labels)
KRR_prec = KRR.predict(x_train.values)
print(valuation(KRR_prec, train_labels))
"""
###############################################模型融合#################################################################
"""
model_xgb_res = model_xgb.predict(x_test.values)
model_lgb_res = model_lgb.predict(x_test.values)
KRR_res = KRR.predict(x_test.values)

final_res = 0.5 * np.expm1(model_xgb_res) + 0.3 * np.expm1(model_lgb_res) + 0.2 * np.expm1(KRR_res)




submission = pd.read_csv("../data/sample_submission.csv")
submission['SalePrice'] = final_res
submission.to_csv('../data/submission_2.csv', index=None)
print("success")