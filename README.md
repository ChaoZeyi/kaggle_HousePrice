# kaggle_HousePrice

根据我的习惯，在解决一个实际问题时，主要可分为以下几个步骤：问题分析、数据分析与处理、特征工程、模型设计和模型融合。具体原因参见[个人主页](http://lucky365.xin/2017/11/26/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E7%B1%BB%E8%B5%9B%E9%A2%98%E7%9A%84%E8%A7%A3%E9%A2%98%E6%AD%A5%E9%AA%A4/)。

## 问题分析

首先，这是一个多特征的回归问题，预测目标是房屋的价格，最终衡量指标是取log后的RMSE。

给出了多个可能的影响因素，也就是特征。我们要做的就是：选取哪些特征作为模型输入，选择什么模型，使得RMSE最小。

## 数据分析与处理

### 数据分析

1. 总共有1460条训练数据，除ID、价格以外，其余列均为特征
2. 价格数据正常，没有为0的数据，大都分布在50000-500000，最后作为预测目标时，需要取对数

> count      1460.000000
> mean     180921.195890
> std       79442.502883（标准差）
> min       34900.000000
> 25%      129975.000000
> 50%      163000.000000
> 75%      214000.000000
> max      755000.000000
>

3. 计算各列为nan的数据量，如果超过一半则直接去掉这一列

    **训练数据：**

> LotFrontage      259
> **Alley           1369**
> MasVnrType         8
> MasVnrArea         8
> BsmtQual          37
> BsmtCond          37
> BsmtExposure      38
> BsmtFinType1      37
> BsmtFinType2      38
> Electrical         1
> FireplaceQu      690
> GarageType        81
> GarageYrBlt       81
> GarageFinish      81
> GarageQual        81
> GarageCond        81
> **PoolQC          1453**
> **Fence           1179**
> **MiscFeature     1406**
>

​	**测试数据：**

> MSZoning           4
> LotFrontage      227
> **Alley           1352**
> Utilities          2
> Exterior1st        1
> Exterior2nd        1
> MasVnrType        16
> MasVnrArea        15
> BsmtQual          44
> BsmtCond          45
> BsmtExposure      44
> BsmtFinType1      42
> BsmtFinSF1         1
> BsmtFinType2      42
> BsmtFinSF2         1
> BsmtUnfSF          1
> TotalBsmtSF        1
> BsmtFullBath       2
> BsmtHalfBath       2
> KitchenQual        1
> Functional         2
> FireplaceQu      730
> GarageType        76
> GarageYrBlt       78
> GarageFinish      78
> GarageCars         1
> GarageArea         1
> GarageQual        78
> GarageCond        78
> **PoolQC          1456**
> **Fence           1169**
> **MiscFeature     1408**
> SaleType           1
>

加粗的四个特征选择去掉，‘FireplaceQu'缺失值几乎占50%，暂且留着

### 数据处理——缺失值处理

根据上面得到的缺失值信息，分类型进行填充

#### 数值类型

大部分缺失值可以用0来填充，但还是会存在一些特殊情况

1. 该列受其他列的影响

   比如'LotFrontage'明显和'Neighborhood'是相关的，所以我们选择取相同'Neighborhood'的'LotFrontage'的中位数来填充

2. 有明显存在意义，不可能为0的

   比如‘Electrical’，每套房子肯定都存在电路系统，以及‘Exterior1st’， ‘Exterior2nd'，外墙肯定是由某种材料覆盖的，不可能为0。在这种情况下，我们可以考虑采用众数（即出现次数最多的数）来填充

#### 类别类型

观察数据可以看到，大部分缺失的值均表示不存在该类别，如’No basement' 'No garage'，所以直接用None代替。像‘MSZoning’、‘Utilities’类型也选择用众数来填充

### 数据处理——数据编码

直观来看，我们的数据主要分为两类，数值型和字符串型，数值型又可能包含用数字表示的类型，比如‘OverallQual’列，用10代表very excellent，9代表excellent，1代表ver poor，这是用数字表示的类别类型，不能简单的当作数值类型进行归一化等操作。对于类别类型，我们通常有两种编码方式：LabelEncoder和OneHotEncoder，两者的试用场景不同，如果类别特征没有大小意义（如颜色类包含红、绿、黑），适合用one-hot编码，但如果有大小意义（如大小类包含S、M、L、XL、XXL），选择LabelEncoder。

观察我们所有的数据特征，总共可以分为三类处理方法：

1. 单纯数值型：

   > 'LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF',            '2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch', 'ScreenPorch','PoolArea'

   直接当成普通的数值型，后续再进行归一化

2. 数值无大小意义类别型

   > 'MSSubClass'

   先将数值型转为字符型，astype(str)，再用one-hot编码

3. 数值有大小意义类别型

   > 'OverallCond','OverallQual','YearBuilt','YearRemodAdd',

   直接用LabelEncoder编码

4. 字符无大小意义类别型

   > 'MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','Electrical','Functional','GarageType','PavedDrive','SaleType','SaleCondition'

   用one-hot编码

5. 字符有大小意义类别型

   > 'ExterQual','ExterCond', 'BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1', 'BsmtFinType2'        

   先将字符类型转为数字类型，再用LabelEncoder编码

其中有一个问题，OneHotEncoder无法直接对字符串型的类别变量编码，所以我们改而采用pandas模块下的get_dummies函数，同样可以得到哑变量（dummy_variables)）

数据分析和处理的工作是最考验耐性的，需要一遍遍地去看每列数据代表的意义，让我们更熟悉哪些列可能会影响最终的售价。

## 特征工程

### 特征选择

1. 根据相关性热力图

   ```
   impoer seaborn as an
   corr = train_data.corr()
   sn = heatmap(corr)
   plt.show()
   ```

2. 根据random_forest的feature_importance

## 模型建立

我们首先尝试的是xgboost和lightgbm，并采用KFold交叉验证的方式来查看模型的性能

## 模型融合

我做的模型融合比较简单，对多个base_model进行加权平均，得到最终的预测值

得到的最好的结果是Top 8% ( 197 / 2586 ) ，截止到2017.11.28

![leaderboard.png](https://github.com/ChaoZeyi/kaggle_HousePrice/blob/master/pics/leaderboard.png?raw=true)



