# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:56:30 2019

@author: Widdy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import preprocessing
import math
#import  datetime
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#from sklearn import datasets
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from xgboost import XGBClassifier
#from xgboost import plot_importance
#from sklearn.metrics import average_precision_score
#import re
#from scipy import stats
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import scale


#讀檔
path='/users/DamienChen/Desktop/python/kaggle/data/'

train = pd.read_csv(path+'train.csv') 
test = pd.read_csv(path+'test.csv') 
sub = pd.read_csv(path+'submission.csv') 
#把 TRAIN的 LABEL切出來
train_y = train['price_per_ping']
train_X = train.iloc[:,:35]
#TRAIN TEST合併
data = pd.concat([train_X,test])
#將不需要用的COLUMN DROP掉
df=data.drop(columns=['備註','非都市土地使用分區','非都市土地使用編定','編號'])
#計算NULL數
df.isnull().sum()
test.isnull().sum()

#補NAN值
df['主要建材'].describe()
df['主要建材'] = df['主要建材'].fillna('鋼筋混凝土造')

df['主要用途'].describe()
df['主要用途'] = df['主要用途'].fillna('住家用')

df['移轉層次'].describe()
df['移轉層次'] = df['移轉層次'].fillna('三層')

df['總樓層數'].describe()
df['總樓層數'] = df['總樓層數'].fillna('五層')

df['車位類別'].describe()
df['車位類別'] = df['車位類別'].fillna('無')

df['都市土地使用分區'].describe()
df['都市土地使用分區'] = df['都市土地使用分區'].fillna('其他')

df['num_of_bus_stations_in_100m'].describe()
df['num_of_bus_stations_in_100m'] = df['num_of_bus_stations_in_100m'].fillna(0)

df['income_avg'].describe()
df['income_avg'] = df['income_avg'].fillna(1290)

df['low_use_electricity'].describe()
df['low_use_electricity'] = df['low_use_electricity'].fillna('4.5%')

df['income_var'].describe()
df['income_var'] = df['income_var'].fillna(230)

df['location_type'].describe()
df['location_type'] = df['location_type'].fillna('ROOFTOP')

df['nearest_tarin_station'].describe()
df['nearest_tarin_station'] = df['nearest_tarin_station'].fillna('台北101/世貿中心站')

df['nearest_tarin_station_distance'].describe()
df['nearest_tarin_station_distance'] = df['nearest_tarin_station_distance'].fillna(620)


df['nearest_tarin_station_distance'].describe()
df['nearest_tarin_station_distance'] = df['nearest_tarin_station_distance'].fillna(620)

df['lat'].describe()
df['lat'] = df['lat'].fillna(25.055868)

df['lng'].describe()
df['lng'] = df['lng'].fillna(121.543811)

# SPLIT 建築完成年月 的年份
make_year=[]
for i in range(len(df)):
    make_year.append(str(df.iloc[i,14])[:-6])
    

df['make_year'] =make_year
df['make_year']= df['make_year'].replace("", "90")
df['make_year']= df['make_year'].astype(float)

df['make_year'].describe()





df_f = df.drop(columns=['建築完成年月'])

# SPLIT low use of electricty 的 %
lue=[]
for i in range(len(df_f)):
    lue.append(np.float(df_f.iloc[i,25].split('%')[0]))
    

df_f["low_use_electricity"] = lue

# SPLIT 交易年月日 的年及月

time=[]
for i in range(len(df_f)):
    time.append(df_f.iloc[i,3].split(' ')[0])

df_f["交易年月日"] = time

year=[]
for i in range(len(df_f)):
    year.append(np.float(df_f.iloc[i,3].split('-')[0]))
    
month=[]
for i in range(len(df_f)):
    month.append(np.float(df_f.iloc[i,3].split('-')[1]))
    
df_f["交易年月日"] = year
df_f["month"] = month

df_f['屋齡'] = df_f["交易年月日"] - df['make_year']


#SPLIT 土地區段位置/建物區段門牌 的 區 路 段

road=[]
for i in range(len(df_f)):
    road.append(df_f.iloc[i,6].split('區')[-1])
 
df_f['土地區段位置/建物區段門牌'] =road   
    
road1=[]
for i in range(len(df_f)):
    road1.append(df_f.iloc[i,6].split('路')[0])        
    
df_f['土地區段位置/建物區段門牌'] =road1   

road2=[]
for i in range(len(df_f)):
    road2.append(df_f.iloc[i,6].split('街')[0])    
    
df_f['土地區段位置/建物區段門牌'] =road2     


road3=[]
for i in range(len(df_f)):
    road3.append(df_f.iloc[i,6].split('段')[0])    
    
df_f['土地區段位置/建物區段門牌'] =road3

#將原本 DUMMY值有些 COLUMN 為數值將其保留，由於做完 DUMMY會不見因此拉出來先做再合併
dummy=df_f[['建物現況格局-廳','建物現況格局-房','建物現況格局-衛','num_of_bus_stations_in_100m','make_year','交易年月日','month']]

#將資料做 DUMMY(就是ONEHOT ENCODING)

df_dummy=pd.get_dummies(df_f,prefix=['主要建材','主要用途','交易標的','交易筆棟數','建物型態','建物現況格局-廳','建物現況格局-房','建物現況格局-衛','建物現況格局-隔間','num_of_bus_stations_in_100m'
                                 ,'有無管理組織','移轉層次','總樓層數','車位類別','都市土地使用分區','鄉鎮市區','location_type','nearest_tarin_station','make_year','交易年月日','month','土地區段位置/建物區段門牌'], 
                    columns=['主要建材','主要用途','交易標的','交易筆棟數','建物型態','建物現況格局-廳','建物現況格局-房','建物現況格局-衛','建物現況格局-隔間','num_of_bus_stations_in_100m'
                                 ,'有無管理組織','移轉層次','總樓層數','車位類別','都市土地使用分區','鄉鎮市區','location_type','nearest_tarin_station','make_year','交易年月日','month','土地區段位置/建物區段門牌'])


#將 DUMMY及DF_DUMMY合併
df_dummy=pd.concat([dummy,df_dummy],axis=1)
#把INDEX DROP
df_dummy=df_dummy.drop(columns=['index'])

#正規化(NORMALIZE)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df_dummy)
df_dummy=pd.DataFrame(x_scaled)

#將 TRAIN TEST 分開
train_X = df_dummy.iloc[:69170,:]
test = df_dummy.iloc[69170:,:]

#在 TRAIN 中切 VALIDATION 出來
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.02, random_state=42)

#XGBOOST
#import xgboost as xgb
eval_set = [(X_val, y_val)]


xg = xgb.XGBRegressor(n_estimators=100000, learning_rate=0.07, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=13, objective ='reg:linear')



xg.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=50 ,verbose=True)



predictions = xg.predict(X_val)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_val, predictions)


rmse=math.sqrt(mse)

y_pre = xg.predict(test)

sub['price_per_ping'] = y_pre


sub.to_csv(path+'sub5.csv', index=False)