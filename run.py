import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import preprocess
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("./data/victrious_ace_all_12_21.csv")

df = preprocess.preprocessing(df)

df = df.dropna()

def add_feature(df):
    df["lambda"]=1.5*(df["swp_mean"]**2)
    df["wave_velocity"] = df["lambda"]/df["swp_mean"]
    df["beta"] = df["wave_velocity"]/df["Wind Speed(REL)"]
    
    return df

df = add_feature(df)

input_columns=[
        #船舶関連
        'Speed(Ground)', 'Speed(Water)', 'Heading',
       'DRAFT FORE', 'DRAFT AFT', 'M/E OUTPUT',
        #'Heading_sin', 'Heading_cos',
        #'Speed(Water)_sin', 'Speed(Water)_cos' ,
       # "Water_speed",
         "Heaving(Motion_Standard)",
        #"Heaving(Motion_Mean)",
         #"Pitching(Motion_Mean)","Pitching(Motion_Standard)",
        #"Rolling(Motion_Mean)","Rolling(Motion_Standard)",
        #"lambda","beta","wave_velocity",
        
        #風関連
        'Wind Speed(REL)', "Wind Angle(REL)",
        "Wind Speed(TRUE)","Wind Angle(TRUE)",
        'swp_mean',"swp_fore","swp_stbd","swp_port",
        #'us', 'vs', 'windspeed_ship',
        #'Wind Angle(REL)_sin', 'Wind Angle(REL)_cos',
        
        #波関連
        #'Wave Direction(Radar)', 'Wave Period(Radar)',
       #'Wave Direction(Primary)', 'Wave Period(Primary)',
       #'Wave Direction(2nd)', 'Wave Period(2nd)',
        #'Wave Direction(3rd)','Wave Period(3rd)',       
        #'Wave_Direction(Primary)_sin','Wave_Direction(Primary)_cos',
        #'Wave_Direction(2nd)_sin','Wave_Direction(2nd)_cos',
        #'Wave_Direction(3rd)_sin','Wave_Direction(3rd)_cos',
        #'Wave_Direction(Radar)_sin','Wave_Direction(Radar)_cos'
      ]

params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': "rmse",
        'learning_rate': 0.01,
        "max_depth":9,
        'bagging_fraction' : 0.9,
        'bagging_freq': 10,
        'min_data_in_leaf': 70,
        'num_iteration': 100000,
        "lambda_l1":0.7,
        "lambda_l2":0.7,
        'verbose': 1
}

df = df[(df["swh_mean"]>0)& (df["swh_mean"]<5)]

test_data = df[-2000:]

X_test = test_data[input_columns]
Y_test = test_data["swh_mean"]

X_train, X_valid, Y_train, Y_valid = train_test_split(df[input_columns][:-2000], df["swh_mean"][:-2000], test_size =0.2,random_state = 42)


# 訓練・テストデータの設定
train_data = lgb.Dataset(X_train, label=np.log(Y_train+1))
eval_data = lgb.Dataset(X_valid, label=np.log(Y_valid+1), reference= train_data)



model =lgb.train(params,
               train_data,
               valid_sets=[train_data,eval_data],early_stopping_rounds=50,
               verbose_eval=100)

Y_pred = model.predict(X_test,num_iteration=model.best_iteration) # 検証データを用いて目的変数を予測
Y_pred = np.exp(Y_pred)-1
plt.figure(figsize=[5,5])
plt.scatter(Y_pred, Y_test, color = 'blue', marker=".")      # 残差をプロット 
#plt.hlines(y = 0, xmin = 0, xmax = 4, color = 'black') # x軸に沿った直線をプロット
plt.plot([0,7],[0,7],ls="-", color="red")
plt.xlim([0,7])
plt.ylim([0,7])

plt.title('LGB result')                                # 図のタイトル
plt.xlabel('Predicted Values')                            # x軸のラベル
plt.ylabel("True Values")                                   # y軸のラベル
plt.grid()                                                # グリッド線を表示
plt.show()          


Y_train_pred = np.exp(model.predict(X_train,num_iteration=model.best_iteration))-1  # 学習データに対する目的変数を予測
print('RMSE train data: ', round(np.sqrt(mean_squared_error(Y_train, Y_train_pred)),3)) # 学習データを用いたときの平均二乗誤差を出力
print('RMSE test data: ', round(np.sqrt(mean_squared_error(Y_test, Y_pred)),3))    # 検証データを用いたときの平均二乗誤差を出力
print("r2", round(r2_score(Y_test, Y_pred),3))
print("相関係数",round(np.corrcoef(Y_test,Y_pred)[0][1],3))