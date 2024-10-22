import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import numpy as np

# 한글 폰트 설정
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)

# 데이터 불러오기
file_path = 'C:\Mtest\개인연습\팀플\야구\완성본\구단_전체\data\삼성_결합_기록.csv'
dataF = pd.read_csv(file_path)

# 1. 타겟 변수 (승률)와 피처(나머지 숫자형 변수들) 설정
X = dataF.drop(columns=['연도', '승률', '이닝', '순위', '누적이동거리', '순위_투수', '순위_수비'])
y = dataF['승률']

# 숫자형 데이터만 사용하도록 제한
X_numeric = X.select_dtypes(include=[float, int])

# 2. 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 3. 주성분 분석(PCA)
pca = PCA(n_components=8)

# 4. 결측값을 평균값으로 대체
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_scaled)

# 5. PCA 적용
X_pca = pca.fit_transform(X_imputed)

# 6. 데이터 분할 (훈련/테스트)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 7. 모델 학습 및 평가

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
    'SVR': SVR(kernel='rbf'),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
    'XGBoost Regressor': xgb.XGBRegressor(random_state=42),
    'LightGBM Regressor': lgb.LGBMRegressor(random_state=42),
    'CatBoost Regressor': CatBoostRegressor(verbose=0)
}

# 8. 모델 평가 (R^2 및 MSE 출력)
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{model_name} R^2:', r2_score(y_test, y_pred))
    print(f'{model_name} MSE:', mean_squared_error(y_test, y_pred))

# 9. Neural Network (딥러닝 모델)
model_nn = Sequential()
model_nn.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model_nn.add(Dense(32, activation='relu'))
model_nn.add(Dense(1))  # 출력은 1개 (회귀)
model_nn.compile(optimizer='adam', loss='mse')

# 모델 학습
model_nn.fit(X_train, y_train, epochs=100, verbose=0)

# 예측 및 평가
y_pred_nn = model_nn.predict(X_test)
print('Neural Network R^2:', r2_score(y_test, y_pred_nn))
print("Neural Network MSE:", mean_squared_error(y_test, y_pred_nn))

# 최종 평가
print("모든 모델의 평가가 완료되었습니다.\n")

# 새로운 데이터 입력 (2013년 데이터)
file_paths = {
    'defense_data': 'C:\Mtest\개인연습\팀플\야구\완성본\구단_전체\data\팀별_수비_기록2013.csv',
    'hitter_data': 'C:\Mtest\개인연습\팀플\야구\완성본\구단_전체\data\팀별_타자_기록2013.csv',
    'pitcher_data': 'C:\Mtest\개인연습\팀플\야구\완성본\구단_전체\data\팀별_투수_기록2013.csv'
}

defense_data = pd.read_csv(file_paths['defense_data'])
hitter_data = pd.read_csv(file_paths['hitter_data'])
pitcher_data = pd.read_csv(file_paths['pitcher_data'])



# 수치형 데이터로 변환
def convert_to_numeric(df):
    cols_to_convert = df.columns.difference(['연도', '순위', '팀명'])
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    return df

defense_data = convert_to_numeric(defense_data)
hitter_data = convert_to_numeric(hitter_data)
pitcher_data = convert_to_numeric(pitcher_data)

# 삼성 팀 데이터 병합
def summarize_data():
    hitter_data_grouped = hitter_data[hitter_data['팀명'] == '삼성'].drop(columns=['팀명'])
    pitcher_data_grouped = pitcher_data[pitcher_data['팀명'] == '삼성'].drop(columns=['팀명'])
    defense_data_grouped = defense_data[defense_data['팀명'] == '삼성'].drop(columns=['팀명'])

    merged = pd.merge(hitter_data_grouped, pitcher_data_grouped, on='연도', suffixes=('', '_투수'))
    merged = pd.merge(merged, defense_data_grouped, on='연도', suffixes=('', '_수비'))

    return merged

new_data = summarize_data()

# 불필요한 열 제거
new_data = new_data.drop(columns=['연도', '순위', '순위_투수', '순위_수비', '이닝'])

# 결측값 처리 및 데이터 변환
new_data_imputed = imputer.transform(new_data)
new_data_scaled = scaler.transform(new_data_imputed)
new_data_pca = pca.transform(new_data_scaled)

# 모델을 사용한 새로운 데이터 예측
for model_name, model in models.items():
    predicted_win_rate = model.predict(new_data_pca)
    print(f'{model_name} 예측된 승률:', predicted_win_rate)

# Neural Network 모델을 사용한 예측
predicted_win_rate_nn = model_nn.predict(new_data_pca)
print('Neural Network 예측된 승률:', predicted_win_rate_nn)



# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.impute import SimpleImputer
# from sklearn.decomposition import PCA
# from sklearn.linear_model import Ridge, Lasso
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import numpy as np

# # 한글 폰트 설정
# import matplotlib.font_manager as fm
# font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
# plt.rc('font', family=font_name)

# # 데이터 불러오기
# file_path = './data/삼성_결합_기록.csv'
# dataF = pd.read_csv(file_path)

# # 1. 타겟 변수 (승률)와 피처(나머지 숫자형 변수들) 설정
# X = dataF.drop(columns=['연도', '승률', '이닝', '순위', '누적이동거리', '순위_투수', '순위_수비'])  # '승률'을 제외한 나머지를 피처로 사용
# y = dataF['승률']  # 타겟 변수는 '승률'

# # 숫자형 데이터만 사용하도록 제한
# X_numeric = X.select_dtypes(include=[float, int])

# # 2. 데이터 표준화
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_numeric)


# # 3. 주성분 분석(PCA) - 설명력이 높은 성분을 10개로 설정
# pca = PCA(n_components=8)

# # 4. 회귀분석 적용
# regression = LinearRegression()

# # 5. 결측값을 평균값으로 대체
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X_scaled)

# # 6. PCA 적용
# X_pca = pca.fit_transform(X_imputed)

# # 7. 회귀분석을 위한 데이터 분할 (훈련/테스트)
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# # 8. 회귀분석 적용
# regression.fit(X_train, y_train)

# # 9. 회귀분석 결과 (가중치 및 설명력 출력)
# coefficients = regression.coef_
# r_squared = regression.score(X_test, y_test)

# print('가중치 : ',coefficients, '\nR^2 : ', r_squared)  # 가중치와 R^2 값을 출력

# # 5. 데이터를 훈련/테스트 세트로 분할
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


# ###################기타 다중공선성을 제거하기 위한 시도들 #######################################
# # 1. 타겟 변수 (승률)와 피처(나머지 숫자형 변수들) 설정
# X = dataF.drop(columns=['연도', '누적이동거리', '승률', '이닝', '순위', '누적이동거리', '순위_투수', '순위_수비'])  # '승률'을 제외한 나머지를 피처로 사용
# y = dataF['승률']  # 타겟 변수는 '승률'

# # 숫자형 데이터만 사용하도록 제한
# X_numeric = X.select_dtypes(include=[float, int])

# # 2. 데이터 표준화
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_numeric)

# # 3. 결측값을 평균값으로 대체
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X_scaled)

# # 4. 다중공선성 해결을 위해 VIF (분산 팽창 계수) 계산 함수 추가
# vif_data = pd.DataFrame()
# vif_data['Feature'] = X_numeric.columns
# vif_data['VIF'] = [variance_inflation_factor(X_imputed, i) for i in range(X_imputed.shape[1])]

# # VIF가 10 이상인 변수가 있다면 다중공선성 의심
# high_vif_features = vif_data[vif_data['VIF'] > 10]

# # 5. PCA 적용 (주성분 분석)
# pca = PCA(n_components=8)
# X_pca = pca.fit_transform(X_imputed)

# # 6. 릿지 회귀 적용
# ridge_model = Ridge(alpha=1.0)
# ridge_model.fit(X_pca, y)
# ridge_pred = ridge_model.predict(X_pca)

# # 7. 라쏘 회귀 적용
# lasso_model = Lasso(alpha=0.01)
# lasso_model.fit(X_pca, y)
# lasso_pred = lasso_model.predict(X_pca)

# # 8. VIF 결과와 릿지/라쏘 회귀 결과 반환
# print('#############################VIF 결과와 릿지/라쏘 회귀 결과 반환#############################\n\n', vif_data, ridge_model.coef_, lasso_model.coef_ , '\n\n')

# ################################################################################



# # 6. 여러 모델을 사용해보기



# # (1) Linear Regression
# regression = LinearRegression()
# regression.fit(X_train, y_train)
# y_pred_lr = regression.predict(X_test)
# print('Linear Regression R^2:', r2_score(y_test, y_pred_lr))

# # (2) Random Forest Regressor
# rf_model = RandomForestRegressor(random_state=42)
# rf_model.fit(X_train, y_train)
# y_pred_rf = rf_model.predict(X_test)
# print('Random Forest R^2:', r2_score(y_test, y_pred_rf))

# # (3) Gradient Boosting Regressor
# gb_model = GradientBoostingRegressor(random_state=42)
# gb_model.fit(X_train, y_train)
# y_pred_gb = gb_model.predict(X_test)
# print('Gradient Boosting R^2:', r2_score(y_test, y_pred_gb))

# # (4) TensorFlow Neural Network (딥러닝 모델)
# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1))  # 출력은 1개 (회귀)
# model.compile(optimizer='adam', loss='mse')

# # 모델 학습
# model.fit(X_train, y_train, epochs=100, verbose=0)

# # 예측 및 평가
# y_pred_nn = model.predict(X_test)
# print('Neural Network R^2:', r2_score(y_test, y_pred_nn))

# # 7. 최종 평가
# print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
# print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
# print("Gradient Boosting MSE:", mean_squared_error(y_test, y_pred_gb))
# print("Neural Network MSE:", mean_squared_error(y_test, y_pred_nn))

# # 새로운 데이터 입력 (2013년 데이터)

# file_paths = {
#     'defense_data': './data/팀별_수비_기록2013.csv',
#     'hitter_data': './data/팀별_타자_기록2013.csv',
#     'pitcher_data': './data/팀별_투수_기록2013.csv'
# }
# defense_data = pd.read_csv(file_paths['defense_data'])
# hitter_data = pd.read_csv(file_paths['hitter_data'])
# pitcher_data = pd.read_csv(file_paths['pitcher_data'])



# def convert_to_numeric(df):
#     # '연도', '순위'를 제외한 나머지 열 선택 후 변환
#     cols_to_convert = df.columns.difference(['연도', '순위', '팀명'])
#     df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
#     return df


# defense_data = convert_to_numeric(defense_data)
# hitter_data = convert_to_numeric(hitter_data)
# pitcher_data = convert_to_numeric(pitcher_data)




# def summarize_data():
#     # 삼성 팀 데이터 필터링
    
#     hitter_data_grouped = hitter_data[hitter_data['팀명'] == '삼성'].drop(columns=['팀명'])
#     pitcher_data_grouped = pitcher_data[pitcher_data['팀명'] == '삼성'].drop(columns=['팀명'])
#     defense_data_grouped = defense_data[defense_data['팀명'] == '삼성'].drop(columns=['팀명'])
    
#     print(hitter_data_grouped)
#     print(pitcher_data_grouped)
#     print(defense_data_grouped)
#     # 필요한 열을 수치형으로 변환
    

#     # 데이터 병합
#     merged = pd.merge(hitter_data_grouped, pitcher_data_grouped, on='연도', suffixes=('', '_투수'))

#     # 두 번째 병합: 위에서 병합한 데이터프레임과 defense_data_grouped 병합 (접미사 '_수비' 사용)
#     merged = pd.merge(merged, defense_data_grouped, on='연도', suffixes=('', '_수비'))
#     print(merged)
#     pd.DataFrame(merged).to_csv("./data/삼성_결합_기록2013.csv", index=False, encoding='utf-8-sig')
#     return merged

# new_data = summarize_data()


# # '연도'등 쓸데없는 열을 제거
# new_data = new_data.drop(columns=['연도', '순위', '순위_투수', '순위_수비' , '이닝'])

# # 결측값 처리 (평균값으로 대체)
# imputer = SimpleImputer(strategy='mean')
# new_data_imputed = imputer.fit_transform(new_data)

# # 새로운 데이터도 동일하게 표준화하고 PCA 적용
# new_data_scaled = scaler.transform(new_data_imputed)
# new_data_pca = pca.transform(new_data_scaled)

# # 학습된 그라디언트 부스팅 모델로 예측
# predicted_win_rate = gb_model.predict(new_data_pca)

# # Ridge 회귀를 사용한 예측
# predicted_win_rate_ridge = ridge_model.predict(new_data_pca)
# print('Ridge Regression 예측된 승률:', predicted_win_rate_ridge)

# predicted_win_rate_linear = regression.predict(new_data_pca)
# print('Linear Regression 예측된 승률:', predicted_win_rate_linear)

# predicted_win_rate_lasso = lasso_model.predict(new_data_pca)
# print('Lasso Regression 예측된 승률:', predicted_win_rate_lasso)

# predicted_win_rate_rf = rf_model.predict(new_data_pca)
# print('Random Forest Regression 예측된 승률:', predicted_win_rate_rf)

# predicted_win_rate_gb = gb_model.predict(new_data_pca)
# print('Gradient Boosting Regression 예측된 승률:', predicted_win_rate_gb)

