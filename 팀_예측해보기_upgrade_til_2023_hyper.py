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
import sys
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor


# 한글 폰트 설정
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)

# 데이터 불러오기
# 이름만 수정하면 작동 가능.
file_path = r'C:\Mtest\개인연습\팀플\야구\완성본\구단_전체\data\삼성_결합_기록.csv'
dataF = pd.read_csv(file_path)
data_test = dataF[dataF['연도'] == 2024]
dataF = dataF[dataF['연도'] != 2024]

# 1. 타겟 변수 (승률)와 피처(나머지 숫자형 변수들) 설정

X = dataF.drop(columns=['연도', '승률', '순위',  '이닝', '순위_투수', '순위_수비', '승리', '패배', '도루실패', '피홈런', '자책점', '실점'])
y = dataF['승률']

# 숫자형 데이터만 사용하도록 제한
X_numeric = X.select_dtypes(include=[float, int])

# 2. 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 3. 주성분 분석(PCA)
pca = PCA(n_components=7)

# 4. 결측값을 평균값으로 대체
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_scaled)

# 5. PCA 적용
X_pca = pca.fit_transform(X_imputed)

# 6. 데이터 분할 (훈련/테스트)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 7. 모델 학습 및 평가

models = {
    'LightGBM Regressor': lgb.LGBMRegressor(random_state=42),
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



# 불필요한 열 제거
new_data = data_test.drop(columns=['연도', '승률', '이닝', '순위', '순위_투수', '순위_수비', '승리', '패배', '도루실패', '피홈런', '자책점', '실점'])

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


param_grids = {
    'Linear Regression': {},
    'Ridge Regression': {'alpha': [0.1, 1.0, 10.0]},
    'Lasso Regression': {'alpha': [0.001, 0.01, 0.1, 1.0]},
    'ElasticNet': {'alpha': [0.1, 1.0], 'l1_ratio': [0.3, 0.5, 0.7]},
    'SVR': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    'Decision Tree Regressor': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]},
    'Random Forest Regressor': {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]},
    'Gradient Boosting Regressor': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'XGBoost Regressor': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'LightGBM Regressor': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'CatBoost Regressor': {'iterations': [100, 200], 'learning_rate': [0.01, 0.1]},
    'KNN Regressor': {'n_neighbors': [3, 5, 7]},
}

def build_model(hidden_units=32, learning_rate=0.01):
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(hidden_units // 2, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
    return model

model_nn = KerasRegressor(build_fn=build_model, verbose=0)

def build_model(hidden_units=32, learning_rate=0.01):
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(hidden_units // 2, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
    return model

# scikeras에서는 모델 함수를 'model' 파라미터로 전달합니다.
model_nn = KerasRegressor(model=build_model, verbose=0)

param_grid_nn = {
    'model__hidden_units': [32, 64, 128],
    'model__learning_rate': [0.001, 0.01, 0.1],
    'epochs': [50, 100],
    'batch_size': [8, 16, 32]
}

grid_search_nn = GridSearchCV(
    estimator=model_nn,
    param_grid=param_grid_nn,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search_nn.fit(X_train, y_train)
best_nn_model = grid_search_nn.best_estimator_
print(f"새로운 Neural Network 최적의 파라미터: {grid_search_nn.best_params_}")

# 예측 및 평가
y_pred_nn = best_nn_model.predict(X_test)
print('새로운 Neural Network R^2:', r2_score(y_test, y_pred_nn))
print("새로운 Neural Network MSE:", mean_squared_error(y_test, y_pred_nn))

# 새로운 데이터에 대한 예측
predicted_win_rate_nn = best_nn_model.predict(new_data_pca)
print('새로운 Neural Network 예측된 승률:', predicted_win_rate_nn)


best_models = {}
for model_name, model in models.items():
    print(f"\n{model_name} 하이퍼파라미터 튜닝 중...")
    param_grid = param_grids.get(model_name, {})
    if param_grid:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"최적의 파라미터: {grid_search.best_params_}")
    else:
        # 튜닝할 파라미터가 없으면 기본 모델 사용
        model.fit(X_train, y_train)
        best_model = model
        print("기본 파라미터로 모델 학습 완료.")
    y_pred = best_model.predict(X_test)
    print(f'{model_name} R^2:', r2_score(y_test, y_pred))
    print(f'{model_name} MSE:', mean_squared_error(y_test, y_pred))
    best_models[model_name] = best_model


for model_name, model in best_models.items():
    predicted_win_rate = model.predict(new_data_pca)
    print(f'{model_name} 새롭게 예측된 승률:', predicted_win_rate)



