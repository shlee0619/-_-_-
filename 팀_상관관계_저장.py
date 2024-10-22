import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 한글 폰트 설정
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)

# 데이터 불러오기
file_path = './data/삼성_결합_기록.csv'
data = pd.read_csv(file_path)

# 숫자형 변수만 선택하여 상관관계 계산
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# 승률_x와 다른 변수들 간의 상관관계 계산
correlation_with_win_rate = numeric_data.corr()['승률'].sort_values(ascending=False)

# 상관관계 결과를 데이터프레임으로 변환하고 인덱스를 열로 변환
correlation_df = correlation_with_win_rate.reset_index()

# 열 이름 지정 (변수 이름과 상관관계 값)
correlation_df.columns = ['변수', '상관관계']

# CSV 파일로 저장 (변수 이름과 상관관계 값 포함)
correlation_df.to_csv('삼성상관관계.csv', index=False, encoding='utf-8-sig')



##################### 모델분석 (******제작중***********)



# 원인 결과 설정
X = data[['승률_투수', '볼넷', '출루율', '홈런', '타점', '출루율+장타율', '퀄리티스타트', '삼진_투수', '득점', '세이브']]
y = data['승률']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. 선형분석
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))

# 2. 랜덤 프로스트
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))

# 3. 그라디언트 부스팅
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting R²:", r2_score(y_test, y_pred_gb))
print("Gradient Boosting MSE:", mean_squared_error(y_test, y_pred_gb))

# 4.딥러닝 모델
dl_model = Sequential()
dl_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
dl_model.add(Dense(32, activation='relu'))
dl_model.add(Dense(1))  # Output layer for regression

# 딥러닝모델 컴파일
dl_model.compile(optimizer='adam', loss='mse')

# 모델 training
dl_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# 딥러닝으로 예측
y_pred_dl = dl_model.predict(X_test)
print("Deep Learning R²:", r2_score(y_test, y_pred_dl))
print("Deep Learning MSE:", mean_squared_error(y_test, y_pred_dl))

# 랜덤 포레스트로 예측과 실제 결과 시각화
plt.scatter(y_test, y_pred_rf, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label="Ideal")
plt.xlabel('Actual 승률')
plt.ylabel('Predicted 승률')
plt.title('Random Forest: Predicted vs Actual 승률')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# Random Forest의 예측값과 실제값 비교 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label="Ideal")
plt.title('Random Forest: Predicted vs Actual 승률')
plt.xlabel('Actual 승률')
plt.ylabel('Predicted 승률')
plt.legend()
plt.grid(True)
plt.show()

import altair as alt
import pandas as pd

# 예측값과 실제값 비교를 위한 데이터프레임 생성
result_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_rf.flatten()
})

# Altair를 이용한 시각화
chart = alt.Chart(result_df).mark_point().encode(
    x='Actual',
    y='Predicted',
    tooltip=['Actual', 'Predicted']
).interactive().properties(
    title='Random Forest: Predicted vs Actual 승률'
).configure_axis(
    grid=True
)

chart.display()

import plotly.express as px
import pandas as pd

# 예측값과 실제값 비교를 위한 데이터프레임 생성
result_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_rf.flatten()
})

# Plotly를 이용한 시각화
fig = px.scatter(result_df, x='Actual', y='Predicted', title='Random Forest: Predicted vs Actual 승률')
fig.add_shape(
    type="line",
    x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test),
    line=dict(color="Red",),
)
fig.show()


