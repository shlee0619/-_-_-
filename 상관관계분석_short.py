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
file_paths = {
    'team_data': './data/한화_데이터.csv',
    'defense_data': './data/한화_수비_기록.csv',
    'hitter_data': './data/한화_타자_기록.csv',
    'pitcher_data': './data/한화_투수_기록.csv'
}

team_data = pd.read_csv(file_paths['team_data'])
defense_data = pd.read_csv(file_paths['defense_data'])
hitter_data = pd.read_csv(file_paths['hitter_data'])
pitcher_data = pd.read_csv(file_paths['pitcher_data'])

# 데이터 요약 및 처리
def summarize_data():


    team_data['누적승률'] = pd.to_numeric(team_data['누적승률'], errors = 'coerce')
    team_data_grouped = team_data.groupby('연도').tail(1)[['연도', '누적승률']].reset_index(drop=True)

    # 타자 데이터 요약 및 처리
    hitter_data['타율'] = pd.to_numeric(hitter_data['타율'], errors='coerce')
    hitter_data['득점'] = pd.to_numeric(hitter_data['득점'], errors='coerce')
    hitter_data['홈런'] = pd.to_numeric(hitter_data['홈런'], errors='coerce')
    hitter_data_grouped = hitter_data.groupby('연도').agg({'타율': 'mean', '득점': 'sum', '홈런': 'sum'}).reset_index()

    # 투수 데이터 요약 및 처리
    pitcher_data['ERA'] = pd.to_numeric(pitcher_data['ERA'], errors='coerce')
    pitcher_data['이닝'] = pd.to_numeric(pitcher_data['이닝'], errors='coerce')
    pitcher_data['자책점'] = pd.to_numeric(pitcher_data['자책점'], errors='coerce')
    pitcher_data_grouped = pitcher_data.groupby('연도').agg({'ERA': 'mean', '이닝': 'sum', '자책점': 'sum'}).reset_index()

    # 수비 데이터 요약 및 처리
    defense_data['실책'] = pd.to_numeric(defense_data['실책'], errors='coerce')
    defense_data['풋아웃'] = pd.to_numeric(defense_data['풋아웃'], errors='coerce')
    defense_data['어시스트'] = pd.to_numeric(defense_data['어시스트'], errors='coerce')
    defense_data_grouped = defense_data.groupby('연도').agg({'풋아웃': 'sum', '어시스트': 'sum'}).reset_index()

    # 데이터 병합
    merged = pd.merge(team_data_grouped, hitter_data_grouped, on='연도')
    merged = pd.merge(merged, pitcher_data_grouped, on='연도')
    merged = pd.merge(merged, defense_data_grouped, on='연도')
    
    return merged

# 데이터 병합
merged_data = summarize_data()

# 상관관계 분석
def show_correlations():
    correlation_matrix = merged_data.corr()
    top_10_corr = correlation_matrix['누적승률'].drop('누적승률').sort_values(ascending=False).head(10)
    print("누적승률과 상위 10개의 상관관계:")
    print(top_10_corr)

# 상관관계 출력
show_correlations()

# 모델링 함수 정의
def run_models(X, y):
    # 훈련/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. 다중 회귀 모델
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    print("다중 회귀 R^2:", lr_model.score(X_test, y_test))

    # 2. 랜덤 포레스트 회귀 모델
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("랜덤 포레스트 R^2:", r2_score(y_test, y_pred_rf))
    print("랜덤 포레스트 MSE:", mean_squared_error(y_test, y_pred_rf))

    # 3. 그라디언트 부스팅 회귀 모델
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    print("그라디언트 부스팅 R^2:", r2_score(y_test, y_pred_gb))
    print("그라디언트 부스팅 MSE:", mean_squared_error(y_test, y_pred_gb))

    return X_train, X_test, y_train, y_test

# 독립 변수와 종속 변수 설정
X = merged_data[['타율', '득점', '홈런', 'ERA', '이닝', '자책점', '풋아웃', '어시스트']].dropna()
y = merged_data['누적승률'].dropna()

# 모델 실행
X_train, X_test, y_train, y_test = run_models(X, y)

# 딥러닝 모델
def run_deep_learning_model(X_train, X_test, y_train, y_test):
    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 딥러닝 모델 생성
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # 모델 학습
    history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=32)

    # 예측 및 평가
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("딥러닝 모델 R^2:", r2)
    print("딥러닝 모델 MSE:", mse)

    # 학습 곡선 시각화
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# 딥러닝 모델 실행
run_deep_learning_model(X_train, X_test, y_train, y_test)
