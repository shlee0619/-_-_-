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
    'team_data': './data/구단별_10년_승률.csv',
    'defense_data': './data/팀별_수비_기록.csv',
    'hitter_data': './data/팀별_타자_기록.csv',
    'pitcher_data': './data/팀별_투수_기록.csv'
}

team_data = pd.read_csv(file_paths['team_data'], encoding='cp949')
defense_data = pd.read_csv(file_paths['defense_data'])
hitter_data = pd.read_csv(file_paths['hitter_data'])
pitcher_data = pd.read_csv(file_paths['pitcher_data'])

def convert_to_numeric(df):
    # '연도', '순위'를 제외한 나머지 열 선택 후 변환
    cols_to_convert = df.columns.difference(['연도', '순위', '팀명'])
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    return df

# 각 데이터셋에 적용

defense_data = convert_to_numeric(defense_data)
hitter_data = convert_to_numeric(hitter_data)
pitcher_data = convert_to_numeric(pitcher_data)
cols_to_convert = team_data.columns.difference(['연도','팀명'])
team_data[cols_to_convert] = team_data[cols_to_convert].apply(pd.to_numeric, errors='coerce')


# 데이터 요약 및 처리
def summarize_data():
    # 삼성 팀 데이터 필터링
    team_data_grouped = team_data[team_data['팀명'] == '삼성'].drop(columns=['팀명'])  # '팀명' 열 삭제
    hitter_data_grouped = hitter_data[hitter_data['팀명'] == '삼성'].drop(columns=['팀명'])
    pitcher_data_grouped = pitcher_data[pitcher_data['팀명'] == '삼성'].drop(columns=['팀명'])
    defense_data_grouped = defense_data[defense_data['팀명'] == '삼성'].drop(columns=['팀명'])
    print(team_data_grouped)
    print(hitter_data_grouped)
    print(pitcher_data_grouped)
    print(defense_data_grouped)
    # 필요한 열을 수치형으로 변환
    team_data_grouped['승률'] = pd.to_numeric(team_data_grouped['승률'], errors='coerce')
    hitter_data_grouped[['타율', '득점', '홈런']] = hitter_data_grouped[['타율', '득점', '홈런']].apply(pd.to_numeric, errors='coerce')
    pitcher_data_grouped[['ERA', '이닝', '자책점']] = pitcher_data_grouped[['ERA', '이닝', '자책점']].apply(pd.to_numeric, errors='coerce')
    defense_data_grouped[['풋아웃', '어시스트']] = defense_data_grouped[['풋아웃', '어시스트']].apply(pd.to_numeric, errors='coerce')

    # 데이터 병합
    merged = pd.merge(team_data_grouped, hitter_data_grouped, on='연도')
    merged = pd.merge(merged, pitcher_data_grouped, on='연도')
    merged = pd.merge(merged, defense_data_grouped, on='연도')
    print(merged)
    pd.DataFrame(merged).to_csv("./data/삼성_결합_기록.csv", index=False, encoding='utf-8-sig')
    return merged

# 데이터 병합
merged_data = summarize_data()

# 상관관계 분석
def show_correlations():
    # 결측치 제거 후 상관관계 분석
    correlation_matrix = merged_data.dropna().corr()
    top_10_corr = correlation_matrix['승률'].drop('승률').sort_values(ascending=False).head(10)
    print("승률과 상위 10개의 상관관계:")
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
y = merged_data['승률'].dropna()

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
