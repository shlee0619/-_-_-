import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# 한글 폰트 설정
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)


def analyze_feature_importance(team_name, file_path):
    # 데이터 불러오기
    dataF = pd.read_csv(file_path)

    # 타겟 변수 (승률)와 피처(나머지 숫자형 변수들) 설정
    X = dataF.drop(columns=['연도', '승률', '이닝', '순위', '누적이동거리', '순위_투수', '순위_수비'])
    y = dataF['승률']

    # 숫자형 데이터만 사용하도록 제한
    X_numeric = X.select_dtypes(include=[float, int])

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # 결측값을 평균값으로 대체
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_scaled)

    # 데이터 분할 (PCA 없이)
    X_train_original, X_test_original, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # RandomForest 모델 학습
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train_original, y_train)

    # 특성 중요도 추출 및 시각화 (원래 피처 이름 사용)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 팀 이름과 함께 출력
    print(f"\nTeam: {team_name}")
    print("Feature ranking:")
    for i in range(X_train_original.shape[1]):
        print(f"{i + 1}. feature {X_numeric.columns[indices[i]]} ({importances[indices[i]]})")

    # 특성 중요도 시각화
    plt.figure(figsize=(12, 8))
    plt.title(f"Feature importances for {team_name}")
    plt.bar(range(X_train_original.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train_original.shape[1]), [X_numeric.columns[i] for i in indices], rotation=90)
    plt.xlim([-1, X_train_original.shape[1]])
    plt.show()

# 파일 경로 설정 (팀별 데이터)
team_files = {
    '삼성': './data/삼성_결합_기록.csv',
    '키움': './data/키움_결합_기록.csv',
    '롯데': './data/롯데_결합_기록.csv',
    'KIA' : './data/KIA_결합_기록.csv',
    '두산' : './data/두산_결합_기록.csv',
    'KT' : './data/KT_결합_기록.csv',
    'SSG' : './data/SSG_결합_기록.csv',
    'LG' : './data/LG_결합_기록.csv',
    '한화' : './data/한화_결합_기록.csv',
    'NC' : './data/NC_결합_기록.csv',
}

# 각 팀에 대해 특성 중요도 분석 수행
for team_name, file_path in team_files.items():
    analyze_feature_importance(team_name, file_path)