import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)

plt.rcParams['axes.unicode_minus'] = False
# 파일 경로 리스트
file_paths = [
    './data/SSG_결합_기록.csv',
    './data/키움_결합_기록.csv',
    './data/두산_결합_기록.csv',
    './data/삼성_결합_기록.csv',
    './data/한화_결합_기록.csv',
    './data/KIA_결합_기록.csv',
    './data/NC_결합_기록.csv',
    './data/KT_결합_기록.csv',
    './data/LG_결합_기록.csv',
    './data/롯데_결합_기록.csv'
]

# 각 구단별 상관관계를 저장할 딕셔너리
correlations = {}

# 각 파일에 대해 이동거리와 승률 간 상관관계 계산
for file_path in file_paths:
    # 파일 불러오기
    data = pd.read_csv(file_path)
    
    # 이동거리와 승률 간 상관관계 계산
    correlation = data['누적이동거리'].corr(data['승률'])
    
    # 파일 이름에서 구단 이름 추출
    team_name = file_path.split('/')[-1].split('_')[0]
    
    # 상관관계 저장
    correlations[team_name] = correlation

# 상관관계 결과 출력
print(correlations)

# 구단 이름과 상관관계 값을 리스트로 변환
teams = list(correlations.keys())
correlation_values = list(correlations.values())

# 막대 그래프로 시각화
plt.figure(figsize=(10, 6))
plt.bar(teams, correlation_values, color='skyblue')
plt.xlabel('구단')
plt.ylabel('누적 이동거리와 승률 간 상관관계')
plt.title('구단별 누적 이동거리와 승률 간 상관관계')
plt.xticks(rotation=45)
plt.ylim(-1, 1)  # 상관계수의 범위는 -1에서 1
plt.tight_layout()
plt.show()