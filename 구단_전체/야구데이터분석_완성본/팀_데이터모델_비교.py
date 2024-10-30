import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
light_orange = '#faab23'  # 밝은 오렌지색
pale_blue = '#7db0c7'
# Load the uploaded Excel files
file_before = './data/변인 추가 전.xlsx'
file_after = './data/변인 추가 후.xlsx'

# Read the Excel files to inspect the data
data_before = pd.read_excel(file_before)
data_after = pd.read_excel(file_after)


font_name = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
matplotlib.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False 

# 성능 비교를 위한 차이 계산 (실제 승률 - 예측 승률)
data_before['성능 차이'] = data_before['실제 승률'] - data_before['모델 예측 승률']
data_after['성능 차이'] = data_after['실제 승률'] - data_after['모델 예측 승률']

# # 시각화를 위한 데이터 준비
# models_before = data_before['사용한 모델']
# performance_before = data_before['성능 차이']
# models_after = data_after['사용한 모델']
# performance_after = data_after['성능 차이']



teams_before = data_before['팀']
predictions_before = data_before['모델 예측 승률']
actuals_before = data_before['실제 승률']

teams_after = data_after['팀']
predictions_after = data_after['모델 예측 승률']
actuals_after = data_after['실제 승률']

# 시각화: 팀별 예측 승률과 실제 승률 비교 (변인 추가 전/후)
plt.figure(figsize=(12, 8))

# 변인 추가 전
plt.scatter(teams_before, predictions_before, color=light_orange, label='변인 추가 전 - 예측 승률', alpha=0.6)
plt.scatter(teams_before, actuals_before, color=light_orange, marker='x', label='변인 추가 전 - 실제 승률')

# 변인 추가 후
plt.scatter(teams_after, predictions_after, color=pale_blue, label='변인 추가 후 - 예측 승률', alpha=0.6)
plt.scatter(teams_after, actuals_after, color=pale_blue, marker='x', label='변인 추가 후 - 실제 승률')

# 그래프 제목 및 라벨 설정
plt.title('팀별 예측 승률과 실제 승률 비교 (변인 추가 전/후)', fontsize=16)
plt.xlabel('팀명', fontsize=12)
plt.ylabel('승률', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend()

# 그래프 표시
plt.tight_layout()
plt.show()

# 예측 승률과 실제 승률 간의 절대 차이 계산
data_before['오차'] = abs(data_before['실제 승률'] - data_before['모델 예측 승률'])
data_after['오차'] = abs(data_after['실제 승률'] - data_after['모델 예측 승률'])

# 각 데이터셋의 평균 오차 계산
mean_error_before = data_before['오차'].mean()
mean_error_after = data_after['오차'].mean()

labels = ['변인 추가 전', '변인 추가 후']
mean_errors = [mean_error_before, mean_error_after]


# 색상 설정
light_orange = '#faab23'  # 밝은 오렌지색
pale_blue = '#7db0c7'


plt.figure(figsize=(8, 6))
plt.bar(labels, mean_errors, color=[light_orange, pale_blue], alpha=0.7)

# 그래프 제목 및 라벨 설정
plt.title('변인 추가 전후의 평균 오차 비교', fontsize=16)
plt.ylabel('평균 오차', fontsize=12)

# 그래프 표시
plt.tight_layout()
plt.show()