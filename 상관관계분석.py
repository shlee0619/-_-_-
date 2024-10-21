import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
# 한글 폰트 설정
font_name = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
matplotlib.rc('font', family=font_name)

# 파일 경로
file_paths = {

    'team_data': r'./data/한화_데이터.csv',
    'defense_data':r'./data/한화_수비_기록.csv',
    'hitter_data':r'./data/한화_타자_기록.csv',
    'pitcher_data':r'./data/한화_투수_기록.csv'
    
}
team_data = pd.read_csv(file_paths['team_data'])
defense_data = pd.read_csv(file_paths['defense_data'])
hitter_data = pd.read_csv(file_paths['hitter_data'])
pitcher_data = pd.read_csv(file_paths['pitcher_data'])

# 팀 데이터 요약
team_data_grouped = team_data.groupby('연도').tail(1)[['연도', '누적승률']].reset_index(drop=True)

# 타자 데이터 요약 및 처리
hitter_data['타율'] = pd.to_numeric(hitter_data['타율'], errors='coerce')
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
merged_data = pd.merge(team_data_grouped, hitter_data_grouped, on='연도')
merged_data = pd.merge(merged_data, pitcher_data_grouped, on='연도')
merged_data = pd.merge(merged_data, defense_data_grouped, on='연도')

# 상관계수 계산
correlation_matrix = merged_data.corr()

# 누적승률과의 상관관계가 가장 높은 10개의 요인 추출
top_10_correlations = correlation_matrix['누적승률'].drop('누적승률').sort_values(ascending=False).head(10)

print(top_10_correlations)

'''
어시스트    0.445688
타율      0.277261
풋아웃     0.197350
홈런      0.147450
자책점     0.101016
ERA     0.086017
득점      0.016315
이닝     -0.194205
연도     -0.508402


어시스트 (0.4457): 수비에서 어시스트는 다른 수비수에게 공을 전달해 아웃을 만들어내는 중요한 역할을 합니다. 이 상관관계는 팀이 더 많은 어시스트를 기록할수록 승률이 증가할 수 있음을 시사합니다.

타율 (0.2773): 타격에서 타율이 높을수록 팀이 적절하게 공을 타격하는 능력을 보여줍니다. 더 많은 안타를 기록할수록 팀이 득점을 만들 가능성이 높아지고, 결국 승률도 높아질 수 있습니다.

풋아웃 (0.1974): 풋아웃은 야수들이 직접적으로 아웃을 잡는 기록입니다. 높은 풋아웃 수는 수비의 안정성을 나타내며, 게임에서 효과적으로 상대팀을 막을 수 있음을 보여줍니다.

홈런 (0.1475): 홈런은 야구 경기에서 즉시 득점을 만들 수 있는 중요한 요소입니다. 홈런이 많을수록 공격력도 상승해 경기의 승률에 긍정적인 영향을 미칠 수 있습니다.

자책점 (0.1010): 자책점은 투수의 실점으로, 이 값이 높을수록 부정적 요인으로 작용해야 하지만, 이 상관계수는 낮기 때문에 실점이 어느 정도 경기 승리에 큰 영향을 미치지 않았을 수 있습니다.

ERA (0.0860): 투수의 평균 자책점(ERA)은 낮을수록 좋은 성적을 의미하지만, 상관관계가 낮아 ERA가 승률에 강한 영향을 주지 않았음을 보여줍니다.

득점 (0.0163): 득점과 승률의 상관관계는 매우 낮습니다. 이는 득점 자체가 누적 승률에 미치는 영향이 그리 크지 않음을 의미합니다.
'''