import pandas as pd
import folium
import matplotlib
from geopy.distance import geodesic
from matplotlib import font_manager

font_name = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
matplotlib.rc('font', family=font_name)


distance_data = pd.read_csv('구단별_10년_이동거리.csv', encoding='cp949')

# 각 홈구장 위치
locations = {
    "삼성": (35.8411289243023, 128.6812363722680),
    "롯데": (35.19403166, 129.06151836),
    "NC": (35.2219848625101, 128.579580117268),
    "KIA": (35.160124, 126.852121),
    "SSG": (37.4980879456876, 126.867026290623),
    "KT": (37.2978428909635, 127.011348102567),
    "LG": (37.5112525852452, 127.072863377526),
    "두산": (37.5112525852452, 127.072863377526),
    "한화": (36.3173370007388, 127.428013823451),
    "키움": (37.4980879456876, 126.867026290623)
}

# 승률과 누적이동거리 열 추가
distance_data['승률'] = distance_data['승률']
distance_data['누적이동거리'] = distance_data['누적이동거리']

# 누적 승률에 따른 원 색깔
def get_color_by_winrate(win_rate):
    if win_rate >= 0.6:
        return 'green'
    elif win_rate >= 0.5:
        return 'yellow'
    else:
        return 'red'

# 누적 승률에 따른 원 크기
def get_radius_by_winrate(win_rate):
    if win_rate >= 0.6:
        return 20
    elif win_rate >= 0.5:
        return 10
    else:
        return 5


for year in distance_data['연도'].unique():
    year_data = distance_data[distance_data['연도'] == year]
    
    # 우리나라 수도를 중심으로
    map_korea_year = folium.Map(location=[36.5, 127.5], zoom_start=7)
    
    for index, row in year_data.iterrows():
        team = row['팀명']
        win_rate = row['승률']
        distance_traveled = row['누적이동거리']
        
        if team in locations:
            folium.CircleMarker(
                location=locations[team],
                radius=get_radius_by_winrate(win_rate),  
                popup=f'{team} ({year}): 승률 {win_rate:.2f}, 이동거리 {distance_traveled:.2f} km',
                color=get_color_by_winrate(win_rate), 
                fill=True,
                fill_color=get_color_by_winrate(win_rate),
                fill_opacity=0.9
            ).add_to(map_korea_year)
    
    # 각 연도별로 html 저장
    map_korea_year.save(f'baseball_team_winrate_map_{year}.html')
