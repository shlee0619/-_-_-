import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import matplotlib
import folium
# 폰트 설정
font_name = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
matplotlib.rc('font', family=font_name)

# CSV 파일 불러오기
file_path = 'C:\Mtest\개인연습\롯데_데이터.csv'

lotte_data_latest = pd.read_csv(file_path)
locations = {
    "대구(삼성)": (35.8411289243023, 128.6812363722680),
    "부산(롯데)": (35.19403166, 129.06151836),
    "창원(NC)": (35.2219848625101, 128.579580117268),
    "광주(기아)": (35.1694249627668, 126.888805470329),
    "대전(한화)": (36.3173370007388, 127.428013823451),
    "수원(KT)": (37.2978428909635, 127.011348102567),
    "서울 잠실(LG, 두산)": (37.5112525852452, 127.072863377526),
    "서울 고척(키움)": (37.4980879456876, 126.867026290623),
    "문학" : (37.436734, 126.690554),
    "사직" : (35.19403166, 129.06151836),
    "울산" : (35.535251, 129.311402)
}

lotte_home_location = locations.get("부산(롯데)")
# 각 구장별 롯데 승률 분석

lotte_data_latest['승리여부'] = lotte_data_latest.apply(lambda row: 1 if row['원정팀'] == '롯데' and row['원정팀결과'] == '승' else 0, axis=1)
stadium_win_rates_latest = lotte_data_latest.groupby('구장')['승리여부'].mean().reset_index()
stadium_win_rates_latest.columns = ['구장', '승률']


# 구장 별로 위도, 경도 추가
stadium_win_rates_latest['위도'] = stadium_win_rates_latest['구장'].map({
    "대구": locations["대구(삼성)"][0],
    "부산": locations["부산(롯데)"][0],
    "창원": locations["창원(NC)"][0],
    "광주": locations["광주(기아)"][0],
    "대전": locations["대전(한화)"][0],
    "수원": locations["수원(KT)"][0],
    "잠실": locations["서울 잠실(LG, 두산)"][0],
    "고척": locations["서울 고척(키움)"][0],
    "문학": locations["문학"][0],
    "사직": locations["사직"][0],
    "울산": locations["울산"][0]
})

stadium_win_rates_latest['경도'] = stadium_win_rates_latest['구장'].map({
    "대구": locations["대구(삼성)"][1],
    "부산": locations["부산(롯데)"][1],
    "창원": locations["창원(NC)"][1],
    "광주": locations["광주(기아)"][1],
    "대전": locations["대전(한화)"][1],
    "수원": locations["수원(KT)"][1],
    "잠실": locations["서울 잠실(LG, 두산)"][1],
    "고척": locations["서울 고척(키움)"][1],
    "문학": locations["문학"][1],
    "사직": locations["사직"][1],
    "울산": locations["울산"][1]
})

print(stadium_win_rates_latest)
stadium_win_rates_latest.dropna(inplace=True)
print(stadium_win_rates_latest)


# folium 맵 initialize
map_center = [36.5, 127.5]
map_folium_updated = folium.Map(location=map_center, zoom_start=7)

# Adding updated markers for each stadium
for _, row in stadium_win_rates_latest.iterrows():
    stadium_name = row['구장']
    win_rate = row['승률']
    latitude = row['위도']
    longitude = row['경도']
    if win_rate >= 0.6:
        color = 'darkgreen'
    elif win_rate >= 0.4:
        color = 'green'
    elif win_rate >= 0.2:
        color = 'orange'
    else:
        color = 'red'

    size = 10 + win_rate * 20   
    
    # 승률에 따라서 맵 그리기
    folium.CircleMarker(
        location=(latitude, longitude),
        radius=10 + (win_rate * 10),  # 승률에 따라 원 크기 변화
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        tooltip=f"{stadium_name}: 승률 {win_rate:.2%}"
    ).add_to(map_folium_updated)

#맵 저장
updated_map_path = 'C:\Mtest\개인연습\팀플\야구\data\lotte_stadium_win_rates_map_updated.html'
map_folium_updated.save(updated_map_path)