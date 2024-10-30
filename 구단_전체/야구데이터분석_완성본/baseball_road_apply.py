import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import matplotlib
import folium

# 폰트 설정
font_name = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
matplotlib.rc('font', family=font_name)

#거리 데이터
distances_data = {
    '잠실': {'문학': 54, '사직': 400, '대구': 297, '창원': 363, '대전': 172, '고척': 31, '광주': 261, '수원': 36},
    '문학': {'잠실': 54, '사직': 417, '대구': 320, '창원': 386, '대전': 156, '고척': 24, '광주': 269, '수원': 43},
    '사직': {'잠실': 400, '문학': 417, '대구': 101, '창원': 59, '대전': 260, '고척': 414, '광주': 262, '수원': 379},
    '대구': {'잠실': 297, '문학': 320, '사직': 101, '창원': 103, '대전': 163, '고척': 316, '광주': 238, '수원': 235},
    '창원': {'잠실': 363, '문학': 386, '사직': 59, '대구': 103, '대전': 224, '고척': 364, '광주': 212, '수원': 345},
    '대전': {'잠실': 172, '문학': 156, '사직': 260, '대구': 163, '창원': 224, '고척': 188, '광주': 187, '수원': 149},
    '고척': {'잠실': 31, '문학': 24, '사직': 414, '대구': 316, '대전': 188, '창원': 364, '광주': 313, '수원': 33},
    '광주': {'잠실': 261, '문학': 269, '사직': 262, '대구': 238, '창원': 212, '대전': 187, '고척': 313, '수원': 240},
    '수원': {'잠실': 36, '문학': 43, '사직': 379, '대구': 235, '창원': 345, '대전': 149, '고척': 33, '광주': 240},
}


distances_df = pd.DataFrame(distances_data)

# 팀명과 홈구장은 수작업으로 교체
file_path = r'C:\Mtest\개인연습\삼성_데이터.csv'
games_df = pd.read_csv(file_path)
lotte_data = pd.read_csv(file_path)


years, distances, winrates = [], [], []

# 각 구장별 누적거리 계산
def calculate_distance_by_year(year):
    year_data = games_df[games_df['연도'] == year]
    previous_ground = None
    total_distance = 0

    for index, row in year_data.iterrows():
        ground = row['구장']

        if previous_ground is not None and previous_ground != ground:
            # 구장 변화가 있을 경우
            if previous_ground in distances_df.index and ground in distances_df.columns:
                distance = distances_df.at[previous_ground, ground]
                total_distance += distance
                print(f"{year}: {previous_ground}에서 {ground}으로 이동: {distance} km")
            else:
                print(f"{year}: 거리 정보를 찾을 수 없습니다: {previous_ground} -> {ground}")
        
        # 연도와 누적거리, 누적 승률을 입력
        years.append(year)
        distances.append(total_distance)
        winrates.append(lotte_data['누적승률'].iloc[index])

        # 현재 구장을 최신화
        previous_ground = ground

    # 
    if previous_ground != '대구':
        if previous_ground in distances_df.index and '사직' in distances_df.columns:
            return_distance = distances_df.at[previous_ground, '사직']
            total_distance += return_distance
            print(f"{year} 마지막 경기 후 {previous_ground}에서 사직으로 이동: {return_distance} km")
            
    print(f"{year} 총 이동 거리: {total_distance} km")


# 2015년부터 2024년까지 계산
for year in range(2015, 2025):
    calculate_distance_by_year(year)
print(len(distances))


lotte_data['누적이동거리'] = distances
lotte_data.to_csv(file_path, index=False, encoding='utf-8-sig')
