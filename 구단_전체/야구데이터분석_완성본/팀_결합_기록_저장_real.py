import pandas as pd
import matplotlib.pyplot as plt


# 한글 폰트 설정
import matplotlib.font_manager as fm
font_name = fm.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)

# 데이터 불러오기
file_paths = {
    'team_data': './data/구단별_10년_승률.csv',
    'defense_data': './data/팀별_수비_기록.csv',
    'hitter_data': './data/팀별_타자_기록.csv',
    'pitcher_data': './data/팀별_투수_기록.csv',
    'money_data' : './data/연도별평균연봉.csv',
    'people_data': './data/관중데이터.csv'
}

team_data = pd.read_csv(file_paths['team_data'], encoding='cp949')
defense_data = pd.read_csv(file_paths['defense_data'])
hitter_data = pd.read_csv(file_paths['hitter_data'])
pitcher_data = pd.read_csv(file_paths['pitcher_data'])
money_data = pd.read_csv(file_paths['money_data'])
people_data = pd.read_csv(file_paths['people_data'])

defense_data = defense_data.replace("넥센", "키움")
hitter_data = hitter_data.replace("넥센", "키움")
pitcher_data = pitcher_data.replace("넥센", "키움")

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

defense_data = defense_data.replace("넥센", "키움")
hitter_data = hitter_data.replace("넥센", "키움")
pitcher_data = pitcher_data.replace("넥센", "키움")
defense_data = defense_data.replace("SK", "SSG")
hitter_data = hitter_data.replace("SK", "SSG")
pitcher_data = pitcher_data.replace("SK", "SSG")


# 팀명을 변경해서 저장.

def summarize_data():
    # SSG 팀 데이터 필터링
    team_data_grouped = team_data[team_data['팀명'] == 'SSG'].drop(columns=['팀명'])  # '팀명' 열 삭제
    hitter_data_grouped = hitter_data[hitter_data['팀명'] == 'SSG'].drop(columns=['팀명'])
    pitcher_data_grouped = pitcher_data[pitcher_data['팀명'] == 'SSG'].drop(columns=['팀명'])
    defense_data_grouped = defense_data[defense_data['팀명'] == 'SSG'].drop(columns=['팀명'])
    money_data_grouped = money_data[money_data['팀명'] == 'SSG'].drop(columns=['팀명'])
    people_data_grouped = people_data[people_data['팀명'] == 'SSG'].drop(columns=['팀명'])
    print(team_data_grouped)
    print(hitter_data_grouped)
    print(pitcher_data_grouped)
    print(defense_data_grouped)
    print(money_data_grouped)
    print(people_data_grouped)
    # 필요한 열을 수치형으로 변환
    team_data_grouped['승률'] = pd.to_numeric(team_data_grouped['승률'], errors='coerce')
    money_data_grouped['평균연봉'] = pd.to_numeric(money_data_grouped['평균연봉'], errors='coerce')
    people_data_grouped['관중 데이터'] = pd.to_numeric(people_data_grouped['관중 데이터'], errors='coerce')


    # 데이터 병합
    merged = pd.merge(team_data_grouped, hitter_data_grouped, on='연도', suffixes=('_팀', '_타자'))
    merged = pd.merge(merged, pitcher_data_grouped, on='연도', suffixes=('', '_투수'))  # 중복 없는 경우 빈 값 사용
    merged = pd.merge(merged, defense_data_grouped, on='연도', suffixes=('', '_수비'))
    merged = pd.merge(merged, money_data_grouped, on = '연도')
    merged = pd.merge(merged, people_data_grouped, on = '연도')
    print(merged)
    pd.DataFrame(merged).to_csv("./data/SSG_결합_기록.csv", index=False, encoding='utf-8-sig')
    return merged

# 데이터 병합
merged_data = summarize_data()


