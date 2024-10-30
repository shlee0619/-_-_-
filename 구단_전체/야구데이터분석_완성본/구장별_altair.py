import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
import altair as alt
# 한글 폰트 설정
font_name = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
matplotlib.rc('font', family=font_name)

# 파일 경로
files = {
    'KIA': r'C:\Mtest\개인연습\KIA_데이터.csv',
    'KT': r'C:\Mtest\개인연습\KT_데이터.csv',
    'LG': r'C:\Mtest\개인연습\LG_데이터.csv',
    'NC': r'C:\Mtest\개인연습\NC_데이터.csv',
    'SSG': r'C:\Mtest\개인연습\SSG_데이터.csv',
    '두산': r'C:\Mtest\개인연습\두산_데이터.csv',
    '롯데': r'C:\Mtest\개인연습\롯데_데이터.csv',
    '삼성': r'C:\Mtest\개인연습\삼성_데이터.csv',
    '키움': r'C:\Mtest\개인연습\키움_데이터.csv',
    '한화': r'C:\Mtest\개인연습\한화_데이터.csv'
}

# 모든 CSV 파일을 로드하여 하나의 DataFrame으로 결합
combined_data = pd.concat([pd.read_csv(file) for file in files.values()], ignore_index=True)

# Convert '연도' to string for better handling in Altair
combined_data['연도'] = combined_data['연도'].astype(str)

# 팀별 연도별 승률과 이동거리를 시각화하는 인터랙티브 차트
chart = alt.Chart(combined_data).mark_circle(size=60).encode(
    x=alt.X('누적이동거리', title='누적 이동거리 (km)'),
    y=alt.Y('누적승률', title='누적 승률'),
    color='원정팀',
    tooltip=['원정팀', '연도', '누적승률', '누적이동거리'],
    size=alt.Size('원정팀점수', title='원정팀 점수'),
    facet=alt.Facet('연도', title='연도')
).properties(
    title='팀별 연도별 누적 이동거리와 승률 비교',
    width=200,
    height=200
).interactive()


chart.save('구장별_altair_chart.html')
