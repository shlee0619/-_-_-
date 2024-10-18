import altair as alt
import pandas as pd

# 데이터 로드 및 전처리
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

combined_data = pd.concat([pd.read_csv(file) for file in files.values()], ignore_index=True)
combined_data['연도'] = combined_data['연도'].astype(str)

# 1. 이동거리와 승률을 보여주는 기본 차트
base_chart = alt.Chart(combined_data).mark_circle(size=60).encode(
    x=alt.X('누적이동거리', title='누적 이동거리 (km)'),
    y=alt.Y('누적승률', title='누적 승률'),
    color='원정팀',
    tooltip=['원정팀', '연도', '누적승률', '누적이동거리'],
    size=alt.Size('원정팀점수', title='원정팀 점수'),
).interactive()

# 2. 히트맵: 연도별 평균 승률 시각화
heatmap = alt.Chart(combined_data).mark_rect().encode(
    x=alt.X('연도:O', title='연도'),
    y=alt.Y('원정팀:O', title='원정팀'),
    color=alt.Color('mean(누적승률):Q', scale=alt.Scale(scheme='blues'), title='평균 승률'),
    tooltip=['연도', '원정팀', 'mean(누적승률)']
).properties(
    title="연도별 팀 승률 히트맵",
    width=300
)

# 3. 바 차트: 선택된 구간에서 팀별 경기 수
selection = alt.selection_interval(encodings=['x', 'y'])

bars = alt.Chart(combined_data).mark_bar().encode(
    x='count():Q',
    y=alt.Y('원정팀:N', sort='-x'),
    color='원정팀:N',
    tooltip=['원정팀', 'count()']
).transform_filter(
    selection
).properties(
    title='선택된 구간 내 팀별 경기 수',
    width=300
)

# 4. Layered Charts: Scatter plot and brushed bar chart combined
layered_chart = alt.vconcat(
    base_chart.add_params(selection),
    bars
)

# 5. 최종 레이어: 히트맵과 상호작용 가능한 차트 병합
final_chart = alt.hconcat(
    layered_chart,
    heatmap
)

# 최종 차트 렌더링

final_chart.save('구장별_altair_chart_expand.html')
