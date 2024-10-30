import pandas as pd
import matplotlib
import altair as alt
import os
from matplotlib import font_manager

# 한글 폰트 설정
matplotlib.rc('font', family=matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name())

# 파일 경로 설정 및 데이터 로드
file_path = r'C:\Mtest\개인연습\두산_데이터.csv'
combined_data = pd.read_csv(file_path) if os.path.exists(file_path) else None

# 데이터가 없을 경우 중단
if combined_data is None:
    print(f"파일 경로가 잘못되었습니다: {file_path}")
else:
    # '원정팀'이 'KIA'인 데이터만 필터링
    combined_data = combined_data[combined_data['원정팀'] == '두산']

    # '연도'를 문자열로 변환
    combined_data['연도'] = combined_data['연도'].astype(str)

    # '점수차이' 컬럼 추가 (홈팀점수 - 원정팀점수)
    combined_data['점수차이'] = combined_data['홈팀점수'] - combined_data['원정팀점수']

    # Scatter plot with line, color gradients, size for score difference
    scatter_line_chart = alt.Chart(combined_data).mark_circle(size=100).encode(
        x=alt.X('누적이동거리', title='누적 이동거리 (km)'),
        y=alt.Y('누적승률', title='누적 승률'),
        color=alt.Color('연도:N', legend=alt.Legend(title="연도")),  # Color by year
        size=alt.Size('점수차이', title='점수 차이'),
        tooltip=['원정팀', '연도', '누적승률', '누적이동거리', '점수차이']
    ).properties(
        title='두산 원정거리와 승률의 상관관계 분석',
        width=600,
        height=400
    ).interactive()

    # Line chart to show cumulative trends
    line_chart = alt.Chart(combined_data).mark_line(point=True).encode(
        x=alt.X('누적이동거리', title='누적 이동거리 (km)'),
        y=alt.Y('누적승률', title='누적 승률'),
        color=alt.Color('연도:N', legend=alt.Legend(title="연도")),
        tooltip=['원정팀', '연도', '누적승률', '누적이동거리']
    ).properties(
        title='두산 연도별 원정거리와 승률의 추세',
        width=600,
        height=400
    ).interactive()

    # Heatmap to show the density of win rates over ranges of travel distance
    heatmap = alt.Chart(combined_data).mark_rect().encode(
        x=alt.X('누적이동거리:Q', bin=alt.Bin(maxbins=30), title='누적 이동거리 (km)'),
        y=alt.Y('누적승률:Q', bin=alt.Bin(maxbins=30), title='누적 승률'),
        color=alt.Color('count()', scale=alt.Scale(scheme='blueorange'), title='경기 수')
    ).properties(
        title='두산 이동거리와 승률 간 밀도 분석 (히트맵)',
        width=600,
        height=400
    )

    # Combining charts using Altair's concat function
    combined_chart = alt.vconcat(scatter_line_chart, line_chart, heatmap).resolve_scale(
        color='independent'
    )

    # 차트를 HTML로 저장
    output_html_path = '두산_correlation_analysis.html'
    combined_chart.save(output_html_path)
    print(f"차트가 다음 경로에 저장되었습니다: {output_html_path}")
