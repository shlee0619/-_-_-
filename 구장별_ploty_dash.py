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

# Dash 앱 초기화
app = dash.Dash(__name__)

# 앱 레이아웃 설정
app.layout = html.Div([
    html.H1("KBO 팀별 누적 이동거리와 승률 대시보드"),

    # Dropdown for selecting a year
    html.Label("연도를 선택하세요:"),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': str(year), 'value': year} for year in combined_data['연도'].unique()],
        value=combined_data['연도'].min(),  # 기본 값: 가장 낮은 연도
        clearable=False
    ),

    # 그래프 출력
    dcc.Graph(id='scatter-plot'),

    # 승률과 이동거리에 대한 설명
    html.Div(id='output-container', style={'margin-top': '20px'})
])

# 대시보드 상호작용 설정
@app.callback(
    Output('scatter-plot', 'figure'),
    Output('output-container', 'children'),
    Input('year-dropdown', 'value')
)
def update_graph(selected_year):
    # 선택된 연도의 데이터 필터링
    filtered_data = combined_data[combined_data['연도'] == selected_year]

    # Plotly Express를 사용하여 Scatter Plot 생성
    fig = px.scatter(
        filtered_data, 
        x='누적이동거리', 
        y='누적승률', 
        color='원정팀', 
        size='원정팀점수',
        hover_name='원정팀',
        title=f'{selected_year}년 팀별 누적 이동거리와 승률'
    )

    # 승률과 이동거리에 대한 요약 텍스트 생성
    summary = (
        f"선택된 연도: {selected_year}, "
        f"총 {len(filtered_data)}개의 경기가 기록되었습니다."
    )

    return fig, summary

# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True)
