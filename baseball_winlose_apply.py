import pandas as pd

file_path = 'C:\Mtest\개인연습\롯데_데이터.csv'
lotte_data = pd.read_csv(file_path)

# Define functions to calculate game result based on scores
def away_team_result(row):
    return '승' if row['원정팀점수'] > row['홈팀점수'] else '패' if row['원정팀점수'] < row['홈팀점수'] else '무'

def home_team_result(row):
    return '승' if row['홈팀점수'] > row['원정팀점수'] else '패' if row['홈팀점수'] < row['원정팀점수'] else '무'

# Apply these functions to add results columns
lotte_data['원정팀결과'] = lotte_data.apply(away_team_result, axis=1)
lotte_data['홈팀결과'] = lotte_data.apply(home_team_result, axis=1)

lotte_data['Wins'] = (lotte_data['원정팀결과'] == '승').cumsum()
lotte_data['Losses'] = (lotte_data['원정팀결과'] == '패').cumsum()

lotte_data['누적승률'] = lotte_data['Wins'] / (lotte_data['Wins'] + lotte_data['Losses'])


lotte_data.to_csv(file_path, index=False, encoding='utf-8-sig')

