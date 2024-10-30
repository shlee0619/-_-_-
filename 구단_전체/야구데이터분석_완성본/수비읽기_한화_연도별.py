import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 웹드라이버 초기화
url = "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx?sort=HRA_RT"
driver = webdriver.Chrome()
driver.get(url)
driver.find_element(By.XPATH, '//*[@id="contents"]/div[2]/div[2]/ul/li[3]/a').click()



years = []
ranks = []
player_names = []

positions = []
games = []
starts = []
innings_pitched = []
errors = []
pick_offs = []
put_outs = []
assists = []
double_plays = []
fielding_pcts = []
passed_balls = []
stolen_bases = []
caught_stealings = []
caught_stealing_pcts = []

# 시작 연도와 종료 연도 설정
start_year = 2015
end_year = 2024

while start_year <= end_year:
    idx = 1  # 행의 인덱스 초기화

    # 연도와 팀 선택을 페이지가 로드될 때마다 재탐색
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason')))
    

    # 연도와 팀 선택 시마다 Select 요소 재탐색
    year_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason'))
    time.sleep(1)
    # 연도 선택
    year_select.select_by_value(str(start_year))

    # 잠시 대기 후 다시 시리즈 선택 (재탐색으로 stale element 문제 방지)
    
    time.sleep(2)
    
    # 팀 선택이 제대로 이루어지도록 대기
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlTeam_ddlTeam')))
    team_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlTeam_ddlTeam'))
    team_select.select_by_value('HH')

    while True:
        try:
            time.sleep(0.1)
            
            # 각 셀을 한 번에 처리
            row_xpath = f'//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[2]/table/tbody/tr[{idx}]'
            cells = driver.find_elements(By.XPATH, f'{row_xpath}/td')

            # 데이터 추출
            rank = cells[0].text
            player_name = cells[1].find_element(By.TAG_NAME, 'a').text  # 변수명을 'name'에서 'player_name'으로 변경
            
            position = cells[3].text
            game = cells[4].text
            start = cells[5].text
            inning_pitched = cells[6].text
            error = cells[7].text
            pick_off = cells[8].text
            put_out = cells[9].text
            assist = cells[10].text
            double_play = cells[11].text
            fielding_pct = cells[12].text
            passed_ball = cells[13].text
            stolen_base = cells[14].text
            caught_steal = cells[15].text
            caught_steal_pct = cells[16].text

            # 데이터를 리스트에 추가
            years.append(start_year)
            ranks.append(rank)
            player_names.append(player_name)
            
            positions.append(position)
            games.append(game)
            starts.append(start)
            innings_pitched.append(inning_pitched)
            errors.append(error)
            pick_offs.append(pick_off)
            put_outs.append(put_out)
            assists.append(assist)
            double_plays.append(double_play)
            fielding_pcts.append(fielding_pct)
            passed_balls.append(passed_ball)
            stolen_bases.append(stolen_base)
            caught_stealings.append(caught_steal)
            caught_stealing_pcts.append(caught_steal_pct)

            # 출력
            print(rank, player_name, position, game, start, inning_pitched, error, pick_off, put_out, assist, double_play, fielding_pct, passed_ball, stolen_base, caught_steal, caught_steal_pct)

            idx += 1

        except Exception as e:
            print(f"Error processing row {idx} on {start_year}: {e}")
            break  

    start_year += 1  # 다음 해로 넘어감

# 데이터프레임 생성
df = pd.DataFrame({
    '연도': years,              # Year -> 연도
    '순위': ranks,              # Rank -> 순위
    '선수명': player_names,     # Name -> 선수명
    '포지션': positions,        # POS (Position) -> 포지션
    '경기수': games,            # G (Games) -> 경기수
    '선발경기수': starts,       # GS (Games Started) -> 선발경기수
    '수비이닝': innings_pitched, # IP (Innings Pitched) -> 투구이닝
    '실책': errors,             # E (Errors) -> 실책
    '견제사': pick_offs,        # PKO (Pick Offs) -> 견제사
    '풋아웃': put_outs,     # PO (Put Outs) -> 풋아웃
    '어시스트': assists,        # A (Assists) -> 어시스트
    '병살타': double_plays,     # DP (Double Plays) -> 병살
    '수비율': fielding_pct,     # FPCT (Fielding Percentage) -> 수비율
    '포일': passed_balls,   # PB (Passed Balls) -> 포일
    '도루허용': stolen_bases,       # SB (Stolen Bases) -> 도루허용
    '도루실패': caught_stealings,# CS (Caught Stealing) -> 도루실패
    '도루저지율': caught_stealing_pcts # CS% (Caught Stealing Percentage) -> 도루저지율
})

# CSV 파일로 저장
df.to_csv("한화_수비_기록.csv", index=False, encoding='utf-8-sig')
print(df)

# 웹드라이버 종료
driver.quit()