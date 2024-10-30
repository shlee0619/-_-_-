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

# 데이터 저장을 위한 리스트 초기화
years = []
ranks = []
names = []
team_names = []
eras = []
games = []
wins = []
losses = []
saves = []
holds = []
win_pcts = []
innings_pitched = []
hits_allowed = []
homeruns_allowed = []
walks = []
hit_by_pitch = []
strikeouts = []
runs_allowed = []
earned_runs = []
whips = []

# 시작 연도와 종료 연도 설정
start_year = 2015
end_year = 2024

while start_year <= end_year:
    idx = 1  # 행의 인덱스 초기화
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason')))
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlSeries_ddlSeries')))
    # 연도와 팀 선택
    year_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason'))
    series_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeries_ddlSeries'))

    # 연도 선택
    year_select.select_by_value(str(start_year))

    time.sleep(2)

    # 잠시 대기 후 다시 시리즈 선택 (재탐색으로 stale element 문제 방지)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlSeries_ddlSeries')))
    series_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeries_ddlSeries'))
    series_select.select_by_value("0")
    
    time.sleep(1)
    # 팀 선택이 제대로 이루어지도록 대기
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlTeam_ddlTeam')))
    team_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlTeam_ddlTeam'))
    team_select.select_by_value('HH')

    while True:
        try:
            time.sleep(0.1)
            
            # 각 셀을 한 번에 처리
            row_xpath = f'//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[3]/table/tbody/tr[{idx}]'
            cells = driver.find_elements(By.XPATH, f'{row_xpath}/td')

            # 데이터 추출
            rank = cells[0].text
            name = cells[1].find_element(By.TAG_NAME, 'a').text  # 선수 이름은 <a> 태그에서 추출
            era = cells[3].text
            game = cells[4].text
            win = cells[5].text
            loss = cells[6].text
            save = cells[7].text
            hold = cells[8].text
            win_pct = cells[9].text
            inning_pitched = cells[10].text
            hit_allowed = cells[11].text
            homerun_allowed = cells[12].text
            walk = cells[13].text
            hbp = cells[14].text
            strikeout = cells[15].text
            run_allowed = cells[16].text
            earned_run = cells[17].text
            whip = cells[18].text

            # 데이터를 리스트에 추가
            years.append(start_year)
            ranks.append(rank)
            names.append(name)
            eras.append(era)
            games.append(game)
            wins.append(win)
            losses.append(loss)
            saves.append(save)
            holds.append(hold)
            win_pcts.append(win_pct)
            innings_pitched.append(inning_pitched)
            hits_allowed.append(hit_allowed)
            homeruns_allowed.append(homerun_allowed)
            walks.append(walk)
            hit_by_pitch.append(hbp)
            strikeouts.append(strikeout)
            runs_allowed.append(run_allowed)
            earned_runs.append(earned_run)
            whips.append(whip)

            # 출력
            print(rank, name, era, game, win, loss, save, hold, win_pct, inning_pitched, hit_allowed, homerun_allowed, walk, hbp, strikeout, run_allowed, earned_run, whip)

            idx += 1

        except Exception as e:
            print(f"Error processing row {idx} on {start_year}: {e}")
            break  # 더 이상 데이터가 없으면 while 루프 탈출

    start_year += 1  # 다음 해로 넘어감

# 데이터프레임 생성
df = pd.DataFrame({
    '연도': years,
    '순위': ranks,
    '선수명': names,
    'ERA': eras,
    '경기': games,
    '승': wins,
    '패': losses,
    '세이브': saves,
    '홀드': holds,
    '승률': win_pcts,
    '이닝': innings_pitched,
    '피안타': hits_allowed,
    '피홈런': homeruns_allowed,
    '볼넷': walks,
    '사구': hit_by_pitch,
    '삼진': strikeouts,
    '실점': runs_allowed,
    '자책점': earned_runs,
    'WHIP': whips
})

# CSV 파일로 저장
df.to_csv("한화_투수_기록.csv", index=False, encoding='utf-8-sig')
print(df)

# 웹드라이버 종료
driver.quit()

