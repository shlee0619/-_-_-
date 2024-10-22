import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 웹드라이버 초기화
url = "https://www.koreabaseball.com/Record/Team/Hitter/BasicOld.aspx"
driver = webdriver.Chrome()
driver.get(url)
time.sleep(1)
driver.find_element(By.XPATH, '//*[@id="contents"]/div[2]/div[2]/ul/li[3]/a').click()
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason')))

year_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason'))

year_select.select_by_value('2013')
time.sleep(1)



years = []
ranks = []
team_names = []
gs = []     # Games (경기수)
es = [] # E: 실책
pkos = [] # PKO: 견제사
pos = [] # PO: 풋아웃
assists = [] # ASSIST: 어시스트
dps = [] # DP: 병살
fpcts = [] # FPCT: 수비율
pbs = [] # PB : 포일
sbs = [] # SB: 도루허용
css = [] # CS: 도루실패
cs_percents = [] # CS%: 도루저지율

# 시작 연도와 종료 연도 설정
start_year = 2013
end_year = 2013

while start_year <= end_year:
    idx = 1  # 행의 인덱스 초기화
    # 잠시 대기 후 다시 시리즈 선택 (재탐색으로 stale element 문제 방지)
    
    time.sleep(2)
    

    while True:
        try:
            time.sleep(0.1)
            
            # 각 셀을 한 번에 처리
            row_xpath = f'//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[2]/table/tbody/tr[{idx}]'

            cells = driver.find_elements(By.XPATH, f'{row_xpath}/td')
            


            rank = cells[0].text
            team_name = cells[1].text
            g = cells[2].text
            e = cells[3].text
            pko = cells[4].text
            po = cells[5].text
            assist = cells[6].text
            dp = cells[7].text
            fpct = cells[8].text
            pb = cells[9].text
            sb = cells[10].text
            cs = cells[11].text
            csp = cells[12].text


            # 데이터를 리스트에 추가
            years.append(start_year)
            ranks.append(rank)
            team_names.append(team_name)
            gs.append(g)
            es.append(e)
            pkos.append(pko)
            pos.append(po)
            assists.append(assist)
            dps.append(dp)
            fpcts.append(fpct)
            pbs.append(pb)
            sbs.append(sb)
            css.append(cs)
            cs_percents.append(csp)


            # 출력
            print(rank, team_name)

            idx += 1

        except Exception as e:
            print(f"Error processing row {idx} on {start_year}: {e}")
            break  

    start_year += 1  # 다음 해로 넘어감
    if(start_year>end_year):
        break
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason')))
    year_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason'))
    year_select.select_by_value(str(start_year))

# 데이터프레임 생성
df = pd.DataFrame({
    '연도': years,              # Year -> 연도
    '순위': ranks,              # Rank -> 순위
    '팀명': team_names,         # Team Names -> 팀명
    '경기수': gs,               # Games -> 경기수
    '실책': es,                 # Errors -> 실책
    '견제사': pkos,             # Pick Offs -> 견제사
    '풋아웃': pos,              # Put Outs -> 풋아웃
    '어시스트': assists,        # Assists -> 어시스트
    '병살': dps,                # Double Plays -> 병살
    '수비율': fpcts,            # Fielding Percentage -> 수비율
    '포일': pbs,                # Passed Balls -> 포일
    '도루허용': sbs,            # Stolen Bases -> 도루허용
    '도루실패': css,            # Caught Stealing -> 도루실패
    '도루저지율': cs_percents   # Caught Stealing Percentage -> 도루저지율
})
# CSV 파일로 저장
df.to_csv("팀별_수비_기록2013.csv", index=False, encoding='utf-8-sig')
print(df)

# 웹드라이버 종료
driver.quit()