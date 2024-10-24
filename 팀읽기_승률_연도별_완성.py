import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 웹드라이버 초기화
url = "https://www.koreabaseball.com/Record/TeamRank/TeamRank.aspx"
driver = webdriver.Chrome()
driver.get(url)
time.sleep(1)
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlYear')))

year_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlYear'))

#cphContents_cphContents_cphContents_ddlYear
year_select.select_by_value('2015')
time.sleep(1)



years = []
ranks = []
team_names = []
win_rates = []


# 시작 연도와 종료 연도 설정
start_year = 2015
end_year = 2024

while start_year <= end_year:
    idx = 1  # 행의 인덱스 초기화
    # 잠시 대기 후 다시 시리즈 선택 (재탐색으로 stale element 문제 방지)
    
    time.sleep(0.1)
    

    while True:
        try:
            time.sleep(0.1)
            
            # 각 셀을 한 번에 처리
            row_xpath = f'//*[@id="cphContents_cphContents_cphContents_udpRecord"]/table/tbody/tr[{idx}]'
            cells = driver.find_elements(By.XPATH, f'{row_xpath}/td')
            

            year = start_year
            rank = cells[0].text
            team_name = cells[1].text
            win_rate = cells[6].text


            # 데이터를 리스트에 추가
            years.append(start_year)
            ranks.append(rank)
            team_names.append(team_name)
            win_rates.append(win_rate)





            # 출력
            print(year, rank, team_name, win_rate)

            idx += 1

        except Exception as e:
            print(f"Error processing row {idx} on {start_year}: {e}")
            break  

    start_year += 1  # 다음 해로 넘어감
    if(start_year>end_year):
        break
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlYear')))
    year_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlYear'))
    year_select.select_by_value(str(start_year))

# 데이터프레임 생성
df = pd.DataFrame({
    '연도': years,              # Year -> 연도
    '순위': ranks,              # Rank -> 순위
    '팀명': team_names,         # Team Names -> 팀명
    '승률' : win_rates
})
# CSV 파일로 저장
df.to_csv("./data/팀별_승률_기록.csv", index=False, encoding='utf-8-sig')
print(df)

# 웹드라이버 종료
driver.quit()