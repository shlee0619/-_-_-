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

# 데이터 저장을 위한 리스트 초기화
years = []
ranks = []
names = []
avgs = []
games = []
plate_as = []
at_bats = []
runs = []
hits = []
hit_2s = []
hit_3s = []
homeruns = []
tbs = []
rbis = []
sac_hits = []
sac_flies = []




bbs = []
ibbs = []
hbps = []
sos = []
gdps = []
slgs = []
obps = []
opss = []
mhs = []
risps = []
ph_bas = []


# 시작 연도와 종료 연도 설정
start_year = 2015
end_year = 2024



while start_year <= end_year:
    idx = 1  # 행의 인덱스 초기화
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason')))
    
    # 연도와 팀 선택
    year_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason'))
    series_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeries_ddlSeries'))

    # 연도 선택
    year_select.select_by_value(str(start_year))
    series_select.select_by_value("0")
    time.sleep(2)
    # 팀 선택이 제대로 이루어지도록 대기
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlTeam_ddlTeam')))
    team_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlTeam_ddlTeam'))
    team_select.select_by_value('HH')
    while True:
        try:
            time.sleep(0.1)
             # 페이지가 로드될 때까지 대기

        
        # 각 셀을 한 번에 처리
            row_xpath = f'//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[3]/table/tbody/tr[{idx}]'
            cells = driver.find_elements(By.XPATH, f'{row_xpath}/td')

            # 데이터 추출
            
            rank = cells[0].text
            name = cells[1].find_element(By.TAG_NAME, 'a').text  # 선수 이름은 <a> 태그에서 추출
            avg = cells[3].text
            game = cells[4].text
            plate_a = cells[5].text
            at_bat = cells[6].text
            run = cells[7].text
            hit = cells[8].text
            hit_2 = cells[9].text
            hit_3 = cells[10].text
            homerun = cells[11].text
            tb = cells[12].text
            rbi = cells[13].text
            sac_hit = cells[14].text
            sac_fly = cells[15].text
            time.sleep(0.1)
            driver.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[2]/div[2]/a[2]').click()

            time.sleep(0.1)
            row_xpath = f'//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[3]/table/tbody/tr[{idx}]'
            cells = driver.find_elements(By.XPATH, f'{row_xpath}/td')
            
            time.sleep(0.1)
            bb = cells[4].text
            ibb = cells[5].text
            hbp = cells[6].text
            so = cells[7].text
            gdp = cells[8].text
            slg = cells[9].text
            obp = cells[10].text
            ops = cells[11].text
            mh = cells[12].text
            risp = cells[13].text
            ph_ba = cells[14].text

            driver.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[2]/div[2]/a[1]').click()

            # 데이터를 리스트에 추가
            years.append(start_year)
            ranks.append(rank)
            names.append(name)
            avgs.append(avg)
            games.append(game)
            plate_as.append(plate_a)
            at_bats.append(at_bat)
            runs.append(run)
            hits.append(hit)
            hit_2s.append(hit_2)
            hit_3s.append(hit_3)
            homeruns.append(homerun)
            tbs.append(tb)
            rbis.append(rbi)
            sac_hits.append(sac_hit)
            sac_flies.append(sac_fly)
            bbs.append(bb)
            ibbs.append(ibb)
            hbps.append(hbp)
            sos.append(so)
            gdps.append(gdp)
            slgs.append(slg)
            obps.append(obp)
            opss.append(ops)
            mhs.append(mh)
            risps.append(risp)
            ph_bas.append(ph_ba)

            # 출력
            print(rank, name, avg, game, plate_a, at_bat, run, hit, hit_2, hit_3, homerun, tb, rbi, sac_hit, sac_fly, bb, ibb, hbp, so, gdp, slg, obp, ops, mh, risp, ph_ba)

            idx += 1
                       
        except Exception as e:
            print(f"Error processing row {idx} on {start_year}: {e}")
            break  # 더 이상 데이터가 없으면 while 루프 탈출

    start_year += 1  # 다음 해로 넘어감
    if start_year > end_year:
        break            


# 데이터프레임 생성
df = pd.DataFrame({
    '연도' : years,       # Year -> 연도
    '순위': ranks,        # Rank -> 순위
    '선수명': names,      # Name -> 선수명
    '타율': avgs,         # AVG (Batting Average) -> 타율
    '경기': games,        # G (Games) -> 경기
    '타석': plate_as,     # PA (Plate Appearances) -> 타석
    '타수': at_bats,      # AB (At Bats) -> 타수
    '득점': runs,         # R (Runs) -> 득점
    '안타': hits,         # H (Hits) -> 안타
    '2루타': hit_2s,      # 2B (Doubles) -> 2루타
    '3루타': hit_3s,      # 3B (Triples) -> 3루타
    '홈런': homeruns,     # HR (Home Runs) -> 홈런
    '루타': tbs,          # TB (Total Bases) -> 루타
    '타점': rbis,         # RBI (Runs Batted In) -> 타점
    '희생번트': sac_hits, # SAC (Sacrifice Hits) -> 희생번트
    '희생플라이': sac_flies, # SF (Sacrifice Flies) -> 희생플라이
    '볼넷': bbs,
    '고의타구': ibbs,
    '사구': hbps,
    '삼진': sos,
    '병살타': gdps,
    '장타율': slgs,
    '출루율': obps,
    '출루율+장타율': opss,
    '멀티히트': mhs,
    '득점권타율': risps,
    '대타타율': ph_bas 
})

# CSV 파일로 저장
df.to_csv("한화_타자_기록.csv", index=False, encoding='utf-8-sig')
print(df)

# 웹드라이버 종료
driver.quit()
