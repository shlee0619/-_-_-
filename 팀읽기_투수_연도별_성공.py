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
driver.find_element(By.XPATH, '//*[@id="contents"]/div[2]/div[2]/ul/li[2]/a').click()
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason')))

year_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason'))
series_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeries_ddlSeries'))
year_select.select_by_value('2015')
time.sleep(1)
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'cphContents_cphContents_cphContents_ddlSeries_ddlSeries')))
series_select = Select(driver.find_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeries_ddlSeries'))
series_select.select_by_value("0")


years = []
ranks = []
team_names = []

eras = []
gs = []     # Games (경기수)
ws = []     # Wins (승리)
ls = []     # Losses (패배)
svs = []    # Saves (세이브)
hlds = []   # Holds (홀드)
wpcts = []  # Winning Percentage (승률)
ips = []    # Innings Pitched (이닝 수)
hs = []     # Hits (피안타)
hrs = []    # Home Runs (피홈런)
bbs = []    # Walks (볼넷)
hbps = []   # Hit By Pitch (사구)
sos = []    # Strikeouts (삼진)
rs = []     # Runs (실점)
ers = []    # Earned Runs (자책점)
whips = []  # Walks plus Hits per Inning Pitched (WHIP)


cgs = []   # Complete Games (완투)
shos = []  # Shutouts (완봉)
qss = []   # Quality Starts (퀄리티 스타트)
bsvs = []  # Blown Saves (블론 세이브)
tbfs = []  # Total Batters Faced (총 타자 상대)
nps = []   # Number of Pitches (투구 수)
avgs = []  # Batting Average Against (피안타율)
dbs = []   # Doubles (2루타)
tbs = []   # Triples (3루타)
sacs = []  # Sacrifice Hits (희생타)
sfs = []   # Sacrifice Flies (희생 플라이)
ibbs = []  # Intentional Walks (고의 사구)
wps = []   # Wild Pitches (폭투)
bks = []   # Balks (보크)

# 시작 연도와 종료 연도 설정
start_year = 2015
end_year = 2024

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
            

            # 데이터 추출



            rank = cells[0].text
            team_name = cells[1].text
            era = cells[2].text
            g = cells[3].text
            w = cells[4].text
            l = cells[5].text
            sv = cells[6].text
            hld = cells[7].text
            wpct = cells[8].text
            ip = cells[9].text
            h = cells[10].text
            hr = cells[11].text
            bb = cells[12].text
            hbp = cells[13].text
            so = cells[14].text
            r = cells[15].text
            er = cells[16].text
            whip = cells[17].text

            # 데이터를 리스트에 추가
            years.append(start_year)
            ranks.append(rank)
            team_names.append(team_name)
            eras.append(era)
            gs.append(g)
            ws.append(w)
            ls.append(l)
            svs.append(sv)
            hlds.append(hld)
            wpcts.append(wpct)
            ips.append(ip)
            hs.append(h)
            hrs.append(hr)
            bbs.append(bb)
            hbps.append(hbp)
            sos.append(so)
            rs.append(r)
            ers.append(er)
            whips.append(whip)

            driver.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[2]/div/div/a[2]').click()

            time.sleep(0.1)
            row_xpath = f'//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[2]/table/tbody/tr[{idx}]'
            cells = driver.find_elements(By.XPATH, f'{row_xpath}/td')


            cg = cells[3].text
            sho = cells[4].text
            qs = cells[5].text
            bsv = cells[6].text
            tbf = cells[7].text
            np = cells[8].text
            avg = cells[9].text
            db = cells[10].text
            tb = cells[11].text
            sac = cells[12].text
            sf = cells[13].text
            ibb = cells[14].text
            wp = cells[15].text
            bk = cells[16].text
            
            cgs.append(cg)
            shos.append(sho)
            qss.append(qs)
            bsvs.append(bsv)
            tbfs.append(tbf)
            nps.append(np)
            avgs.append(avg)
            dbs.append(db)
            tbs.append(tb)
            sacs.append(sac)
            sfs.append(sf)
            ibbs.append(ibb)
            wps.append(wp)
            bks.append(bk)
            driver.find_element(By.XPATH, '//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[2]/div/div/a[1]').click()

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
    '팀명': team_names,          # Team Names -> 팀명
    'ERA': eras,                # ERA -> 평균 자책점
    '경기수': gs,               # Games -> 경기수
    '승리': ws,                 # Wins -> 승리
    '패배': ls,                 # Losses -> 패배
    '세이브': svs,              # Saves -> 세이브
    '홀드': hlds,               # Holds -> 홀드
    '승률': wpcts,              # Winning Percentage -> 승률
    '이닝': ips,                # Innings Pitched -> 이닝 수
    '피안타': hs,               # Hits -> 피안타
    '피홈런': hrs,              # Home Runs -> 피홈런
    '볼넷': bbs,                # Walks -> 볼넷
    '사구': hbps,               # Hit By Pitch -> 사구
    '삼진': sos,                # Strikeouts -> 삼진
    '실점': rs,                 # Runs -> 실점
    '자책점': ers,              # Earned Runs -> 자책점
    'WHIP': whips,              # WHIP -> WHIP
    '완투': cgs,                # Complete Games -> 완투
    '완봉': shos,               # Shutouts -> 완봉
    '퀄리티스타트': qss,         # Quality Starts -> 퀄리티 스타트
    '블론세이브': bsvs,         # Blown Saves -> 블론 세이브
    '총타자상대': tbfs,         # Total Batters Faced -> 총 타자 상대
    '투구수': nps,              # Number of Pitches -> 투구 수
    '피안타율': avgs,           # Batting Average Against -> 피안타율
    '2루타': dbs,               # Doubles -> 2루타
    '3루타': tbs,               # Triples -> 3루타
    '희생타': sacs,             # Sacrifice Hits -> 희생타
    '희생플라이': sfs,          # Sacrifice Flies -> 희생 플라이
    '고의사구': ibbs,           # Intentional Walks -> 고의 사구
    '폭투': wps,                # Wild Pitches -> 폭투
    '보크': bks                 # Balks -> 보크
})

# CSV 파일로 저장
df.to_csv("팀별_투수_기록.csv", index=False, encoding='utf-8-sig')
print(df)

# 웹드라이버 종료
driver.quit()