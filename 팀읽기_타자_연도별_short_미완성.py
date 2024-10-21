import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 웹드라이버 초기화
url = "https://www.koreabaseball.com/Record/Team/Hitter/Basic1.aspx"
driver = webdriver.Chrome()
driver.get(url)

# 대기 함수
def wait_for_element(by, value, timeout=10):
    time.sleep(1)
    return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((by, value)))

# 셀에서 텍스트 추출 함수
def get_cell_text(row_idx, col_idx):
    xpath = f'//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[2]/table/tbody/tr[{row_idx}]/td[{col_idx}]'
    return driver.find_element(By.XPATH, xpath).text

# 연도별 데이터 수집 함수
def collect_data_for_year(year):
    data = []
    idx = 1
    while True:
        try:
            # 각 셀 데이터를 추출하여 리스트에 저장
            row = [
                year,                                  # 연도
                get_cell_text(idx, 1),                 # 순위
                get_cell_text(idx, 2),                 # 팀명
                get_cell_text(idx, 3),                 # 타율
                get_cell_text(idx, 4),                 # 경기
                get_cell_text(idx, 5),                 # 타석
                get_cell_text(idx, 6),                 # 타수
                get_cell_text(idx, 7),                 # 득점
                get_cell_text(idx, 8),                 # 안타
                get_cell_text(idx, 9),                 # 2루타
                get_cell_text(idx, 10),                # 3루타
                get_cell_text(idx, 11),                # 홈런
                get_cell_text(idx, 12),                # 루타
                get_cell_text(idx, 13),                # 타점
                get_cell_text(idx, 14),                # 희생번트
                get_cell_text(idx, 15)                 # 희생플라이
            ]
            data.append(row)
            idx += 1

        except Exception as e:
            print(f"Error processing row {idx} on {year}: {e}")
            break  # 더 이상 데이터가 없으면 while 루프 탈출

    return data

# 연도 변경 함수
def change_year(year):
    year_select = Select(wait_for_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeason_ddlSeason'))
    series_select = Select(wait_for_element(By.ID, 'cphContents_cphContents_cphContents_ddlSeries_ddlSeries'))
    time.sleep(1)
    year_select.select_by_value(str(year))
    time.sleep(2)
    series_select.select_by_value("0")
    time.sleep(1)

# 데이터 저장을 위한 리스트
data = []

# 시작 연도와 종료 연도 설정
start_year = 2015
end_year = 2024

# 각 연도별 데이터 수집
for year in range(start_year, end_year + 1):
    change_year(year)
    data.extend(collect_data_for_year(year))

# 데이터프레임 생성
df = pd.DataFrame(data, columns=[
    '연도', '순위', '팀명', '타율', '경기', '타석', '타수', '득점', '안타', '2루타', '3루타', '홈런', '루타', '타점', '희생번트', '희생플라이'
])

# CSV 파일로 저장
df.to_csv("팀별_타자_기록.csv", index=False, encoding='utf-8-sig')
print(df)

# 웹드라이버 종료
driver.quit()
