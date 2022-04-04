import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By


url = 'https://www.youtube.com/c/F1/videos'


dr = webdriver.Chrome('./chromedriver')

dr.get(url)

dr.find_element(By.XPATH,"//button[@aria-label='Agree to the use of cookies and other data for the purposes described']").click()

height = dr.execute_script("return document.documentElement.scrollHeight")
lastheight = 0

start = time.perf_counter()
elapsed = 0

while elapsed < 120:
    lastheight = height
    dr.execute_script("window.scrollTo(0, " + str(height) + ");")
    time.sleep(0.25)
    height = dr.execute_script("return document.documentElement.scrollHeight")
    end = time.perf_counter()
    elapsed = end - start
    print(elapsed)

elems = dr.find_elements(By.XPATH, "//a[@id='video-title']")

with open('links.csv','w') as out_file:
    out_file.write(f'URL\n')
    for elem in elems:
        title = elem.text
        if 'Highlights' in title and '2021' in title and 'Grand Prix' in title:
            link = elem.get_attribute('href')
            print(title)
            print(link)
            out_file.write(f'{link}\n')

dr.quit()