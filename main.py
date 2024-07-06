import feedparser
import requests
import time
import random
import re
import json
from bs4 import BeautifulSoup
import os

def download_map3(url,name):

    filename = name+".mp3"
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"檔案下載完成: {filename}")
    else:
        print("無法下載檔案")
    
def get_rss_file(newsurl,test_n):
    
    #得到節目名稱
    def get_title_name(url):
        response = requests.get(url)

        # 檢查是否成功獲取網頁內容
        if response.status_code == 200:
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # 尋找包含RSS網址的元素
            find_rss=soup.find_all("",text=re.compile("item"))

            #節目名稱
            title=soup.find_all("title")[0].text
        return title
    

                    
            
    if newsurl[13:21]=="firstory":
        
        # 發送GET請求獲取網頁內容
        response = requests.get(newsurl)

        # 檢查是否成功獲取網頁內容
        if response.status_code == 200:
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # 尋找包含RSS網址的元素
            find_rss=soup.find_all("",text=re.compile("item"))

            #節目名稱  ＃會創造一個同名資料夾
            title=soup.find_all("title")[0].text
            # 指定資料夾路徑
            folder_path = title

            # 檢查資料夾是否存在，如果不存在則創建它
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            
            #有幾集
            n=len(soup.find_all("item"))
            print(title,"一共",n,"集")
            #全部的全部的集數名稱

            if n<=2 or test_n=="all":
                test_n=n

            #for i in range(n):   #抓全部集數      #這邊的迴圈  是指定跑幾個幾集
            #抓2集測試
            for i in range(test_n):
                time.sleep(random.randrange(1,5))
                mp3_url=soup.find_all("item")[i].find('enclosure').get('url')
                name=soup.find_all("item")[i].find('itunes:title').text
                download_map3(mp3_url,title+"/"+name)
                #print(name,"下載完畢")
    else:
        file = feedparser.parse(newsurl)
        #print(file)
        #節目名稱 ＃會創造一個同名資料夾
        title=get_title_name(newsurl)
        # 指定資料夾路徑
        folder_path = title

        # 檢查資料夾是否存在，如果不存在則創建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        
        #有多少集數
        n=len(file["entries"])
        print(title,"一共",n,"集")
        #全部的全部的集數名稱

        if n<=2 or test_n=="all":
            test_n=n
            
        #for i in range(n):   #抓全部集數           #這邊的迴圈  是指定跑幾個幾集
        #抓2集測試
        for i in range(test_n):
            time.sleep(random.randrange(1,5))
            try:
                mp3_url=file["entries"][i]["links"][1]["href"]
            except:
                mp3_url=file["entries"][i]["links"][0]["href"]
            name=file["entries"][i]["title"]
            download_map3(mp3_url,title+"/"+name)
            #print(name,"下載完畢")
            
            
            
        
        
# 測試字符是否為中文
def is_chinese(char):
    # 判斷字符的 Unicode 編碼值是否在中文範圍內
    if '\u4e00' <= char <= '\u9fff':
        return True
    else:
        return False      
    
def get_all_category():
    all_cata=['all',
     'Arts',
     'Arts / Books',
     'Arts / Design',
     'Arts / Fashion & Beauty',
     'Arts / Food',
     'Arts / Performing Arts',
     'Arts / Visual Arts',
     'Business',
     'Business / Careers',
     'Business / Entrepreneurship',
     'Business / Investing',
     'Business / Management',
     'Business / Marketing',
     'Business / Non-Profit',
     'Comedy',
     'Comedy / Comedy Interviews',
     'Comedy / Improv',
     'Comedy / Stand-Up',
     'Education',
     'Education / Courses',
     'Education / How To',
     'Education / Language Learning',
     'Education / Self-Improvement',
     'Fiction',
     'Fiction / Comedy Fiction',
     'Fiction / Drama',
     'Fiction / Science Fiction',
     'Government',
     'Health & Fitness',
     'Health & Fitness / Alternative Health',
     'Health & Fitness / Fitness',
     'Health & Fitness / Medicine',
     'Health & Fitness / Mental Health',
     'Health & Fitness / Nutrition',
     'Health & Fitness / Sexuality',
     'History',
     'Kids & Family',
     'Kids & Family / Education for Kids',
     'Kids & Family / Parenting',
     'Kids & Family / Pets & Animals',
     'Kids & Family / Stories for Kids',
     'Leisure',
     'Leisure / Animation & Manga',
     'Leisure / Automotive',
     'Leisure / Aviation',
     'Leisure / Crafts',
     'Leisure / Games',
     'Leisure / Hobbies',
     'Leisure / Home & Garden',
     'Leisure / Video Games',
     'Music',
     'Music / Music Commentary',
     'Music / Music History',
     'Music / Music Interviews',
     'News',
     'News / Business News',
     'News / Daily News',
     'News / Entertainment News',
     'News / News Commentary',
     'News / Politics',
     'News / Sports News',
     'News / Tech News',
     'Religion & Spirituality',
     'Religion & Spirituality / Buddhism',
     'Religion & Spirituality / Christianity',
     'Religion & Spirituality / Hinduism',
     'Religion & Spirituality / Islam',
     'Religion & Spirituality / Judaism',
     'Religion & Spirituality / Religion',
     'Religion & Spirituality / Spirituality',
     'Science',
     'Science / Astronomy',
     'Science / Chemistry',
     'Science / Earth Sciences',
     'Science / Life Sciences',
     'Science / Mathematics',
     'Science / Natural Sciences',
     'Science / Nature',
     'Science / Physics',
     'Science / Social Sciences',
     'Society & Culture',
     'Society & Culture / Documentary',
     'Society & Culture / Personal Journals',
     'Society & Culture / Philosophy',
     'Society & Culture / Places & Travel',
     'Society & Culture / Relationships',
     'Sports',
     'Sports / Baseball',
     'Sports / Basketball',
     'Sports / Cricket',
     'Sports / Fantasy Sports',
     'Sports / Football',
     'Sports / Golf',
     'Sports / Hockey',
     'Sports / Rugby',
     'Sports / Running',
     'Sports / Soccer',
     'Sports / Swimming',
     'Sports / Tennis',
     'Sports / Volleyball',
     'Sports / Wilderness',
     'Sports / Wrestling',
     'TV & Film',
     'TV & Film / After Shows',
     'TV & Film / Film History',
     'TV & Film / Film Interviews',
     'TV & Film / Film Reviews',
     'TV & Film / TV Reviews',
     'Technology',
     'True Crime']
    all_category=[]
    for i in range(len(all_cata)):
        all_category.append((all_cata[i].replace(" / ","-").replace(" & ","-").replace(" ","-").lower()))
    return all_category


def get_name_href(category_name):
    #抓到總排行榜   全部的網址 跟名稱
    rank_list_herf=[]
    rank_list_name=[]
    # 目標網站的URL

    url="https://rephonic.com/charts/apple/tw/"+category_name
    # 發送GET請求獲取網頁內容
    response = requests.get(url)

    # 檢查是否成功獲取網頁內容
    if response.status_code == 200:
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # 在這裡編寫你的程式碼來處理解析後的網頁內容

        # 以下是一個示例，尋找所有<a>標籤並獲取其連結和文字內容
        for link in soup.find_all('a'):
            href = link.get('href')
            text = link.get_text()
            if len(text)>0 and is_chinese(text)==True:
                #print(f"連結: {href}\t文字內容: {text}")
                rank_list_herf.append("https://rephonic.com"+href)
                rank_list_name.append(text)
            else:
                pass
    else:
        print("無法獲取網頁內容")
    return  rank_list_herf,rank_list_name



def get_rss(url):
    # 目標網頁的URL
    #url = "https://rephonic.com/podcasts/li-jing-lei-de-chen-jing-shi-jian"

    # 發送GET請求獲取網頁內容
    response = requests.get(url)

    # 檢查是否成功獲取網頁內容
    if response.status_code == 200:
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # 尋找包含RSS網址的元素
        find_rss=soup.find_all("",text=re.compile("@context"))

        # 要解析的 JSON 字串
        json_str = find_rss[0]
        # 解析 JSON 字串
        data = json.loads(json_str)
        # 提取 identifier 後面的文字
        identifier_text = data["identifier"]
        print(identifier_text)

    else:
        print("無法獲取網頁內容")
    return identifier_text




def main():
    
    #先得到所有分類的名字
    category_name=get_all_category()
    #使用者輸入想跑多少分類
    category_n=input("你想要跑多少種分類？ 輸入數字，或者all抓取全部")
    if category_n=="all":       
        category_n=len(category_name)
    else:
        category_n=int(category_n)
    
    for i in range(category_n):      #這邊的迴圈  是指定跑幾個分類    
        #得到該分類的所有節目名稱跟網址
        rank_list_herf,rank_list_name=get_name_href(category_name[i])
        #使用者輸入想跑多少節目
        rank_list_n=5        #("一個分類想要抓取多少節目？ 輸入數字，或者all抓取全部")
        if rank_list_n=="all":    
            rank_list_n=len(rank_list_herf)
        else:
            rank_list_n=int(rank_list_n)
        
        for j in range(rank_list_n):     #這邊的迴圈  是指定跑幾個節目
            
            #得到該節目的rss
            rss_url=get_rss(rank_list_herf[j])
            #下載所有節目
            get_rss_file(rss_url,3)    #後面的後面的數字   是測試用的時候。要下載幾集。  ＃要全部的集數。 輸入"all" 
main()



# #貼上rss即可下載。＃新資料夾
# get_rss_file('https://feeds.soundon.fm/podcasts/ecd31076-d12d-46dc-ba11-32d24b41cca5.xml')
# #史塔克實驗室
# get_rss_file("https://feeds.soundon.fm/podcasts/e4f101be-289a-4101-bb11-59fc61e5c88b.xml")
# #達特嘴哥地圖砲 
# get_rss_file("https://open.firstory.me/rss/user/ckcdy2bijlk7n0918zfcwxyyr")