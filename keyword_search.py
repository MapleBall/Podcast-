import os
import shutil

def search_and_copy_files(src_folder, dest_folder, keyword):
    """
    搜尋指定資料夾內所有文本檔案中包含關鍵字的檔案，並複製到新的資料夾內。
    
    :param src_folder: 原始資料夾路徑
    :param dest_folder: 目標資料夾路徑
    :param keyword: 搜尋的關鍵字
    """
    # 確保目標資料夾存在
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # 遍歷原始資料夾內的檔案
    for filename in os.listdir(src_folder):
        # 確認檔案是否是文本檔案
        if filename.endswith(".txt"):
            src_file_path = os.path.join(src_folder, filename)
            
            # 開啟檔案並搜尋關鍵字
            with open(src_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if keyword in content:
                    # 複製符合條件的檔案到目標資料夾
                    dest_file_path = os.path.join(dest_folder, filename)
                    shutil.copy(src_file_path, dest_file_path)
                    print(f'檔案 {filename} 已複製到 {dest_folder}')

# # 使用範例
# src_folder = r"D:\Podcast_mp3存放區\轉錄文本存放區\關鍵字搜尋\老高與小茉 Mr & Mrs Gao"  # 原始資料夾路徑
# dest_folder = r"D:\Podcast_mp3存放區\轉錄文本存放區\關鍵字搜尋\finded_keyword"  # 目標資料夾路徑
# keyword = '食物'  # 關鍵字
# search_and_copy_files(src_folder, dest_folder, keyword)
