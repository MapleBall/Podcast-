import os
from opencc import OpenCC

# 初始化簡體轉繁體的轉換器
cc = OpenCC('s2t')  # s2t 代表簡體到繁體

# 指定要處理的資料夾路徑
folder_path = r"D:\Podcast_mp3存放區\轉錄文本存放區\老高與小茉 Mr & Mrs Gao"  # 替換成你的資料夾路徑
output_folder_path = os.path.join(folder_path, 'converted_files')

# 創建一個新資料夾來存放轉換後的文件
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 瀏覽資料夾內的所有 .txt 檔案
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        
        # 讀取文件內容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 將內容從簡體翻譯成繁體
        converted_content = cc.convert(content)
        
        # 設置新的文件路徑
        new_filename = f'{filename}'  # 新的文件名稱
        new_file_path = os.path.join(output_folder_path, new_filename)
        
        # 將翻譯後的內容寫入新的文件
        with open(new_file_path, 'w', encoding='utf-8') as file:
            file.write(converted_content)

print("done")
