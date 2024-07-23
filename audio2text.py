import whisper
from tqdm import tqdm
import os

# 加載 Whisper 模型
model = whisper.load_model("base")

def transcribe_audio(file_path, output_file):
    if os.path.exists(file_path):
        pass
    else:
        print(f"文件不存在: {file_path}")
        return
    
    # 進行音訊轉錄
    result = model.transcribe(file_path)
    
    # 開啟文本文件，準備寫入轉錄結果
    with open(output_file, "w", encoding="utf-8") as f:
        segments = result["segments"]
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]  # 取得時間範圍內的文本內容
            
            # 轉換時間格式為 min:sec
            start_min, start_sec = divmod(start_time, 60)
            end_min, end_sec = divmod(end_time, 60)
            formatted_start_time = f"{int(start_min):02}:{int(start_sec):02}"
            formatted_end_time = f"{int(end_min):02}:{int(end_sec):02}"
            
            formatted_text = f"({formatted_start_time}~{formatted_end_time}) {text}\n"
            f.write(formatted_text)  # 將格式化的文本寫入文件

    return output_file

def transcribe_folder(input_folder, output_folder):
    # 確認輸出資料夾是否存在，若不存在則創建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 收集所有音訊文件
    audio_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # 確認是音訊文件
            if file.lower().endswith(('.mp3', '.wav', '.m4a')):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)

    # 使用 tqdm 創建進度條並遍歷音訊文件
    with tqdm(total=len(audio_files), desc="Transcribing Files") as pbar:
        for file_path in audio_files:
            output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.txt")
            
            # 調用轉錄函數
            transcribe_audio(file_path, output_file)
            
            # 更新進度條
            pbar.update(1)

# 確認音訊文件資料夾路徑
input_folder = r"D:\podrecom-main\podrecom-main\podcast_crawler\科技浪 Techwav"
output_folder = r"D:\podrecom-main\transcriptions_text_list"

# 調用函數進行資料夾中的所有文件轉錄
transcribe_folder(input_folder, output_folder)
