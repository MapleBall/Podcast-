# Podcast-

# 安裝說明

1. 安裝 PyTorch:
   請訪問 PyTorch 官網 (https://pytorch.org/get-started/locally/) 並根據您的系統和 CUDA 版本選擇適當的安裝命令。
   
3. 安装 ffmpeg
   若ffmpeg和ffprobe已安裝则跳過。

#### Ubuntu/Debian 用户
```bash
sudo apt update && sudo apt install ffmpeg
```
#### Windows 用户

- 下載[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

3. 安裝其他依賴:
```python
pip install -r requirements.txt
```

3. FAISS 安裝 (可選):
   - 對於 CPU 版本: pip install faiss-cpu
   - 對於 GPU 版本 (需要 CUDA 支持): pip install faiss-gpu

注意: 
- 確保安裝的 PyTorch 版本與 transformers 庫兼容。如果遇到兼容性問題，可能需要調整 transformers 的版本。
- 如果您使用 GPU 版本的 FAISS，請確保您的 CUDA 環境與 PyTorch 和 FAISS 兼容。

# 依賴項

本項目使用了以下主要的 Python 包：

- numpy 和 scikit-learn：用於數值計算和機器學習任務
- sentence-transformers：用於文本嵌入和相似度計算
- langchain 和 langchain-community：用於構建 AI 應用程序






