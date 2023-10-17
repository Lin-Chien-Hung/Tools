# 實際可以用的

48.py  :  將音檔轉取樣平率(Sample_rate)轉換換為48kHZ。

remake.ipynb  :  將下載下來的音檔(Librispeech)，去掉.txt檔，並將該語者底下所有的音檔整理至上一層資料夾當中

room_fix.py  :  將聲音(音檔)固定在四個象限，房間大小也固定來去生成4通道音檔。

# Cone of Silence

separation_for_train.py  :  修改原始的seperation程式而來的，可分離多個音檔，且duration設置為與輸入音檔時間長度相同。

# 測試用

room_fix_備份.ipynb  :  生成4通道的音檔，房間環境為固定的。

room_random.ipynb  :  生成4通道的音檔，房間環境為隨機的。
