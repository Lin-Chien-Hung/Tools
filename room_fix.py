import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import soundfile as sf
import random
from tqdm import tqdm 

#=========================================================================================
# 資料夾讀取
for path in glob.glob(os.path.join("train-clean-100", "*")):
    
    print(path)
    filename = re.sub(r"train-clean-100/","",path)
    file_dir = re.sub(r"train-clean-100","mix",path)
    
    # 資料夾不存在創建
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    #=========================================================================================
    # 存放該資料夾底下音檔用
    audio_files = []
    for path_1 in tqdm(glob.glob(os.path.join(path, "*.flac"))): 
        audio_files.append(path_1)
    #=========================================================================================
    # 取 1 / 3 個音檔做合成
    count = 0
    while(count < (len(audio_files)/3)):
        print(str(count)+"/"+str(int(len(audio_files)/3)))
        # 隨機挑選四個音檔做合成
        random_audio_file1 = random.choice(audio_files)
        random_audio_file2 = random.choice(audio_files)
        random_audio_file3 = random.choice(audio_files)
        random_audio_file4 = random.choice(audio_files)
        #=========================================================================================
        # 音檔讀取
        audio1, fs = sf.read(random_audio_file1)
        audio2, fs = sf.read(random_audio_file2)
        audio3, fs = sf.read(random_audio_file3)
        audio4, fs = sf.read(random_audio_file4)
        
        #=========================================================================================
        # 創建房間
        left_wall = -15
        right_wall = 15
        top_wall = 15
        bottom_wall = -15
        absorption = np.random.uniform(low=0.1, high=0.99)

        corners = np.array([[left_wall, bottom_wall], [left_wall, top_wall],
                            [   right_wall, top_wall], [right_wall, bottom_wall]]).T

        room = pra.Room.from_corners(corners,
                fs=16000,
                max_order=10,
                absorption=absorption)
        #=========================================================================================
        # 將音檔長度弄成一致的
        length1 = len(audio1)
        length2 = len(audio2)
        length3 = len(audio3)
        length4 = len(audio4)

        max_length = max(length1, length2, length3, length4)
        
        if length1 < max_length:
            while len(audio1) < max_length:
                audio1 = np.concatenate((audio1, audio1[:max_length - len(audio1)]))
        
        if length2 < max_length:
            while len(audio2) < max_length:
                audio2 = np.concatenate((audio2, audio2[:max_length - len(audio2)]))
        
        if length3 < max_length:
            while len(audio3) < max_length:
                audio3 = np.concatenate((audio3, audio3[:max_length - len(audio3)]))

        if length4 < max_length:
            while len(audio4) < max_length:
                audio4 = np.concatenate((audio4, audio4[:max_length - len(audio4)]))
        #=========================================================================================
        # 音檔輸入
        # 第一象限
        room.add_source([5, 5],   signal=audio1)
        # 第二象限
        room.add_source([5, -5],  signal=audio2)
        # 第三象限
        room.add_source([-5, -5], signal=audio3)
        # 第四象限
        room.add_source([-5, 5],  signal=audio4)
        #=========================================================================================
        # 創建麥克風
        R = pra.circular_2D_array(center=[0., 0.], M=4, phi0=0, radius=.03231)
        room.add_microphone_array(pra.MicrophoneArray(R, 16000))
    
        room.simulate()
        #=========================================================================================
        save_dir = f"/workspace/test/LibriSpeech/mix/" + filename + "/" + filename + "_mix_" + str(count) + ".flac"
        # 存檔的位置
        room.mic_array.to_wav(
            save_dir,
            norm=True,
            bitdepth=np.int16,
        )
        
        count+=1
        #=========================================================================================
"""
        #=========================================================================================
        # 評量方式
        rt60 = room.measure_rt60()
        print("The measured RT60 is {}".format(rt60[1, 0]))
        #=========================================================================================
        # 畫出房間的圖
        fig, ax = room.plot()
        plt.grid()
        plt.show()
"""