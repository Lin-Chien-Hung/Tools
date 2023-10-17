import glob
import os
import subprocess
from tqdm import tqdm 

for path in tqdm(glob.glob(os.path.join("CN_AISHELL_1445", "*"))):
    
    basename = os.path.basename(path)
    print(basename)
    #if not os.path.isdir("CN_AISHELL_1445_48/"+basename):
    #    os.mkdir("CN_AISHELL_1445_48/"+basename)
    
    for path_1 in tqdm(glob.glob(os.path.join(path, "*.wav"))):
        
        basename_1 = os.path.basename(path_1)
        #print(basename_1)
        sox = " ".join(["ffmpeg", "-i", path_1, "-ar", "48000", "-ac", "1", "./CN_AISHELL_1445_48/" + basename + "/" + basename_1, ">/dev/null 2>/dev/null"])
        
        subprocess.run(sox, shell=True)