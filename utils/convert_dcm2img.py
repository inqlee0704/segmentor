# convert_dcm2img.py
# python dcm2img.py
# ----------------------------------------------------
# Use ProjSubjList to check if zunu_vida-ct.img exists
# if not, it will generate one. 
# ----------------------------------------------------
# 20211220, In Kyu Lee
# ----------------------------------------------------

import os
import pandas as pd
from DCM2IMG import DCMtoVidaCT
import sys
from tqdm.auto import tqdm

def convert_dcm2img():
    ProjSubjList_path = str(sys.argv[1]) # '/data4/inqlee0704/ENV18PM_ProjSubjList_cleaned.in'
    df = pd.read_csv(ProjSubjList_path,sep='\t')
    missing_path = []
    missing_dcm = []
    pbar = tqdm(range(len(df)))
    for i in pbar:
        case_path = df.loc[i,'ImgDir']
        if os.path.exists(case_path):
            if not os.path.exists(os.path.join(case_path,'zunu_vida-ct.img')):
                try:
                    DCMtoVidaCT(case_path)
                except:
                    missing_dcm.append(case_path)
        else:
            missing_path.append(case_path)
    print()
    print("***************************")
    print('Paths not found: ')
    print("***************************")
    print(missing_path)
    print()
    print("***************************")
    print('DCM not found: ')
    print("***************************")
    print(missing_dcm)


if __name__ == "__main__":
    convert_dcm2img()

