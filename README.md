# segmentor
Segmentor is a deep learning segmentation engine.

## Data Preparation
0. Get VIDA_sheet.csv
- Download VIDA_sheet as a .csv from either google drive or one drive. 
- save as VIDA_sheet.csv

1. Get ProjSubjList.in
```bash
cd utils
python get_ProjSubjList.py /d/ENV18PM/VIDA_sheet_20220120.csv /d/ENV18PM/ENV18PM_ProjSubjList_20220120.in /d/ENV18PM/ImageData
```

2. Convert dicom to analyze
```bash
python convert_dcm2img.py /d/ENV18PM/ENV18PM_ProjSubjList_20220120.in
```