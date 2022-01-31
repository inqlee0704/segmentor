# segmentor
Segmentor is a deep learning segmentation engine.


# Inference

```python
python run_inference --mask=airway
                     --model=ZUNet
                     --subj_path=25 
                     --parameter_path=RESULTS/airway/airway_ZUNet.pth
```
- mask: airway, lobe, lung, vessels
- model: UNet, ZUNet
- subj_path: Subject folder which has a dicom folder
- parameter_path: path to the *.pth


# Train
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

## Start training
```python
python main.py
```
- mask: airway, vessels, lung, lobe
- model: UNet, ZUNet
- debug: True, False
- save: True, False
- lr: learning rate
- train_bs: train batch size
- valid_bs: valid batch size
- epochs: train epoch
- n_case: number of cases to use

