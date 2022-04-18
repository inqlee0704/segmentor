# segmentor
Segmentor is a deep learning segmentation engine.


# Inference
Predict and save segmentation results.
### Requirements:
- model file: RESULTS/lobe/ZUNet.pth
- dicom: {subj_path}/dicom/
- in file: ProjSubjListDCM.in (if you want to run multiple cases) 

```python
# Infer one case (lobes)
python run_inference --subj_path=25 

# Infer a list of cases
python run_inference --in_file_path=TE_ProjSubjListDCM.in
```
#### Parameters:
- mask: airway, lobes, lung, vessels
- model: UNet, ZUNet
- subj_path: Subject folder which has a dicom folder
- parameter_path: path to the *.pth

#### Outputs:
- {model}_{mask}.img.gz
- {model}_{mask}.hdr.gz

# Test
Calculate segmentation accuracy using the mask from the 'run_inference.py' and ground truth.
### Requirements:
- predicted mask: *.img.gz & *.hdr
- label: *.img.gz & *.hdr

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

