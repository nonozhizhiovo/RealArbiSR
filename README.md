# Learning Dual-Level Implicit Representation for Real-World Scale Arbitrary Super-Resolution

## 1. RealArbiSR Dataset Preparation
Dataset is available at [RealArbiSRdataset - Google Drive](). 

Arrange dataset into the path like `load/Train/...` and `load/Test/...`

## 2. DDIR Code

### Train
`python train_realliif_deform.py --gpu [GPU] --config [CONFIG_NAME]`

### Test on Pretrained Models
The pretrained models can be downloaded from the google drive links below:

[EDSR-DDIR](https://drive.google.com/file/d/1idnTUqSkQzA3f1BPBuHPeOUCe-XQyd7o/view?usp=drive_link)

[RDN-DDIR](https://drive.google.com/file/d/1AJGnAyAq424RPZnUSQJJhgz3KiKSBTfn/view?usp=drive_link)

To test at all scale factors:

`bash ./scripts/test-realsrarbi-deform.sh [MODEL_PATH] [GPU]`

## Citation

