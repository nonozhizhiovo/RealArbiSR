# Learning Dual-Level Implicit Representation for Real-World Scale Arbitrary Super-Resolution

## 1. RealArbiSR Dataset Preparation

### Version 2
In version 2, we further refine the dataset quality and increase the size of x1.7/x2.3/x2.7/x3.3/x3.7 testset from 83 scenes to 100 scenes. 

Dataset Version 2 is available at [RealArbiSRdatasetv2 - Google Drive](https://drive.google.com/file/d/1bll8UDYU9c318XsgPcdq5xwsSzdzyVJ9/view?usp=drive_link) 

The pretrained models and the PSNR results of RealArbiSR dataset Version 2 are listed below: 

[EDSR-DDIR-v2](https://drive.google.com/file/d/1src8POjvX4WolpCwOWhtpWQF7au6TUb9/view?usp=drive_link)

[RDN-DDIR-v2](https://drive.google.com/file/d/1JI7-_VquTF1fZAQ5oKw1k9E_NAlFWhfw/view?usp=drive_link)

 |Methods    |PSNR|      x1.5      |      x2.0      |      x2.5      |      x3.0      |      x3.5      |      x4.0      |  
 |-----------|----|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
 |Bicubic   |    |    34.87    |    31.61    |    29.81    |   28.56    |    27.64    |    27.00    |           
 |EDSR-LIIF   |    |    36.55    |    33.63    |    31.76    |    30.49    |    29.47    |    28.80   |  
 |EDSR-LTE  |    |    36.56    |    33.63    |    31.75    |    30.48    |    29.52    |    28.84    |  
 |EDSR-CiaoSR  |    |    36.67    |    33.84    |    32.01    |    30.74    |    29.75    |    29.01    |
 |EDSR-DDIR|    |    36.91    |    34.09    |    32.20    |    30.94    |    29.94    |    29.19    |  
 |RDN-LIIF   |    |    36.64    |    33.84    |    31.94    |    30.69    |    29.69    |    29.00   |  
 |RDN-LTE  |    |    36.60    |    33.80    |    31.95    |    30.67    |    29.70    |    29.00    |  
 |RDN-CiaoSR  |    |    36.85    |    34.07    |    32.18    |    30.87    |    29.86    |    29.10    |
 |RDN-DDIR|    |    37.04    |    34.28    |    32.35    |    31.05    |    30.04    |    29.26    | 
 
 |Methods    |PSNR|      x1.7      |      x2.3      |      x2.7      |      x3.3      |      x3.7      |  
 |-----------|----|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
 |Bicubic   |    |    31.31    |    28.54    |    27.51    |    26.42    |    25.83    |           
 |EDSR-LIIF   |    |    33.37    |    30.57    |    29.37    |    28.02    |    27.32    |  
 |EDSR-LTE  |    |    33.47    |    30.64    |    29.40    |    28.02    |    27.30    |    
 |EDSR-CiaoSR  |    |    33.04    |    30.58    |    29.50    |    28.23    |    27.48    |
 |EDSR-DDIR|    |    33.71    |    30.97    |    29.76    |    28.37    |    27.64    |    
 |RDN-LIIF   |    |    33.49    |    30.71    |    29.51    |    28.16    |    27.43    |  
 |RDN-LTE  |    |    33.54    |    30.83    |    29.61    |    28.23    |    27.51    |  
 |RDN-CiaoSR  |    |    33.16    |    30.81    |    29.74    |    28.44    |    27.69    |
 |RDN-DDIR|    |    33.77    |    31.06    |    29.85    |    28.46    |    27.72    | 

### Version 1 (used in the original paper)
Dataset is available at [RealArbiSRdataset - Google Drive](https://drive.google.com/file/d/1RNb5Q5zI2vNPbw1u9hDVkZ4Jx1NIBVBZ/view?usp=drive_link). 

Arrange dataset into the path like `load/Train/...` and `load/Test/...`

## 2. DDIR Code

### Train
`python train_realliif_deform.py --gpu [GPU] --config [CONFIG_NAME] --save_name [SAVE_NAME]`

### Test on Pretrained Models
The pretrained models (for Verision 1, used in the original paper) can be downloaded from the google drive links below:

[EDSR-DDIR](https://drive.google.com/file/d/1idnTUqSkQzA3f1BPBuHPeOUCe-XQyd7o/view?usp=drive_link)

[RDN-DDIR](https://drive.google.com/file/d/1AJGnAyAq424RPZnUSQJJhgz3KiKSBTfn/view?usp=drive_link)

To test at all scale factors:

`bash ./scripts/test-realsrarbi-deform.sh [MODEL_PATH] [GPU]`

## Citation
If you find this code useful in your work then please cite:

```
@article{li2024learning,
  title={Learning Dual-Level Deformable Implicit Representation for Real-World Scale Arbitrary Super-Resolution},
  author={Li, Zhiheng and Li, Muheng and Fan, Jixuan and Chen, Lei and Tang, Yansong and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:2403.10925},
  year={2024}
}
```

## Contact
Please contact Zhiheng Li @ lizhihan21@mails.tsinghua.edu.cn if any issue.

## Acknowledgements

This code is built on [LIIF](https://github.com/yinboc/liif). We thank the authors for sharing their codes. 
