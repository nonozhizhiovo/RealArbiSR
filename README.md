# Learning Dual-Level Implicit Representation for Real-World Scale Arbitrary Super-Resolution

## 1. RealArbiSR Dataset Preparation
Dataset is available at [RealArbiSRdataset - Google Drive](https://drive.google.com/file/d/1RNb5Q5zI2vNPbw1u9hDVkZ4Jx1NIBVBZ/view?usp=drive_link). 

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

This code is built on [LIIF](https://github.com/yinboc/liif) and . We thank the authors for sharing their codes and extracted features.
