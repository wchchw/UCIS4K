# Expose Camouflage in the Water: Underwater Camouflaged Instance Segmentation and Dataset

Source code and dataset for our paper “**[Expose Camouflage in the Water: Underwater Camouflaged Instance Segmentation and Dataset]()**” by Chuhong Wang,  Hua Li, Huazhong Liu, Xiongxin Tang, and Sam Kwong. 



## UCIS4K dataset
**UCIS4K** is the first dataset for the underwater camouflaged instance segmentation task. It comprises 3,953 images with instance-level annotations, featuring a diverse array of marine organisms, including fish, shrimp, crabs, and seahorses, across various camouflage scenarios.

Google drive: [UCIS4K](). **UCIS4K is free for academic research, not for any commercial purposes**.
![](USIS4K.png)<br>
**Note**: UCIS4K dataset includes camouflaged objects with similar colors and textures to the background,
blurred contours, small sizes, multiple objects, occlusion, complex contours, transparency, and underwater scenes with light and shadow effects.
## UCIS4K dataset Examples
Uncamouflaged objects appear unclear due to motion or backlighting in Fig.(a), Color camouflaged objects in Fig.(b), Texture camouflaged objects in Fig.(c), Edge blur camouflaged objects in Fig. (d).
![](category1.png)

## Benchmark
We compare the results with other instance segmentation methods on UCIS4K and UIIS datasets. From top to bottom: the original image is followed by ground truth and results of **[OSFormer](https://github.com/PJLallen/OSFormer)**, **CE-OST**, **DCNet**, **[WaterMask](https://github.com/LiamLian0727/WaterMask)**, **Mask2Former**, **SAM+bbox** and our **UCIS-SAM**. Each camouflaged instance is represented by a unique color. The first 4 columns are from our UCIS4K dataset, and the last 3 columns are from the UIIS dataset.
![](Compare.png)

## Bibliography entry
If you think our work is helpful, please cite
```
@article{hong2025usis4k,
  title={Expose Camouflage in the Water: Underwater Camouflaged Instance Segmentation and Dataset},
  journal={},
  year={2025}
}
```

## Acknowledgement
We thank the developer of [OSFormer](https://github.com/PJLallen/OSFormer), [WaterMask](https://github.com/LiamLian0727/WaterMask), and [USIS-SAM]() for providing their open-source code, which greatly facilitated our benchmark evaluations.

## Note to active participants

**UCIS-SAM aims to jointly address two key challenges in accurately segmenting objects that blend closely with their surroundings. We hope UCIS4K can facilitate research within the fields of marine computer vision and marine robotics.** 
