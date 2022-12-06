# Blood-Cell-Detection
 Detect blood cell such as RBC, WBC and platelets using BCCD Datasets.

<br>

## Environment

Dataset: Train Dataset 584, Test Dataset 216

<br>

## Preprocess Data

|                  训练样本的 X 均值和 Y 均值                  |                  训练样本的 X 均值和 Y 均值                  |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="README/image-20221205220427768.png" alt="image-20221205220427768" style="zoom: 80%;" /> | <img src="README/image-20221205220438194.png" alt="image-20221205220438194" style="zoom:80%;" /> |

<br>

| 训练样本的血液细胞计数                                       | 测试样本的血液细胞计数                                       |
| :------------------------------------------------------------: | :------------------------------------------------------------: |
| ![image-20221206010219196](README/image-20221206010219196.png) | ![image-20221206010229053](README/image-20221206010229053.png) |

<br>

## Algorithm Flowchart

<img src="README/image-20221205185301725.png" style="align-center">

<h4 align="center">Fig 1: Algorithm flowchart based on K-Means with initialized centroids using mean values</h4>

<br>

## Result

#### Blood Cell Detection Result using K-Means

|             基于 K-Means 聚类方法的血液细胞检测              |
| :----------------------------------------------------------: |
| <img src="README/image-20221205231449638.png" alt="image-20221205231449638" style="zoom:50%;" /> |

<br>
|        Visualize K-Means Clustering in Train Dataset         |         Visualize K-Means Clustering in Test Dataset         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20221205215645392](README/image-20221205215645392.png) | ![image-20221205215640029](README/image-20221205215640029.png) |

#### <br>Blood Cell Detection Result using SVM

| 模型类型                           |               基于 SVM 分类方法的血液细胞检测                |
| :----------------------------------: | :----------------------------------------------------------: |
| **基于自定义 SMO 的 SVM (RBF 核)** | <img src="README/image-20221205224450598.png" alt="image-20221205224450598" style="zoom: 67%;" /> |
| **基于 sklearn 库的 SVM (RBF 核)** | ![image-20221206011337981](README/image-20221206011337981.png) |

<br>

| 可视化训练样本的 SVM-RBF 分类                                | 可视化测试样本的 SVM-RBF 分类                                |
| :------------------------------------------------------------: | :------------------------------------------------------------: |
| ![image-20221206151226678](README/image-20221206151226678.png) | ![image-20221206161816230](README/image-20221206161816230.png) |

<br>

#### Blood Cell Detection Result using YOLOv6


| ![image-20221205180154713](README/image-20221205180154713.png) | ![image-20221205180201776](README/image-20221205180201776.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20221205180212027](README/image-20221205180212027.png) | ![image-20221205180220039](README/image-20221205180220039.png) |
| ![image-20221205180229426](README/image-20221205180229426.png) | ![image-20221205180241910](README/image-20221205180241910.png) |
| ![image-20221205180358890](README/image-20221205180358890.png) | ![image-20221205180311787](README/image-20221205180311787.png) |

<br>

| 基于 YOLOv6 分类方法的血液细胞检测                           |
| :------------------------------------------------------------: |
| <img src="README/image-20221205234541835.png" alt="image-20221205234541835" style="zoom:80%;" /> |
| ![image-20221205234607331](README/image-20221205234607331.png) |





