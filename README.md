# COL868-8395_Project

## Embedding datasets

1. CIFAR100 (CLIP VIT32)
    - Download the dataset from [OSF - CIFAR100-clip_vit32_b.npy](https://osf.io/cwyx8/files/x2uhw) and save it in `data/CIFAR100` folder.
2. GIST1M
    - Download the `gist.tar.gz` file from [HF - fzliu/gist1m](https://huggingface.co/datasets/fzliu/gist1m/tree/main).
    - Extract the downloaded dataset and save the files in `data/GIST1M` folder.
3. ImageNetMini1000 (CLIP VIT32)
    - Download the dataset from [OSF - DatasetImageNetMini1000-clip_vit32_b.npy](https://osf.io/cwyx8/files/28bgh) and save it in `data/ImageNetMini1000` folder.
4. SIFT1M
    - Download the dataset from [figshare - sift.tar.gz](https://figshare.com/articles/dataset/sift_data/7428974?file=13755344).
    - Extract the downloaded dataset and save the files in `data/SIFT1M` folder.

The structure of the `data` folder should look like this:
```
data/
├──CIFAR100
    ├──CIFAR100-clip_vit32_b.npy
├──GIST1M
    ├──gist_base.fvecs
    ├──gist_groundtruth.ivecs
    ├──gist_learn.fvecs
    ├──gist_query.fvecs
├──ImageNetMini1000
    ├──DatasetImageNetMini1000-clip_vit32_b.npy
├──SIFT1M
    ├──sift_base.fvecs
    ├──sift_groundtruth.ivecs
    ├──sift_learn.fvecs
    ├──sift_query.fvecs
```

## Relevant Papers
- [Optimized Product Quantization for Approximate Nearest Neighbor Search](https://openaccess.thecvf.com/content_cvpr_2013/papers/Ge_Optimized_Product_Quantization_2013_CVPR_paper.pdf)
- [Additive Quantization for Extreme Vector Compression](https://openaccess.thecvf.com/content_cvpr_2014/papers/Babenko_Additive_Quantization_for_2014_CVPR_paper.pdf)