# COL868-8395 Project

This project implements OPQ (Optimized Product Quantization) and AQ (Additive Quantization) for multiple embedding datasets and calculates the relevant metrics. We also vary the hyperparameters of OPQ and AQ to understand how the metrics change.

Refer to [Project Proposal](<COL868 Project Proposal.pdf>) for more info.

To run the experiments, follow the steps below:
1. Download the relevant datasets. [Section: Embedding datasets](#embedding-datasets)
2. Install the dependencies. [Section: Dependencies](#dependencies)
3. Prepare the standardized version of the datasets and then run the experiments to calculate the metrics. [Section: Execution](#execution)

## Embedding datasets

1. CIFAR100 (CLIP VIT32)
    - Download the dataset from [OSF - CIFAR100-clip_vit32_b.npy](https://osf.io/cwyx8/files/x2uhw) and save it in `data/CIFAR100` folder.
    - Final disk size: 176MB
2. GIST1M
    - Download the `gist.tar.gz` file from [HF - fzliu/gist1m](https://huggingface.co/datasets/fzliu/gist1m/tree/main).
    - Extract the downloaded dataset and save the files in `data/GIST1M` folder.
    - Final disk size: 5.37GB
3. ImageNetMini1000 (CLIP VIT32)
    - Download the dataset from [OSF - DatasetImageNetMini1000-clip_vit32_b.npy](https://osf.io/cwyx8/files/28bgh) and save it in `data/ImageNetMini1000` folder.
    - Final disk size: 113MB
4. SIFT1M
    - Download the dataset from [figshare - sift.tar.gz](https://figshare.com/articles/dataset/sift_data/7428974?file=13755344).
    - Extract the downloaded dataset and save the files in `data/SIFT1M` folder.
    - Final disk size: 550MB

The structure of the `data` folder should look like this once all datasets have been downloaded as mentioned above:
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

## Dependencies
This project was built using Python 3.13. Dependencies are listed in `pyproject.toml`. Using that, a `requirements.txt` file also has been created.

Dependencies can be installed with `uv` via 
```sh 
uv sync
```

Or, if using `pip`, the dependencies can be installed via
```sh
pip install -r requirements.txt
```

## Execution

### 1. Prepare the standardized version of the datasets
In `prepare_datasets.py`, only the relevant code for `CIFAR100` has been left uncommented. To include the other datasets, uncomment the relevant lines in that file.

The standardized versions of the datasets can be prepared using

```sh
python prepare_datasets.py
```

This creates 3 `.npy` files for each dataset in `processed_data` folder:
- `data_vectors.npy`: Contains the dataset to instantiate the index.
- `query_vectors.npy`: Query vectors to compute the metrics for.
- `groundtruth.npy`: For each query vector, this contains the IDs of the most relevant data vectors in order of relevance.

### 2. Run the experiments
In `experiments.py`, only the experiments for `CIFAR100` and some hyperparameter combinations of `M` and `nbits` have been left uncommented. To include the other datasets or hyperparameter combinations, uncomment the relevant lines in that file.

The experiments can be run using
```sh
python experiments.py
```

This will create a CSV report in the `reports` folder with the timestamp containing all the metrics for each experiment run.

## Suggested environment
RAM usage especially when implementing OPQ and AQ for `GIST1M` is high. Linux machine with 64 to 128 GBs of RAM is recommended to run all hyperparameter combinations for all datasets.

## Relevant Papers
- [Optimized Product Quantization for Approximate Nearest Neighbor Search](https://openaccess.thecvf.com/content_cvpr_2013/papers/Ge_Optimized_Product_Quantization_2013_CVPR_paper.pdf)
- [Additive Quantization for Extreme Vector Compression](https://openaccess.thecvf.com/content_cvpr_2014/papers/Babenko_Additive_Quantization_for_2014_CVPR_paper.pdf)