from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def read_fvecs(filename):
    """
    Reads .fvecs files (binary float32 vectors with int32 dimension headers).
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"File {filename} not found.")
    
    a = np.fromfile(filename, dtype = 'int32')

    if a.size == 0:
        return np.zeros((0, 0), dtype = 'float32')
    
    dim = a[0]

    return a.reshape(-1, dim + 1)[:, 1:].copy().view('float32')

def read_ivecs(filename):
    """
    Reads .ivecs files (binary int32 vectors with int32 dimension headers).
    Used typically for ground truth indices.
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"File {filename} not found.")
    
    a = np.fromfile(filename, dtype = 'int32')

    if a.size == 0:
        return np.zeros((0, 0), dtype = 'int32')
    
    dim = a[0]

    return a.reshape(-1, dim + 1)[:, 1:].copy()

def prepare_groundtruth_from_classes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    count: int,
    output_dir: Path
):
    '''
    For each vector in `X_test`, find the `count` most similar vectors in `X_train` with the same class and save the results
    '''
    groundtruth = []

    for i in range(X_test.shape[0]):
        cls_id = y_test[i]
        same_cls_idx = np.where(y_train == cls_id)[0]
        same_cls_sims = cosine_similarity(X_test[i].reshape(1, -1), X_train[same_cls_idx])[0]
        top_matches = np.argsort(same_cls_sims)[::-1][:count]
    
        groundtruth.append(same_cls_idx[top_matches])
    
    groundtruth = np.array(groundtruth)

    with output_dir.joinpath("groundtruth.npy").open("wb") as f:
        np.save(f, groundtruth)
        print(f"Saved groundtruth.npy of shape {groundtruth.shape} to {output_dir}")
    
    with output_dir.joinpath("data_vectors.npy").open("wb") as f:
        np.save(f, X_train)
        print(f"Saved data_vectors.npy of shape {X_train.shape} to {output_dir}")
    
    with output_dir.joinpath("query_vectors.npy").open("wb") as f:
        np.save(f, X_test)
        print(f"Saved query_vectors.npy of shape {X_test.shape} to {output_dir}")

def process_vec_data(
    data_dir: Path,
    dataset_prefix: str,
    output_dir: Path
):
    groundtruth = read_ivecs(data_dir.joinpath(f"{dataset_prefix}_groundtruth.ivecs"))
    data_vectors = read_fvecs(data_dir.joinpath(f"{dataset_prefix}_base.fvecs"))
    query_vectors = read_fvecs(data_dir.joinpath(f"{dataset_prefix}_query.fvecs"))

    with output_dir.joinpath("groundtruth.npy").open("wb") as f:
        np.save(f, groundtruth)
        print(f"Saved groundtruth.npy of shape {groundtruth.shape} to {output_dir}")
    
    with output_dir.joinpath("data_vectors.npy").open("wb") as f:
        np.save(f, data_vectors)
        print(f"Saved data_vectors.npy of shape {data_vectors.shape} to {output_dir}")
    
    with output_dir.joinpath("query_vectors.npy").open("wb") as f:
        np.save(f, query_vectors)
        print(f"Saved query_vectors.npy of shape {query_vectors.shape} to {output_dir}")

if __name__ == "__main__":
    OUTPUT_DIR = Path("processed_data", "CIFAR100")
    OUTPUT_DIR.mkdir(exist_ok = True, parents = True)
    raw_data = np.load(
        Path("data", "CIFAR100", "CIFAR100-clip_vit32_b.npy"),
        allow_pickle = True
    ).item()
    prepare_groundtruth_from_classes(
        raw_data['X_train'],
        raw_data['y_train'],
        raw_data['X_test'],
        raw_data['y_test'],
        100,
        OUTPUT_DIR
    )

    # OUTPUT_DIR = Path("processed_data", "ImageNetMini1000")
    # OUTPUT_DIR.mkdir(exist_ok = True, parents = True)
    # raw_data = np.load(
    #     Path("data", "ImageNetMini1000", "DatasetImageNetMini1000-clip_vit32_b.npy"),
    #     allow_pickle = True
    # ).item()
    # prepare_groundtruth_from_classes(
    #     raw_data['X_train'],
    #     raw_data['y_train'],
    #     raw_data['X_test'],
    #     raw_data['y_test'],
    #     5,
    #     OUTPUT_DIR
    # )

    # OUTPUT_DIR = Path("processed_data", "SIFT1M")
    # OUTPUT_DIR.mkdir(exist_ok = True, parents = True)
    # process_vec_data(
    #     Path("data", "SIFT1M"),
    #     "sift",
    #     OUTPUT_DIR
    # )

    # OUTPUT_DIR = Path("processed_data", "GIST1M")
    # OUTPUT_DIR.mkdir(exist_ok = True, parents = True)
    # process_vec_data(
    #     Path("data", "GIST1M"),
    #     "gist",
    #     OUTPUT_DIR
    # )