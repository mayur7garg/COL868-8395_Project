import time
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

from implementations import OPQ, AQ, StandardPQ, Flat

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok = True, parents = True)

K = [1, 5, 10, 50, 100]

def evaluate(
    index,
    data_vecs,
    query_vecs,
    ground_truth,
    K
):
    K = [k for k in K if k <= ground_truth.shape[1]]
    results = {}

    start = time.time()
    distances, indices = index.search(query_vecs, max(K))
    end = time.time()
    results[f'Query Time (s)'] = end - start

    for k in K:
        recall = 0
        precision = 0
        average_precision = 0

        for i in range(query_vecs.shape[0]):
            retrieved = indices[i, :k]
            relevant = ground_truth[i, :k]
            is_relevant = np.isin(retrieved, relevant)
            
            common = is_relevant.sum()
            recall += common / ground_truth.shape[1]
            precision += common / k

            ap = 0
            hits = 0
            for j in range(k):
                if is_relevant[j]:
                    hits += 1
                    ap += hits / (j + 1)
            average_precision += (ap / k)
        
        recall /= query_vecs.shape[0]
        precision /= query_vecs.shape[0]
        average_precision /= query_vecs.shape[0]
        
        results[f'Recall@{k}'] = recall
        results[f'Precision@{k}'] = precision
        results[f'mAP@{k}'] = average_precision
    
    reconstructed_vectors = index.reconstruct_n(0, index.ntotal)
    approximation_error = np.linalg.norm(reconstructed_vectors - data_vecs, axis = 1).mean()
    results['Approximation Error'] = approximation_error

    with tempfile.NamedTemporaryFile(delete = False) as tmp:
        temp_path = tmp.name
    try:
        faiss.write_index(index, temp_path)
        index_size = Path(temp_path).stat().st_size
        results['Disk Size (KB)'] = index_size / 1024
    except:
        temp_path = "temp_index.faiss"
        faiss.write_index(index, temp_path)
        index_size = Path(temp_path).stat().st_size
        results['Disk Size (KB)'] = index_size / 1024
    finally:
        Path(temp_path).unlink(missing_ok = True)

    return results

for dataset in [
    "CIFAR100",
    # "ImageNetMini1000",
    # "SIFT1M",
    # "GIST1M"
]:
    data_path = Path("processed_data", dataset)
    data_vectors = np.load(data_path / "data_vectors.npy").astype('float32')
    query_vectors = np.load(data_path / "query_vectors.npy").astype('float32')
    groundtruth = np.load(data_path / "groundtruth.npy")

    # Flat
    start = time.time()
    index = Flat(data_vectors)
    end = time.time()

    results = evaluate(index, data_vectors, query_vectors, groundtruth, K)
    results['Build Time (s)'] = end - start
    results['Index'] = "Flat"
    report = pd.DataFrame([results.values()], columns = results.keys())
    print('Completed Flat')

    for M in [
        4, 8, 
        # 16, 32
    ]:
        for nbits in [
            4, 
            # 8
        ]:
            # StandardPQ
            start = time.time()
            index = StandardPQ(data_vectors, M, nbits)
            end = time.time()

            results = evaluate(index, data_vectors, query_vectors, groundtruth, K)
            results['Build Time (s)'] = end - start
            results['Index'] = f"StandardPQ(M = {M}, nbits = {nbits})"
            report = pd.concat([
                report,
                pd.DataFrame([results.values()], columns = results.keys())
            ])
            print(f"Completed {results['Index']}")

            # OPQ
            start = time.time()
            index = OPQ(data_vectors, M, nbits)
            end = time.time()

            results = evaluate(index, data_vectors, query_vectors, groundtruth, K)
            results['Build Time (s)'] = end - start
            results['Index'] = f"OPQ(num_subvectors = {M}, nbits = {nbits})"
            report = pd.concat([
                report,
                pd.DataFrame([results.values()], columns = results.keys())
            ])
            print(f"Completed {results['Index']}")

            # AQ
            start = time.time()
            index = AQ(data_vectors, M, nbits)
            end = time.time()

            results = evaluate(index, data_vectors, query_vectors, groundtruth, K)
            results['Build Time (s)'] = end - start
            results['Index'] = f"AQ(num_codebooks = {M}, nbits = {nbits})"
            report = pd.concat([
                report,
                pd.DataFrame([results.values()], columns = results.keys())
            ])
            print(f"Completed {results['Index']}")

    report.to_csv(REPORTS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{dataset}.csv", index = False)