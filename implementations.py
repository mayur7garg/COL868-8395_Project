import faiss
import numpy as np

def Flat(data, metric = faiss.METRIC_L2):
    d = data.shape[1]
    index = faiss.IndexFlat(d, metric)
    index.add(data)
    return index

def StandardPQ(data, num_subvectors, nbits = 8, metric = faiss.METRIC_L2):
    """
    Standard Product Quantization baseline without rotation or residuals.
    """
    d = data.shape[1]
    index = faiss.IndexPQ(d, num_subvectors, nbits, metric)
    
    if not index.is_trained:
        index.train(data)
    index.add(data)
    
    return index

def OPQ(data, num_subvectors, nbits = 8):
    """
    Implements OPQ using FAISS in the following steps:
    1. Define the OPQ rotation matrix and the underlying PQ index
    2. Combine them using IndexPreTransform to ensure the rotation is applied during training, adding, and searching
    3. Training and add vectors to the index
    
    Args:
        data: np.ndarray (float32) of shape (N, D)
        num_subvectors: Number of sub-quantizers (M)
        nbits: Bits per sub-vector (default 8, which is 256 centroids)
    """
    d = data.shape[1]
    
    opq_matrix = faiss.OPQMatrix(d, num_subvectors)
    index_pq = faiss.IndexPQ(d, num_subvectors, nbits)
    index = faiss.IndexPreTransform(opq_matrix, index_pq)
    
    index.train(data)
    index.add(data)
    
    return index

def AQ(data, num_codebooks, nbits = 8):
    """
    Implements Additive Quantization (Residual Quantization) using FAISS.
    
    Args:
        data: np.ndarray (float32) of shape (N, D)
        num_codebooks: Number of additive stages (M)
        nbits: Bits per stage (usually 8)
    """
    d = data.shape[1]
    
    index = faiss.IndexResidualQuantizer(d, num_codebooks, nbits)
    
    index.train(data)
    index.add(data)
    
    return index