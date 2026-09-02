import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def encode_texts(model, texts):
    """
    Generate embeddings for a list of texts.
    """
    return model.encode(
        texts,
        show_progress_bar=False,
        batch_size=32
    )


def build_similarity_matrix(
    job_embeddings,
    resume_embeddings
):
    """
    Calculate cosine similarity between
    every job and every resume.
    """
    return cosine_similarity(
        job_embeddings,
        resume_embeddings
    )


def retrieve_top_k(
    similarity_matrix,
    job_idx,
    top_k=500
):
    """
    Retrieve the top-K resume indices
    for a given job.
    """

    scores = similarity_matrix[job_idx]

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    top_scores = scores[top_indices]

    return top_indices, top_scores
