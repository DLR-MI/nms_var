import torch
from . import details


def nms_with_variance(boxes, scores, overlap, top_k):
    """Compute non-maximum suppression with variance.
    The variance of all overlapping bounding boxes is computed and returned together with the kept bounding box.
    Computing the variance maintains the uncertainty corresponding to a selected bounding box.

    Parameters
    ----------
    boxes : torch.Tensor
        Tensor of size (N, 4) containing N bounding boxes with format (x1, x2, y1, y2).
    scores : torch.Tensor
        Tensor of size (N) containing the corresponding scores for the bounding boxes.
    overlap : float
        Overlap threshold for IoU computation.
    top_k : int
        Maximum number of detections to keep.

    Returns
    -------
    keep_custom : tuple
        A tuple consisting of:

            - Torch tensor of size (M) containing the indices of the M kept bounding boxes.
            - Tensor of size (M,4) containing the variance for each kept item with respect to all overlapping bounding boxes.
    """
    return details.nms_with_variance(boxes, scores, overlap, top_k)