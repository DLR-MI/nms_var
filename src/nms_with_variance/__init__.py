import torch
from . import details

def nms_with_variance(boxes, scores, overlap, top_k):
    return details.nms_with_variance(boxes, scores, overlap, top_k)