import torch
import time
import numpy as np
from nms_with_variance import nms_with_variance
from torchvision.ops import nms


def check_set(set_a, set_b):
    set_a_list = set_a.flatten().tolist()
    set_b_list = set_b.flatten().tolist()
    num_same = 0
    num_different = len(set_b_list)
    for a in set_a_list:
        for b in set_b_list:
            if a == b:
                num_same += 1
                num_different -= 1
                break
    print("Out of {} indices, {} {} equal, {} {} different"
          .format(len(set_b_list),
                  num_same, "is" if num_same == 1 else "are",
                  num_different, "is" if num_different == 1 else "are"))


def run_nms(boxes, scores, overlap=.7, top_k=200):
    """Run NMS using the standard Torchvision NMS and the custom NMS using variance.
    Computes the timing of both functions and if the output is equal.

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
    keep_custom : torch.Tensor

    """
    t0 = time.time()
    keep_custom, var_per_kept_index = nms_with_variance(boxes, scores, overlap=overlap, top_k=top_k)
    t1 = time.time()
    keep_torch = nms(boxes, scores, overlap)
    t2 = time.time()
    print("Custom approach took: {} ms".format((t1 - t0) * 1000))
    print("Torchvision approach took: {} ms".format((t2 - t1) * 1000))
    delta = torch.sum(keep_custom - keep_torch).cpu().numpy() == 0
    print("Equal? {}".format(delta))
    print(var_per_kept_index)
    return keep_custom.cpu().numpy(), keep_torch.cpu().numpy(), var_per_kept_index.cpu().numpy()


def main():
    device = torch.device('cuda:0')
    boxes = torch.from_numpy(np.load('test_data/boxes.npy')).float().to(device=device)
    scores = torch.from_numpy(np.load('test_data/scores.npy')).float().to(device=device)
    vars_xi = np.load('test_data/vars_xi.npy')
    i = np.load('test_data/i.npy')

    keep_custom, keep_torch, var = run_nms(boxes, scores)

    res = vars_xi - var[:vars_xi.shape[0], :]
    res *= res
    res = np.sqrt(res)
    print("RMS of variance cuda: {}".format(np.mean(res, axis=0)))

    check_set(i, keep_custom)


if __name__ == "__main__":
    main()