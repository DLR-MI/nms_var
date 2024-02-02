import os
import torch
import time
import numpy as np
from nms_var import nms as nms_var
from torchvision.ops import nms
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, save_image
import matplotlib.pyplot as plt


def check_set(set_a: np.ndarray, set_b: np.ndarray):
    """This function checks two sets A and B for similarity and prints the result.
    If the two elements are the same, the difference count is decreased and the same count increased.
    """
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
        result: Tuple[np.ndarray, np.ndarray, np.ndarray]
            - Kept indices from custom computation
            - Kept indices from Torchvision NMS
            - The variance over scores and bounding box per kept index
    """
    # run a few times for warm-up
    num_runs = 100
    t_torch = np.zeros(num_runs)
    t_custom = np.zeros(num_runs)
    for i in range(num_runs):
        t0 = time.time()
        _, _ = nms_var(boxes, scores, overlap=overlap, top_k=top_k)
        t1 = time.time()
        _ = nms(boxes, scores, overlap)
        t2 = time.time()
        t_custom[i] = t1 - t0
        t_torch[i] = t2 - t1
    print("Custom approach took: {:.3f} ms".format(np.mean(t_custom) * 1000))
    print("Torchvision approach took: {:.3f} ms".format(np.mean(t_torch) * 1000))

    keep_custom, var_per_kept_index = nms_var(boxes, scores, overlap=overlap, top_k=top_k)
    keep_torch = nms(boxes, scores, overlap)
    return keep_custom.cpu().numpy(), keep_torch.cpu().numpy(), var_per_kept_index.cpu().numpy()


def main():
    dir_name = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(dir_name, 'dets')

    device = torch.device('cuda:0')

    boxes = torch.from_numpy(np.load(os.path.join(test_dir, 'boxes.npy'))).float().to(device=device)
    scores = torch.from_numpy(np.load(os.path.join(test_dir, 'scores.npy'))).float().to(device=device)

    keep_custom, keep_torch, var = run_nms(boxes, scores)

    check_set(keep_torch, keep_custom)


if __name__ == "__main__":
    main()