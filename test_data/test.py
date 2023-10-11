import torch
import time
import numpy as np
from nms_variance import nms_variance
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
    t0 = time.time()
    keep_custom, num_to_keep, parent_object_index, var = nms_variance(boxes, scores, overlap=overlap, top_k=top_k)
    t1 = time.time()
    keep_torch = nms(boxes, scores, overlap)
    t2 = time.time()
    print("Custom approach took: {} ms".format((t1 - t0) * 1000))
    print("Torchvision approach took: {} ms".format((t2 - t1) * 1000))
    delta = torch.sum(keep_custom[:num_to_keep.item()] - keep_torch).cpu().numpy() == 0
    print("Equal? {}".format(delta))
    return keep_custom[:num_to_keep.item()].cpu().numpy(), keep_torch.cpu().numpy(), var.numpy().reshape(num_to_keep.item(), 4)


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
    print("RMS of variance: ".format(np.mean(res, axis=0)))

    check_set(i, keep_custom)
    # np.save('test_data/keep.npy', keep.to('cpu').numpy())
    # np.save('test_data/parent_object_index.npy', parent_object_index.to('cpu').numpy())


if __name__ == "__main__":
    main()