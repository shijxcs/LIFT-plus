from collections import defaultdict
import torch
from torch.utils.data.sampler import Sampler


def shuffle(a):
    a = torch.tensor(a)
    perm = torch.randperm(len(a))
    return a[perm].tolist()


class DownSampler(Sampler):
    def __init__(self, labels, n_max=100):
        self.cls_idx_dict = defaultdict(list)
        for i, y in enumerate(labels):
            self.cls_idx_dict[y].append(i)
        
        self.n_max = n_max
        self.cls_num_list = [min(n_max, len(cls_idx)) for cls_idx in self.cls_idx_dict.values()]
        self.num_samples = sum(self.cls_num_list)

    def __iter__(self):
        sampled_idx = []
        for cls_num, cls_idx in zip(self.cls_num_list, self.cls_idx_dict.values()):
            idx = shuffle(cls_idx)[:cls_num]
            sampled_idx.extend(idx)
        sampled_idx = shuffle(sampled_idx)
        
        for i in range(self.num_samples):
            yield sampled_idx[i]

    def __len__(self):
        return self.num_samples
