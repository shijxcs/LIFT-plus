import os
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset


class LT_Dataset(Dataset):
    split_txt = {"train": None, "val": None, "test": None}
    
    def __init__(self, root, split="train", transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        with open(self.split_txt[split]) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
    
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list