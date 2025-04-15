from ._lt_data import LT_Dataset


class ImageNet_LT(LT_Dataset):
    split_txt = {
        "train": "./datasets/ImageNet_LT/train.txt",
        "val": "./datasets/ImageNet_LT/val.txt",
        "test": "./datasets/ImageNet_LT/test.txt",
    }
    classnames_txt = "./datasets/ImageNet_LT/classnames.txt"

    def __init__(self, root, split="train", transform=None):
        super().__init__(root, split, transform)

        self.classnames = self.read_classnames()

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        return classnames
