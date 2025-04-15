import json
from ._lt_data import LT_Dataset


class iNaturalist2018(LT_Dataset):
    split_txt = {
        "train": "./datasets/iNaturalist2018/train.txt",
        "test": "./datasets/iNaturalist2018/val.txt",
    }
    categories_json = "./datasets/iNaturalist2018/categories.json"
    category_method = "name"

    def __init__(self, root, split="train", transform=None):
        super().__init__(root, split, transform)

        id2cname, cname2lab = self.read_category_info()

        self.labels = []
        with open(self.split_txt[split]) as f:
            for line in f:
                name = id2cname[int(line.split()[1])]
                self.labels.append(cname2lab[name])

        self.classnames = self.get_classnames()
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)

    @classmethod
    def read_category_info(self):
        with open(self.categories_json, "rb") as file:
            category_info = json.load(file)
        
        id2cname = {}
        for id, info in enumerate(category_info):
            cname = info[self.category_method]
            id2cname[id] = cname

        cnames_unique = sorted(set(id2cname.values()))
        cname2lab = {c: i for i, c in enumerate(cnames_unique)}
        return id2cname, cname2lab

    @classmethod
    def get_classnames(self):
        id2cname, cname2lab = self.read_category_info()
        classnames = list(cname2lab.keys())
        return classnames


class iNaturalist2018_Kingdom(iNaturalist2018):
    category_method = "kingdom"

class iNaturalist2018_Phylum(iNaturalist2018):
    category_method = "phylum"

class iNaturalist2018_Class(iNaturalist2018):
    category_method = "class"

class iNaturalist2018_Order(iNaturalist2018):
    category_method = "order"

class iNaturalist2018_Family(iNaturalist2018):
    category_method = "family"

class iNaturalist2018_Genus(iNaturalist2018):
    category_method = "genus"

class iNaturalist2018_Species(iNaturalist2018):
    category_method = "name"
