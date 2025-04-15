# LIFT+ with different classifiers

python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ classifier LinearClassifier tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ classifier L2NormClassifier tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ classifier CosineClassifier classifier_scale 15 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ classifier CosineClassifier classifier_scale 20 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ classifier CosineClassifier classifier_scale 25 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ classifier CosineClassifier classifier_scale 30 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ classifier CosineClassifier classifier_scale 35 tte True

python main.py -d places_lt -b clip_vit_b16 -m lift+ classifier LinearClassifier tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ classifier L2NormClassifier tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ classifier CosineClassifier classifier_scale 15 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ classifier CosineClassifier classifier_scale 20 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ classifier CosineClassifier classifier_scale 25 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ classifier CosineClassifier classifier_scale 30 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ classifier CosineClassifier classifier_scale 35 tte True

python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 classifier LinearClassifier tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 classifier L2NormClassifier tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 classifier CosineClassifier classifier_scale 15 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 classifier CosineClassifier classifier_scale 20 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 classifier CosineClassifier classifier_scale 25 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 classifier CosineClassifier classifier_scale 30 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 classifier CosineClassifier classifier_scale 35 tte True