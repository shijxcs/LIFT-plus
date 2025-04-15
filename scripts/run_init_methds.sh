# LIFT+ with different classifier initialization methods

python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ classifier_init None tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ classifier_init linear_probing tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ classifier_init class_mean tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ classifier_init semantic tte True

python main.py -d places_lt -b clip_vit_b16 -m lift+ classifier_init None tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ classifier_init linear_probing tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ classifier_init class_mean tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ classifier_init semantic tte True