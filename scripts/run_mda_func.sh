# LIFT+ with different scheduling functions

python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ mda_func min tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ mda_func convex tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ mda_func linear tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ mda_func concave tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ mda_func max tte True

python main.py -d places_lt -b clip_vit_b16 -m lift+ mda_func min tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ mda_func convex tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ mda_func linear tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ mda_func concave tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ mda_func max tte True

python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 mda_func min tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 mda_func convex tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 mda_func linear tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 mda_func concave tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 mda_func max tte True