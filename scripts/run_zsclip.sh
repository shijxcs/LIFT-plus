# Performance of CLIP

python main.py -d imagenet_lt -b clip_vit_b16 -m zs
python main.py -d imagenet_lt -b clip_vit_b16 -m lp lr 0.03  # best lr for classifier fine-tuning (searched from fig. 10)

python main.py -d places_lt -b clip_vit_b16 -m zs
python main.py -d places_lt -b clip_vit_b16 -m lp lr 0.03  # best lr for classifier fine-tuning (searched from fig. 10)

python main.py -d inat2018_k -b clip_vit_b16 -m zs
python main.py -d inat2018_p -b clip_vit_b16 -m zs
python main.py -d inat2018_c -b clip_vit_b16 -m zs
python main.py -d inat2018_o -b clip_vit_b16 -m zs
python main.py -d inat2018_f -b clip_vit_b16 -m zs
python main.py -d inat2018_g -b clip_vit_b16 -m zs
python main.py -d inat2018_s -b clip_vit_b16 -m zs
python main.py -d inat2018 -b clip_vit_b16 -m zs