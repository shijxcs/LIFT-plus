# LIFT+ with different learnable parameters by changing the bottleneck dimension

python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ v.adaptformer False tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 1 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 2 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 4 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 8 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 16 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 32 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 64 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 128 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 256 tte True

python main.py -d places_lt -b clip_vit_b16 -m lift+ v.adaptformer False tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 1 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 2 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 4 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 8 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 16 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 32 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 64 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 128 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ v.adaptformer_dim 256 tte True

python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 v.adaptformer False tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 v.adaptformer_dim 64 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 v.adaptformer_dim 128 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 v.adaptformer_dim 192 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 v.adaptformer_dim 256 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 v.adaptformer_dim 320 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 v.adaptformer_dim 384 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 v.adaptformer_dim 448 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 v.adaptformer_dim 512 tte True