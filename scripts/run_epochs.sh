# LIFT with different training epochs

python main.py -d imagenet_lt -b clip_vit_b16 -m lift num_epochs 5 lr 0.02 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift num_epochs 10 lr 0.01 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift num_epochs 15 lr 0.0066 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift num_epochs 20 lr 0.005 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift num_epochs 25 lr 0.004 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift num_epochs 30 lr 0.0033 tte True

python main.py -d places_lt -b clip_vit_b16 -m lift num_epochs 5 lr 0.02 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift num_epochs 10 lr 0.01 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift num_epochs 15 lr 0.0066 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift num_epochs 20 lr 0.005 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift num_epochs 25 lr 0.004 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift num_epochs 30 lr 0.0033 tte True

python main.py -d inat2018 -b clip_vit_b16 -m lift num_epochs 5 lr 0.04 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift num_epochs 10 lr 0.02 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift num_epochs 15 lr 0.0133 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift num_epochs 20 lr 0.01 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift num_epochs 25 lr 0.008 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift num_epochs 30 lr 0.0066 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift num_epochs 35 lr 0.0057 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift num_epochs 40 lr 0.005 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift num_epochs 45 lr 0.0044 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift num_epochs 50 lr 0.004 tte True

# LIFT+ with different training epochs

python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ num_epochs 5 lr 0.02 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ num_epochs 10 lr 0.01 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ num_epochs 15 lr 0.0066 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ num_epochs 20 lr 0.005 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ num_epochs 25 lr 0.004 tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ num_epochs 30 lr 0.0033 tte True

python main.py -d places_lt -b clip_vit_b16 -m lift+ num_epochs 5 lr 0.02 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ num_epochs 10 lr 0.01 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ num_epochs 15 lr 0.0066 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ num_epochs 20 lr 0.005 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ num_epochs 25 lr 0.004 tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ num_epochs 30 lr 0.0033 tte True

python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 5 lr 0.06 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 10 lr 0.03 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 lr 0.02 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 20 lr 0.015 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 25 lr 0.012 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 30 lr 0.01 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 35 lr 0.0085 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 40 lr 0.0075 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 45 lr 0.0066 tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 50 lr 0.006 tte True