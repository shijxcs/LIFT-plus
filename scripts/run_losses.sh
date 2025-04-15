# LIFT+ with different losses

python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ loss_type CE tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ loss_type Focal tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ loss_type LDAM tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ loss_type CB tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ loss_type GRW tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ loss_type LADE tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ loss_type LA tte True

python main.py -d places_lt -b clip_vit_b16 -m lift+ loss_type CE tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ loss_type Focal tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ loss_type LDAM tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ loss_type CB tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ loss_type GRW tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ loss_type LADE tte True
python main.py -d places_lt -b clip_vit_b16 -m lift+ loss_type LA tte True

python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 loss_type CE tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 loss_type Focal tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 loss_type LDAM tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 loss_type CB tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 loss_type GRW tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 loss_type LADE tte True
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 loss_type LA tte True