# LIFT+ with different tte_expand sizes (after running the main results and getting the checkpoints)

python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 0
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 4
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 8
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 12
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 16
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 20
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 24
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 28
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 32
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 36
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 40
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 44
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 48

python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 0
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 4
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 8
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 12
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 16
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 20
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 24
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 28
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 32
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 36
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 40
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 44
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 48

python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 0
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 4
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 8
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 12
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 16
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 20
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 24
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 28
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 32
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 36
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 40
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 44
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte False tte_expand 48

python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 0
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 4
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 8
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 12
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 16
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 20
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 24
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 28
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 32
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 36
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 40
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 44
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True tte_expand 48

python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 0
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 4
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 8
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 12
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 16
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 20
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 24
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 28
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 32
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 36
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 40
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 44
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte False tte_expand 48

python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 0
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 4
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 8
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 12
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 16
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 20
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 24
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 28
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 32
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 36
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 40
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 44
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True expand 48