# LIFT+ on ImageNet-LT with ViT-L/14
python main.py -d imagenet_lt -b clip_vit_l14 -m lift+ accum_step 4
python main.py -d imagenet_lt -b clip_vit_l14 -m lift+ accum_step 4 test_only True tte True

# LIFT+ on Places-LT with ViT-L/14
python main.py -d places_lt -b clip_vit_l14 -m lift+ accum_step 4
python main.py -d places_lt -b clip_vit_l14 -m lift+ accum_step 4 test_only True tte True

# LIFT+ on iNaturalist 2018 with ViT-L/14
python main.py -d inat2018 -b clip_vit_l14 -m lift+ num_epochs 15 accum_step 4
python main.py -d inat2018 -b clip_vit_l14 -m lift+ num_epochs 15 accum_step 4 test_only True tte True

# LIFT+ on iNaturalist 2018 with ViT-L/14@336px
python main.py -d inat2018 -b clip_vit_l14@336px -m lift+ num_epochs 15 accum_step 8
python main.py -d inat2018 -b clip_vit_l14@336px -m lift+ num_epochs 15 accum_step 8 test_only True tte True

# LIFT+ on ImageNet-LT with ResNet-50
python main.py -d imagenet_lt -b clip_rn50 -m lp v.ssf True
python main.py -d imagenet_lt -b clip_rn50 -m lp v.bitfit True

# LIFT+ on Places-LT with ResNet-50
python main.py -d places_lt -b clip_rn50 -m lp v.ssf True
python main.py -d places_lt -b clip_rn50 -m lp v.bitfit True