# LIFT+ on ImageNet-LT (using CLIP)
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+
python main.py -d imagenet_lt -b clip_vit_b16 -m lift+ test_only True tte True

# LIFT+ on Places-LT (using CLIP)
python main.py -d places_lt -b clip_vit_b16 -m lift+
python main.py -d places_lt -b clip_vit_b16 -m lift+ test_only True tte True

# LIFT+ on Places-LT (using ImageNet-21K pre-trained ViT)
python main.py -d places_lt -b in21k_vit_b16 -m lift+
python main.py -d places_lt -b in21k_vit_b16 -m lift+ test_only True tte True

# LIFT+ on iNaturalist 2018 (using CLIP)
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15
python main.py -d inat2018 -b clip_vit_b16 -m lift+ num_epochs 15 test_only True tte True

# LIFT+ on iNaturalist 2018 (using ImageNet-21K pre-trained ViT)
python main.py -d inat2018 -b in21k_vit_b16 -m lift+ num_epochs 15
python main.py -d inat2018 -b in21k_vit_b16 -m lift+ num_epochs 15 test_only True tte True

# LIFT+ on CIFAR-100-LT (using CLIP)
python main.py -d cifar100_ir100 -b clip_vit_b16 -m lift+
python main.py -d cifar100_ir50 -b clip_vit_b16 -m lift+
python main.py -d cifar100_ir10 -b clip_vit_b16 -m lift+
python main.py -d cifar100_ir100 -b clip_vit_b16 -m lift+ test_only True tte True
python main.py -d cifar100_ir50 -b clip_vit_b16 -m lift+ test_only True tte True
python main.py -d cifar100_ir10 -b clip_vit_b16 -m lift+ test_only True tte True

# LIFT+ on CIFAR-100-LT (using ImageNet-21K pre-trained ViT)
python main.py -d cifar100_ir100 -b in21k_vit_b16 -m lift+
python main.py -d cifar100_ir50 -b in21k_vit_b16 -m lift+
python main.py -d cifar100_ir10 -b in21k_vit_b16 -m lift+
python main.py -d cifar100_ir100 -b in21k_vit_b16 -m lift+ test_only True tte True
python main.py -d cifar100_ir50 -b in21k_vit_b16 -m lift+ test_only True tte True
python main.py -d cifar100_ir10 -b in21k_vit_b16 -m lift+ test_only True tte True