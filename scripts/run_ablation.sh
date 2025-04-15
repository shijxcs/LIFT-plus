# Ablation study for LIFT+

python main.py -d imagenet_lt -b clip_vit_b16 -m lp v.adaptformer False classifier_init None mda False tte False
python main.py -d imagenet_lt -b clip_vit_b16 -m lp v.adaptformer True classifier_init None mda False tte False
python main.py -d imagenet_lt -b clip_vit_b16 -m lp v.adaptformer True classifier_init semantic mda False tte False
python main.py -d imagenet_lt -b clip_vit_b16 -m lp v.adaptformer True classifier_init semantic mda True tte False
python main.py -d imagenet_lt -b clip_vit_b16 -m lp v.adaptformer True classifier_init semantic mda True tte True

python main.py -d places_lt -b clip_vit_b16 -m lp v.adaptformer False classifier_init None mda False tte False
python main.py -d places_lt -b clip_vit_b16 -m lp v.adaptformer True classifier_init None mda False tte False
python main.py -d places_lt -b clip_vit_b16 -m lp v.adaptformer True classifier_init semantic mda False tte False
python main.py -d places_lt -b clip_vit_b16 -m lp v.adaptformer True classifier_init semantic mda True tte False
python main.py -d places_lt -b clip_vit_b16 -m lp v.adaptformer True classifier_init semantic mda True tte True