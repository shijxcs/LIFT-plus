# LIFT+ with different fine-tuning methods

python main.py -d imagenet_lt -b clip_vit_b16 -m zs tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lp lr 0.03 tte True  # best lr for classifier fine-tuning
python main.py -d imagenet_lt -b clip_vit_b16 -m fft v.fft_layers 12 lr 0.001 tte True  # best lr for full fine-tuning
python main.py -d imagenet_lt -b clip_vit_b16 -m aft v.aft_ratio 0.005 lr 0.02 tte True  # best ratio and lr for arbitrary fine-tuning
python main.py -d imagenet_lt -b clip_vit_b16 -m lp v.bitfit True tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lp v.pt True tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lp v.adapter True tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lp v.lora True tte True
python main.py -d imagenet_lt -b clip_vit_b16 -m lp v.adaptformer True tte True

python main.py -d places_lt -b clip_vit_b16 -m zs tte True
python main.py -d places_lt -b clip_vit_b16 -m lp lr 0.03 tte True  # best lr for classifier fine-tuning
python main.py -d places_lt -b clip_vit_b16 -m fft v.fft_layers 12 lr 0.002 tte True  # best lr for full fine-tuning
python main.py -d places_lt -b clip_vit_b16 -m aft v.aft_ratio 0.005 lr 0.02 tte True  # best ratio and lr for arbitrary fine-tuning
python main.py -d places_lt -b clip_vit_b16 -m lp v.bitfit True tte True
python main.py -d places_lt -b clip_vit_b16 -m lp v.pt True tte True
python main.py -d places_lt -b clip_vit_b16 -m lp v.adapter True tte True
python main.py -d places_lt -b clip_vit_b16 -m lp v.lora True tte True
python main.py -d places_lt -b clip_vit_b16 -m lp v.adaptformer True tte True