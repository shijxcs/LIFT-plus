import os
import sys
import time
import datetime
import math
import random
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from clip import clip
from timm.models.vision_transformer import (
    vit_base_patch16_224, vit_base_patch16_384,
    vit_large_patch16_224, vit_large_patch16_384,
)
from timm.models.resnet import resnet50, resnet101, resnet152

from models import PEFT_Model
import datasets

from utils.evaluator import Evaluator
from utils.losses import *
from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.transforms import *


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.build_model()
        self.build_data()
        self.build_criterion()
        self.build_tuner()
        if not (cfg.zero_shot or cfg.test_only):
            self.build_optimizer()
        
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
    
    def build_model(self):
        cfg = self.cfg
        print("Building model")
        
        # load model
        if cfg.backbone.startswith("CLIP-"):
            backbone = cfg.backbone[5:]
            print(f"Loading CLIP (backbone: {backbone})")
            model = clip.load(backbone, device=self.device)[0]
        
        elif cfg.backbone.startswith("IN21K-"):
            backbone = cfg.backbone[6:]
            print(f"Loading IN21K pre-trained model (backbone: {backbone})")
            if backbone == "ViT-B/16":
                model = vit_base_patch16_224(pretrained=True).eval()
            elif backbone == "ViT-B/16@384px":
                model = vit_base_patch16_384(pretrained=True).eval()
            elif backbone == "ViT-L/16":
                model = vit_large_patch16_224(pretrained=True).eval()
            elif backbone == "ViT-L/16@384px":
                model = vit_large_patch16_384(pretrained=True).eval()
            else:
                raise ValueError
        
        elif cfg.backbone.startswith("IN1K-"):
            backbone = cfg.backbone[5:]
            print(f"Loading IN1K pre-trained model (backbone: {backbone})")
            if backbone == "RN50":
                model = resnet50(pretrained=True).eval()
            elif backbone == "RN101":
                model = resnet101(pretrained=True).eval()
            elif backbone == "RN152":
                model = resnet152(pretrained=True).eval()
            else:
                raise ValueError

        else:
            raise ValueError

        self.model = PEFT_Model(model).to(self.device)

        prec = cfg.prec_train
        if prec == "fp16":
            self.model.half()
        elif prec in ["fp32", "amp"]:
            self.model.float()
        else:
            raise ValueError

        model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model params: {model_params}")

    def build_data(self):
        cfg = self.cfg
        resolution = cfg.resolution
        mean = cfg.mean
        std = cfg.std

        print("Building data")

        if cfg.mda:
            transform_train = transforms.Compose([
                MinimalistRandomResizedCrop(resolution, cfg.num_epochs, sched_func=cfg.mda_func, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(resolution, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if cfg.tte:
            expand = cfg.expand if cfg.expand is not None else 24
            transform_test = transforms.Compose([
                transforms.Resize(resolution + expand, interpolation=InterpolationMode.BICUBIC),
                transforms.FiveCrop(resolution),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Normalize(mean, std),
            ])
        else:
            expand = cfg.expand if cfg.expand is not None else resolution // 7
            transform_test = transforms.Compose([
                transforms.Resize(resolution + expand, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        train_dataset = getattr(datasets, cfg.dataset)(cfg.root, split="train", transform=transform_train)
        init_dataset = getattr(datasets, cfg.dataset)(cfg.root, split="train", transform=transform_plain)
        test_dataset = getattr(datasets, cfg.dataset)(cfg.root, split="test", transform=transform_test)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        self.many_classes = (torch.tensor(self.cls_num_list) > 100).nonzero().squeeze()
        self.med_classes = ((torch.tensor(self.cls_num_list) >= 20) & (torch.tensor(self.cls_num_list) <= 100)).nonzero().squeeze()
        self.few_classes = (torch.tensor(self.cls_num_list) < 20).nonzero().squeeze()

        assert cfg.batch_size % cfg.accum_step == 0, "batch_size must be divisible by accum_step"
        micro_batch_size = cfg.batch_size // cfg.accum_step

        self.train_loader = DataLoader(train_dataset,
            batch_size=micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

        self.init_loader = DataLoader(init_dataset,
            batch_size=64, shuffle=False,
            sampler=DownSampler(init_dataset.labels, n_max=100),
            num_workers=cfg.num_workers, pin_memory=True)
        
        self.test_loader = DataLoader(test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
    
        print("Total training points:", sum(self.cls_num_list))
        print("Class frequencies:", self.cls_num_list)

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal":
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM":
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.classifier_scale)
        elif cfg.loss_type == "CB":
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW":
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS":
            self.criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA":
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE":
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        else:
            raise ValueError
    
    def build_tuner(self):
        cfg = self.cfg

        print("Building tuner")

        if cfg.coop:
            if cfg.coop_init is None:
                ctx_len = cfg.coop_ctx_len
                cls_pos = cfg.coop_cls_pos

                if cls_pos == "front":
                    self.template = "{}" + " X" * ctx_len + "."
                    ctx_loc = list(range(-ctx_len - 1, -1))

                elif cls_pos == "middle":
                    ctx_len_left, ctx_len_right = (ctx_len // 2), (ctx_len + 1) // 2
                    self.template = "X " * ctx_len_left + "{}" + " X" * ctx_len_right + "."
                    ctx_loc = list(range(ctx_len_left)) + list(range(-ctx_len_right - 1, -1))

                elif cls_pos == "end":
                    self.template = "X " * ctx_len + "{}."
                    ctx_loc = list(range(ctx_len))
                
                else:
                    raise ValueError
                
                print("Add learnable context with template '{}'.".format(self.template))
                self.model.text_encoder.add_learnable_context(ctx_loc=ctx_loc)
            else:
                if cfg.coop_init == "photo":
                    self.template = "a photo of a {}."
                    ctx_loc = [0, 1, 2, 3, -1]
                else:
                    raise ValueError

                print("Note: Template '{}' is used to initialize the context.".format(self.template),
                      "The context length is {}, and `ctx_len` will be deprecated.".format(len(ctx_loc)))
                self.model.text_encoder.add_learnable_context(ctx_loc=ctx_loc, init_text=self.template)
        else:
            self.template = "a photo of a {}."
            print("Use template '{}' for prompt generation.".format(self.template))

        for _name, _cfg in (("image_encoder", cfg.v), ("text_encoder", cfg.l)):
            if not hasattr(self.model, _name):
                continue
            
            _encoder = getattr(self.model, _name)
            
            def parse_layers(layers):
                n_layers = len(_encoder.blocks)
                if layers is None:
                    return list(range(n_layers))
                elif isinstance(layers, int):
                    return list(range(n_layers - layers, n_layers))
                else:
                    return eval(layers)
            
            if _cfg.fft:
                layers = parse_layers(_cfg.fft_layers)
                print("Fine-tune all parameters in layers {}.".format(layers))
                _encoder.unfreeze_params(layers=layers)

            if _cfg.bitfit:
                layers = parse_layers(_cfg.bitfit_layers)
                print("Fine-tune bias parameters in layers {}.".format(layers))
                _encoder.unfreeze_bias(layers=layers)
            
            if _cfg.pt:
                layers = parse_layers(_cfg.pt_layers)
                prompt_len = _cfg.pt_len or 2 ** max(0, int(math.log2(self.num_classes / (len(layers)))))
                print("Add learnable prompt with length {} to layers {}.".format(prompt_len, layers))
                _encoder.add_learnable_prompt(layers=layers, prompt_len=prompt_len)
            
            if _cfg.lora:
                layers = parse_layers(_cfg.lora_layers)
                bottle_dim = _cfg.lora_dim or 2 ** max(0, int(math.log2(self.num_classes / (len(layers) * 4))))
                print("Add LoRA with bottle dimension {} to layers {}.".format(bottle_dim, layers))
                _encoder.add_lora(layers=layers, bottle_dim=bottle_dim)

            if _cfg.adapter:
                layers = parse_layers(_cfg.adapter_layers)
                bottle_dim = _cfg.adapter_dim or 2 ** max(0, int(math.log2(self.num_classes / (len(layers) * 2))))
                print("Add Adapter with bottle dimension {} to layers {}.".format(bottle_dim, layers))
                _encoder.add_adapter(layers=layers, bottle_dim=bottle_dim)
            
            if _cfg.adaptformer:
                layers = parse_layers(_cfg.adaptformer_layers)
                bottle_dim = _cfg.adaptformer_dim or 2 ** max(0, int(math.log2(self.num_classes / (len(layers) * 2))))
                print("Add AdaptFormer with bottle dimension {} to layers {}.".format(bottle_dim, layers))
                _encoder.add_adaptformer(layers=layers, bottle_dim=bottle_dim)
            
            if _cfg.ssf:
                layers = parse_layers(_cfg.ssf_layers)
                print("Add SSF to layers {}.".format(layers))
                _encoder.add_ssf(layers=layers)

            if _cfg.aft:
                layers = parse_layers(_cfg.aft_layers)
                loc = _cfg.aft_loc
                seed = _cfg.aft_seed
                if _cfg.aft_ratio is not None:
                    ratio = float(_cfg.aft_ratio)
                elif loc == "attn":
                    ratio = self.num_classes / (len(layers) * (_encoder.embed_dim * 4 + 4))
                elif loc == "mlp":
                    ratio = self.num_classes / (len(layers) * (_encoder.embed_dim * 8 + 5))
                elif loc == "all":
                    ratio = self.num_classes / (len(layers) * (_encoder.embed_dim * 12 + 9))
                else:
                    raise ValueError
                print("Fine-tune a random part of parameters in {} layers {}".format(loc, layers))
                _encoder.add_aft(layers=layers, ratio=ratio, loc=loc, seed=seed)
                
        if cfg.proj_tuning:
            print("Fine-tune the projections of both branches.")
            self.model.unfreeze_image_proj()
            self.model.unfreeze_text_proj()
        
        if cfg.clip_adapter:
            bottle_dim = cfg.clip_adapter_dim
            print("Add CLIP-Adapter with bottle dimension {}.".format(bottle_dim))
            self.model.add_clip_adapter(bottle_dim=bottle_dim)

        if cfg.classifier:
            print("Add classifier on top of the vision model.")
            self.model.add_classifier(cfg.classifier, self.num_classes, scale=cfg.classifier_scale)

            if not (cfg.zero_shot or cfg.test_only) and cfg.classifier_init is not None:
                classifier_init = cfg.classifier_init
                
                if classifier_init == "semantic":
                    print("Using semantic-aware initialization.")
                    with torch.no_grad():
                        class_features = self.compute_class_features(self.generate_class_prompts())
                    self.model.init_classifier_weight(class_features, feature_modality="text")

                elif classifier_init == "class_mean":
                    print("Using class mean feature for initialization.")
                    with torch.no_grad():
                        train_features, train_labels = self.compute_train_features()
                    # compute class means
                    sorted_index = train_labels.argsort()
                    train_features = train_features[sorted_index]
                    train_labels = train_labels[sorted_index]
                    _, label_counts = torch.unique(train_labels, return_counts=True)
                    class_means = torch.stack([x.mean(dim=0) for x in torch.split(train_features, label_counts.tolist())])
                    # initialize classifier
                    self.model.init_classifier_weight(class_means, feature_modality="image")

                elif classifier_init == "linear_probing":
                    print("Using linear probing for initialization.")
                    with torch.no_grad():
                        train_features, train_labels = self.compute_train_features()
                    # compute classifier weights
                    clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced")
                    clf.fit(train_features.cpu(), train_labels.cpu())
                    class_weights = torch.from_numpy(clf.coef_).to(train_features.dtype).to(self.device)
                    class_weights = F.normalize(class_weights, dim=-1)
                    # initialize classifier
                    self.model.init_classifier_weight(class_weights, feature_modality="image")

                else:
                    raise ValueError
                
                torch.cuda.empty_cache()

        self.tuner = self.model.tuner
        
        # print parameters
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        print(f"Tuned params: {tuned_params}")
        for name, param in self.tuner.named_parameters():
            print(f"├─{name}: {param.numel()}")
    
    def build_optimizer(self):
        cfg = self.cfg
        
        print("Turning off gradients in the model.")
        for param in self.model.parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner.")
        for param in self.tuner.parameters():
            param.requires_grad_(True)

        self.optim = torch.optim.SGD(self.tuner.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.optim.zero_grad()
        
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = torch.GradScaler("cuda") if cfg.prec_train == "amp" else None

    def generate_class_prompts(self):
        prompts = [self.template.format(name.replace("_", " ")) for name in self.classnames]
        prompts = clip.tokenize(prompts)  # [n_cls, seq_len]
        prompts = prompts.to(self.device)
        return prompts

    def compute_class_features(self, prompts):
        if len(prompts) <= 1000:
            class_features = self.model.text_encoder(prompts)
        else:
            # CUDA out of memory
            prompt_splits = torch.split(prompts, 1000)
            class_features = torch.cat([self.model.text_encoder(x) for x in prompt_splits])
        return class_features
    
    def compute_train_features(self):
        all_features = torch.Tensor([]).to(self.device)
        all_labels = torch.Tensor([]).to(self.device)

        print("Computing training features.")
        for image, label in tqdm(self.init_loader, ascii=True):
            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, return_feature=True)

            all_features = torch.cat([all_features, feature])
            all_labels = torch.cat([all_labels, label])

        return all_features, all_labels
        # sorted_index = all_labels.argsort()
        # all_features = all_features[sorted_index]
        # all_labels = all_labels[sorted_index]
        # _, label_counts = torch.unique(all_labels, return_counts=True)
        # class_means = torch.stack([x.mean(dim=0) for x in torch.split(all_features, label_counts.tolist())])
        # return class_means

    def train(self):
        cfg = self.cfg

        # Initialize tensorboard summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        tb_writer = SummaryWriter(log_dir=writer_dir)
        
        # Initialize average meters
        batch_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        if cfg.classifier:
            model_args = {"use_classifier": True}
        else:
            print("Generating class prompts.")
            text = self.generate_class_prompts()
            model_args = {"text": text, "is_text_feature": False}
        
        print("Start training")
        # Record the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, (image, label) in enumerate(self.train_loader):
                image = image.to(self.device)
                label = label.to(self.device)

                if cfg.prec_train == "amp":
                    with torch.autocast(device_type="cuda"):
                        logit = self.model(image=image, **model_args)
                        loss = self.criterion(logit, label)
                    self.scaler.scale(loss / cfg.accum_step).backward()
                    if ((batch_idx + 1) % cfg.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    logit = self.model(image=image, **model_args)
                    loss = self.criterion(logit, label)
                    (loss / cfg.accum_step).backward()
                    if ((batch_idx + 1) % cfg.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                with torch.no_grad():
                    pred = logit.argmax(dim=1)
                    correct = pred.eq(label)
                    acc = correct.float().mean().mul_(100.0)
                
                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.float().mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = torch.mean(torch.Tensor(cls_accs))
                many_acc = torch.mean(torch.Tensor(cls_accs)[self.many_classes])
                med_acc = torch.mean(torch.Tensor(cls_accs)[self.med_classes])
                few_acc = torch.mean(torch.Tensor(cls_accs)[self.few_classes])
                
                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                    
                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.2f} ({acc_meter.avg:.2f})"]
                    info += [f"(mean {mean_acc:.2f} many {many_acc:.2f} med {med_acc:.2f} few {few_acc:.2f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))
                    sys.stdout.flush()

                iter_idx = epoch_idx * num_batches + batch_idx
                tb_writer.add_scalar("train/lr", current_lr, iter_idx)
                tb_writer.add_scalar("train/loss.val", loss_meter.val, iter_idx)
                tb_writer.add_scalar("train/loss.avg", loss_meter.avg, iter_idx)
                tb_writer.add_scalar("train/acc.val", acc_meter.val, iter_idx)
                tb_writer.add_scalar("train/acc.avg", acc_meter.avg, iter_idx)
                tb_writer.add_scalar("train/mean_acc", mean_acc, iter_idx)
                tb_writer.add_scalar("train/many_acc", many_acc, iter_idx)
                tb_writer.add_scalar("train/med_acc", med_acc, iter_idx)
                tb_writer.add_scalar("train/few_acc", few_acc, iter_idx)

                end = time.time()
            
            self.sched.step()
            for t in self.train_loader.dataset.transform.transforms:
                if isinstance(t, MinimalistRandomResizedCrop):
                    t.step()
            # torch.cuda.empty_cache()
        
        print("Finish training")
        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")
        
        # save model
        self.save_model(cfg.output_dir)

        # close writer
        tb_writer.close()

    def test(self):
        cfg = self.cfg
        self.tuner.eval()

        prec = cfg.prec_test
        if prec == "fp16":
            self.model.half()
        elif prec == "fp32":
            self.model.float()
        else:
            raise ValueError

        print(f"Evaluate on the test set")

        evaluator = Evaluator()

        if cfg.classifier:
            model_args = {"use_classifier": True}
        else:
            print("Pre-computing class features for testing.")
            text = self.generate_class_prompts()
            with torch.no_grad():
                text = self.compute_class_features(text)
            model_args = {"text": text, "is_text_feature": True}
        
        for image, label in tqdm(self.test_loader, ascii=True, desc="Testing"):
            image = image.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                if cfg.tte:  # [bsz, ncrops, C, H, W] 
                    logit = torch.stack([self.model(image=x, **model_args) for x in image.unbind(dim=1)]).mean(dim=0)
                else:
                    logit = self.model(image=image, **model_args)

            evaluator.process(logit, label)

        evaluator.evaluate(self.many_classes, self.med_classes, self.few_classes)

    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
        }

        # remove 'module.' in state_dict's keys
        for key in ["tuner"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)
        tuner_dict = checkpoint["tuner"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict, strict=False)
