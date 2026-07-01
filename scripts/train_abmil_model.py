
from __future__ import annotations

import argparse
import gc
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow `python scripts/train_abmil_model.py --config ...` from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.cv_splits import create_folds_by_case, create_folds_from_fixed_blocks


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_transforms(*, data, CFG):
    if data == "train":
        if CFG.aug.augmentation:
            return A.Compose([
                A.RandomCrop(CFG.preprocess.size, CFG.preprocess.size),
                A.HorizontalFlip(p=CFG.aug.HorizontalFlip.p),
                A.VerticalFlip(p=CFG.aug.VerticalFlip.p),
                A.RandomRotate90(p=CFG.aug.RandomRotate90.p),
                A.ShiftScaleRotate(
                    shift_limit=CFG.aug.ShiftScaleRotate.shift_limit,
                    scale_limit=CFG.aug.ShiftScaleRotate.scale_limit,
                    rotate_limit=CFG.aug.ShiftScaleRotate.rotate_limit,
                    p=CFG.aug.ShiftScaleRotate.p),
                A.RandomBrightnessContrast(
                    brightness_limit=CFG.aug.RandomBrightnessContrast.brightness_limit,
                    contrast_limit=CFG.aug.RandomBrightnessContrast.contrast_limit,
                    p=CFG.aug.RandomBrightnessContrast.p),
                A.CLAHE(
                    clip_limit=(1, 4),
                    p=CFG.aug.CLAHE.p),
                A.OneOf([
                    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5),
                    A.Downscale(scale_min=0.1, scale_max=0.15),
                ], p=CFG.aug.compress.p),
                A.CoarseDropout(
                    max_holes=CFG.aug.CoarseDropout.max_holes,
                    max_height=CFG.aug.CoarseDropout.max_height,
                    max_width=CFG.aug.CoarseDropout.max_width,
                    p=CFG.aug.CoarseDropout.p),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        elif not CFG.aug.augmentation:
            return A.Compose([
                A.RandomCrop(CFG.preprocess.size, CFG.preprocess.size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    elif data == "valid" or data == "test":
        return A.Compose([
            A.CenterCrop(CFG.preprocess.size, CFG.preprocess.size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class MILDataset(Dataset):
    def __init__(self, df, CFG, train=True, transform1=None):
        self.df = df
        self.transform = transform1
        self.CFG = CFG
        self.train = train
        self.labels = self.df["target"].to_numpy()
        self.ids = self.df["ID"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        bag = []
        for i in range(self.CFG.N_patch):
            img_path = os.path.join(self.CFG.data.train_dir, f"{image_id}_{i}.png")
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            img = self.transform(image=img)["image"]
            bag.append(torch.as_tensor(img, dtype=torch.float32))

        bag = torch.stack(bag)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return bag, label


class TestDataset(Dataset):
    def __init__(self, df, CFG, transform1=None):
        self.df = df
        self.transform = transform1
        self.CFG = CFG
        self.ids = self.df["ID"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.load_sample(self.ids[idx])

    def get_by_id(self, image_id):
        if image_id not in self.ids:
            raise ValueError(f"ID {image_id} not found in dataset.")
        return self.load_sample(image_id)

    def load_sample(self, image_id):
        bag = []
        for i in range(self.CFG.N_patch):
            img_path = os.path.join(self.CFG.data.test_dir, f"{image_id}_{i}.png")
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            if self.transform is None:
                raise ValueError("Transform is None. Check if 'transform1' was passed correctly.")

            img = self.transform(image=img)["image"]
            img_tensor = torch.as_tensor(img, dtype=torch.float32)
            bag.append(img_tensor)

        bag = torch.stack(bag)
        return bag


class AttentionMIL(nn.Module):
    def __init__(self, CFG, base_model="swin_small_patch4_window7_224", pretrain=True):
        super(AttentionMIL, self).__init__()
        self.CFG = CFG

        self.model = timm.create_model(
            base_model,
            pretrained=pretrain,
            img_size=CFG.model.img_size,
            num_classes=0,
            in_chans=3)
        self.feature_dim = CFG.model.features_num

        self.attention_V = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(128, 1)

        self.classifier = nn.Linear(self.feature_dim, CFG.model.n_classes)

    def forward(self, bag):
        batch_size, num_instances, C, H, W = bag.size()
        bag = bag.view(-1, C, H, W)

        with torch.cuda.amp.autocast(enabled=self.CFG.train.amp):
            features = self.model.forward_features(bag)
            features = features.permute(0, 3, 1, 2)
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)

        features = features.view(batch_size, num_instances, -1)

        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_w(A_V * A_U)
        A = F.softmax(A, dim=1)

        Z = torch.sum(A * features, dim=1)
        y = self.classifier(Z)
        return y


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, num_steps_per_epoch, min_lr=1e-6):
    def lr_lambda(current_step):
        current_epoch = current_step / num_steps_per_epoch
        if current_epoch < warmup_epochs:
            return current_epoch / float(warmup_epochs)
        else:
            progress = (current_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(cosine_decay, min_lr / optimizer.defaults["lr"])

    return LambdaLR(optimizer, lr_lambda)


def train_fn(CFG, fold, folds, test_pl=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"### fold: {fold} ###")

    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index
    val_folds = folds.loc[val_idx].reset_index(drop=True)
    tra_folds = folds.loc[trn_idx]

    print(f"Validation fold shape: {val_folds.shape}, Train fold shape: {tra_folds.shape}")

    if type(test_pl) != type(0):
        tra_folds = pd.concat([tra_folds, test_pl]).reset_index(drop=True)

    model = AttentionMIL(CFG, base_model=CFG.model.name).to(device)
    model.model.set_grad_checkpointing()

    optimizer = AdamW(model.parameters(), lr=CFG.train.lr, weight_decay=CFG.train.weight_decay)
    criterion = nn.CrossEntropyLoss()
    print(criterion)

    softmax = nn.Softmax(dim=-1)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.train.amp)

    best_score = 0
    best_loss = np.inf
    best_preds = None

    train_losses, val_losses, auc_scores = [], [], []

    tmp_dataset = MILDataset(tra_folds, train=True, transform1=get_transforms(data="train", CFG=CFG), CFG=CFG)
    tmp_loader = DataLoader(tmp_dataset, batch_size=CFG.train.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if CFG.train.scheduler.name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=len(tmp_loader) * CFG.train.epochs,
            eta_min=CFG.train.scheduler.min_lr
        )
    elif CFG.train.scheduler.name == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            warmup_epochs=CFG.train.scheduler.warmup,
            total_epochs=CFG.train.epochs,
            num_steps_per_epoch=len(tmp_loader),
            min_lr=CFG.train.scheduler.min_lr
        )
    del tmp_dataset, tmp_loader
    gc.collect()

    for epoch in range(CFG.train.epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.

        train_dataset = MILDataset(tra_folds, train=True, transform1=get_transforms(data="train", CFG=CFG), CFG=CFG)
        train_loader = DataLoader(train_dataset, batch_size=CFG.train.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))
        first_optimizer_step_done = False

        for i, (bags, labels) in tk0:
            optimizer.zero_grad()
            bags, labels = bags.to(device), labels.to(device)

            with autocast(enabled=CFG.train.amp):
                outputs = model(bags)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            if (i + 1) % CFG.train.ga_accum == 0 or i == len(train_loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if not first_optimizer_step_done:
                    first_optimizer_step_done = True
                else:
                    scheduler.step()

            avg_loss += loss.item() / len(train_loader)

        model.eval()
        avg_val_loss = 0.
        LOGITS = []
        valid_labels = []

        valid_dataset = MILDataset(val_folds, train=False, transform1=get_transforms(data="valid", CFG=CFG), CFG=CFG)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.train.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (bags, labels) in tk1:
            bags, labels = bags.to(device), labels.to(device)

            with torch.inference_mode():
                with autocast(enabled=True):
                    logits = model(bags)
                    loss = criterion(logits, labels)

            avg_val_loss += loss.item() / len(valid_loader)
            LOGITS.append(logits.detach().cpu())
            valid_labels.append(labels.detach().cpu())

        logits_tensor = torch.cat(LOGITS, dim=0).float()
        preds = softmax(logits_tensor).numpy()
        valid_labels = torch.cat(valid_labels).numpy()

        auc_score = roc_auc_score(valid_labels, preds[:, 1])
        elapsed = time.time() - start_time

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        auc_scores.append(auc_score)

        print(f"  Epoch {epoch + 1} - val_auc_score: {auc_score:.6f}")
        print(
            f"  Epoch {epoch + 1} - "
            f"avg_train_loss: {avg_loss:.6f}, avg_val_loss: {avg_val_loss:.6f}, "
            f"time: {elapsed:.0f}s"
        )

        if best_loss > avg_val_loss:
            best_loss = avg_val_loss
            print(f"  Epoch {epoch+1} - Save Best loss: {best_loss:.4f}")
            best_loss_path = os.path.join(CFG.data.output_dir, f"fold{fold}_{CFG.general.exp_num}_best_loss.pth")
            torch.save(model.state_dict(), best_loss_path)

        if auc_score > best_score:
            best_score = auc_score
            print(f"  Epoch {epoch+1} - Save Best AUC: {best_score:.4f}")
            best_preds = preds
            best_AUC_path = os.path.join(CFG.data.output_dir, f"fold{fold}_{CFG.general.exp_num}_best_AUC.pth")
            torch.save(model.state_dict(), best_AUC_path)

        if auc_score >= best_score and avg_val_loss <= best_loss:
            print(f"  Epoch {epoch+1} - Save Best AUC + Best Loss Model")
            best_score = auc_score
            best_loss = avg_val_loss
            best_auc_loss_path = os.path.join(CFG.data.output_dir, f"fold{fold}_{CFG.general.exp_num}_best_auc_loss.pth")
            torch.save(model.state_dict(), best_auc_loss_path)

        del train_loader, valid_loader, train_dataset, valid_dataset
        gc.collect()
        torch.cuda.empty_cache()

    loss_df = pd.DataFrame({
        "epoch": list(range(1, CFG.train.epochs + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "auc_score": auc_scores
    })
    loss_path = os.path.join(CFG.data.output_dir, f"loss_fold{fold}.csv")
    loss_df.to_csv(loss_path, index=False)
    print(f"Losses saved to {loss_path}")

    for i in range(2):
        col = f"pred_{i}"
        val_folds[col] = best_preds[:, i]

    return best_preds, valid_labels, val_folds


def inf_func(models, valid_loader, device, has_labels=False):
    softmax = nn.Softmax(dim=-1)

    for model in models:
        model.eval()

    preds = []
    tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

    for i, batch in tk1:
        if has_labels:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                bags, labels = batch
            else:
                print(f"[Warning] Unexpected batch format at step {i}: {batch}")
                continue
        else:
            bags = batch

        if not isinstance(bags, torch.Tensor):
            print(f"[Warning] bags is not Tensor at step {i}: {type(bags)}")
            continue

        bags = bags.to(device, non_blocking=True)

        with torch.no_grad(), autocast():
            logits_list = [model(bags.float()) for model in models]
            avg_logits = torch.stack(logits_list).mean(0)
            y_preds = softmax(avg_logits)
            preds.append(y_preds)

    preds = torch.cat(preds).cpu().numpy()
    return preds


def oof_prediction(CFG, folds_df, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = []
    valid_labels = []
    oof = pd.DataFrame()

    for fold in range(CFG.general.num_folds):
        model_path = os.path.join(save_dir, f"fold{fold}_{CFG.general.exp_num}_best_auc_loss.pth")

        if not os.path.exists(model_path):
            print(f"[Train] fold{fold} model not found. Starting training...")
            seed_torch(seed=CFG.general.seed)
            _preds, _valid_labels, _oof_val = train_fn(CFG, fold, folds_df)
            preds.append(_preds)
            valid_labels.append(_valid_labels)
            oof = pd.concat([oof, _oof_val], ignore_index=True)
            continue

        print(f"[OOF] Running inference for fold {fold}")
        val_df = folds_df[folds_df["fold"] == fold].reset_index(drop=True)

        model = AttentionMIL(CFG, base_model=CFG.model.name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        val_dataset = MILDataset(val_df, CFG=CFG, train=False, transform1=get_transforms(data="valid", CFG=CFG))
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)

        y_pred = inf_func([model], val_loader, device, has_labels=True)
        y_true = val_df["target"].values

        preds.append(y_pred)
        valid_labels.append(y_true)

        oof_fold = pd.DataFrame({
            "ID": val_df["ID"],
            "target": y_true,
            "pred_0": y_pred[:, 0],
            "pred_1": y_pred[:, 1],
            "fold": fold
        })
        oof = pd.concat([oof, oof_fold], ignore_index=True)

    return preds, valid_labels, oof


def submit(CFG, test_df, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for fold in range(CFG.general.num_folds):
        model_path = os.path.join(save_dir, f"fold{fold}_{CFG.general.exp_num}_best_auc_loss.pth")
        if not os.path.exists(model_path):
            print(f"[Skip] fold{fold} model not found: {model_path}")
            continue

        model = AttentionMIL(CFG, base_model=CFG.model.name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    test_dataset = TestDataset(test_df, CFG, transform1=get_transforms(data="test", CFG=CFG))
    test_loader = DataLoader(test_dataset, batch_size=CFG.test.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_preds = inf_func(models, test_loader, device, has_labels=False)

    test_df = test_df.copy()
    test_df["pred_0"] = test_preds[:, 0]
    test_df["pred_1"] = test_preds[:, 1]
    return test_df


def load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cnf = OmegaConf.load(config_path)
    os.makedirs(cnf.data.output_dir, exist_ok=True)

    config_dst = os.path.join(cnf.data.output_dir, f"config_used_exp{cnf.general.exp_num}.yaml")
    shutil.copyfile(str(config_path), config_dst)
    print(f"Saved config to {config_dst}")
    return cnf


def resolve_folds_from_config(CFG):
    fold_csv_path = os.path.join(CFG.data.output_dir, "fold.csv")

    if hasattr(CFG, "cv"):
        method = CFG.cv.method

        if method == "groupkfold":
            if hasattr(CFG.cv, "groupkfold") and hasattr(CFG.cv.groupkfold, "train_csv"):
                train_csv_path = CFG.cv.groupkfold.train_csv
            elif hasattr(CFG.data, "train_csv"):
                train_csv_path = CFG.data.train_csv
            else:
                raise ValueError("groupkfold requires cv.groupkfold.train_csv or data.train_csv")

            folds = create_folds_by_case(
                data_path=train_csv_path,
                output_path=fold_csv_path,
                n_splits=CFG.general.num_folds,
                random_state=CFG.general.seed,
            )

        elif method == "fixed_blocks":
            if not hasattr(CFG.cv, "fixed_blocks") or not hasattr(CFG.cv.fixed_blocks, "block_csvs"):
                raise ValueError("fixed_blocks requires cv.fixed_blocks.block_csvs")

            block_csvs = list(CFG.cv.fixed_blocks.block_csvs)
            if len(block_csvs) != CFG.general.num_folds:
                raise ValueError(
                    f"fixed_blocks requires exactly {CFG.general.num_folds} CSVs, but got {len(block_csvs)}"
                )

            folds = create_folds_from_fixed_blocks(
                csv_paths=block_csvs,
                output_path=fold_csv_path,
            )
        else:
            raise ValueError(f"Unknown cv.method: {method}")

    else:
        if not hasattr(CFG.data, "train_csv"):
            raise ValueError("Missing train CSV. Use data.train_csv or cv.groupkfold.train_csv")

        folds = create_folds_by_case(
            data_path=CFG.data.train_csv,
            output_path=fold_csv_path,
            random_state=2023,
        )

    return folds, fold_csv_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the ABMIL model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    CFG = load_config(args.config)

    folds, fold_csv_path = resolve_folds_from_config(CFG)
    print(folds.columns)
    print(folds.groupby(["fold", "case", "target"]).size().unstack(fill_value=0))

    folds = pd.read_csv(fold_csv_path)
    preds, valid_labels, oof = oof_prediction(CFG, folds, save_dir=CFG.data.output_dir)

    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    oof_path = os.path.join(CFG.data.output_dir, f"oof_{CFG.general.exp_num}.csv")
    oof.to_csv(oof_path, index=False)
    print(f"OOF saved to {oof_path}")

    if preds.shape[1] == 2:
        auc_score = roc_auc_score(valid_labels, preds[:, 1])
    else:
        auc_score = roc_auc_score(valid_labels, preds, multi_class="ovr")
    print(f"AUC Score_oof: {auc_score}")

    test_df = pd.read_csv(CFG.data.test_csv)
    test_predictions = submit(CFG, test_df, save_dir=CFG.data.output_dir)

    test_path = os.path.join(CFG.data.output_dir, f"test_{CFG.general.exp_num}.csv")
    test_predictions.to_csv(test_path, index=False)
    print(f"Test predictions saved to {test_path}")


if __name__ == "__main__":
    main()
