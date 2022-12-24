import torch 
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp.grad_scaler import GradScaler
from transformers import (
        AutoTokenizer, 
        AutoModel,
        AdamW, 
        get_linear_schedule_with_warmup
        )
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from dataloader import SiameseDataset
from model import PairwiseModel
from torch.utils.data import DataLoader
from utils import collate_fn
from sklearn.metrics import f1_score
from typing import Tuple, Dict
from configs import *
from utils import *
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(model: AutoModel,
                train_loader: DataLoader,
                loss_fn: BCEWithLogitsLoss, 
                optimizer: AdamW,
                scheduler: LambdaLR,
                scaler: GradScaler,
                accumulation_steps: int
                ) -> AutoModel:
    model.train()
    bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    total_loss = 0
    for step, data in bar:
        ids = data["ids"].to(device)
        masks = data["masks"].to(device)
        target = data["target"].to(device)
        preds = model(ids, masks)
        loss = loss_fn(preds.view(-1), target.view(-1))
        loss /= accumulation_steps
        loss.backward()
        if (step+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scaler is not None:
                # ⭐️ ⭐️ Scale Gradients
                scaler.scale(loss).backward()
                # ⭐️ ⭐️ Update Optimizer
                scaler.step(optimizer)
                scaler.update()
            if scheduler is not None:
                scheduler.step()
        bar.set_postfix(loss=loss.item())
        total_loss+=1
    return model

def get_optimizer(
        model: AutoModel, 
        hyperparams: Dict,
        num_train_steps: int, 
        ) -> Tuple: 
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": hyperparams['weight_decay'],
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

    opt = AdamW(optimizer_parameters, lr=hyperparams['lr'])
    sch, scal = None, None
    if hyperparams['is_schedule']:
        sch = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=int(0.05*num_train_steps),
            num_training_steps=num_train_steps,
            last_epoch=-1,
        )
    if hyperparams['is_scaler']:
        scal = torch.cuda.amp.GradScaler()
    return ( opt, sch, scal )

def test_step(
            model: AutoModel, 
            valid_loader: DataLoader,
            loss_fn: BCEWithLogitsLoss
            ) -> Tuple:
    model.eval()
    with torch.no_grad():
        bar = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False)
        targets = []
        all_preds = []
        total_loss = 0
        for _, data in bar:
            ids = data["ids"].to(device)
            masks = data["masks"].to(device)
            target = data["target"].to(device)
            preds = model(ids, masks)
            loss = loss_fn(preds.view(-1), target.view(-1))
            total_loss+=loss
            preds = torch.sigmoid(model(ids, masks))
            all_preds.extend(preds.cpu().view(-1).numpy())
            targets.extend(target.cpu().view(-1).numpy())
        all_preds = np.array(all_preds)
        targets = np.array(targets)
        f1 = f1_score(targets, all_preds > 0.5)
    return (f1, total_loss/len(valid_loader))

def Trainer() -> Dict:
    # Load model pretrain 
    hyperparams = HYPERPARAMETER
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
    model = PairwiseModel(MODEL_NAME)
    df_data = pd.read_csv(PATH_DATA_TRAIN)
    loss_fn = BCEWithLogitsLoss()
    kfold = KFold(n_splits=hyperparams['n_fold'], shuffle=True, random_state=42)
    best_f1 = 0
    best_loss =  1e10
    history = dict()
    for fold, (train_index, test_index) in enumerate(kfold.split(df_data, df_data.label)):
        print("="*100)
        history[fold] = {
            'f1_score': [],
            "loss": []
        }
        model.to(device)
        train_df = df_data.iloc[train_index].reset_index(drop=True)
        val_df = df_data.iloc[test_index].reset_index(drop=True)
        train_dataset = SiameseDataset(train_df, tokenizer, hyperparams['max_length'])
        valid_dataset = SiameseDataset(val_df, tokenizer, hyperparams['max_length'])
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], collate_fn=collate_fn,
                                num_workers=hyperparams['n_workers'], shuffle=True, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=hyperparams['batch_size'], collate_fn=collate_fn,
                                num_workers=hyperparams['n_workers'], shuffle=False, pin_memory=True)
        num_train_steps = len(train_loader) * hyperparams['epochs'] // hyperparams['accumulation_steps']

        optimizer, scheduler, scaler = get_optimizer(model=model, hyperparams=hyperparams, num_train_steps=num_train_steps)
        for epoch in tqdm(range(hyperparams['epochs'])):
            model = train_step(model=model,
                    train_loader=train_loader,
                    loss_fn=loss_fn, 
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    accumulation_steps=hyperparams['accumulation_steps']
                    )
            f1_test, loss_test = test_step(
                model=model, 
                valid_loader=valid_loader,
                loss_fn=loss_fn
            )
            if f1_test > best_f1:
                print(f'F1 score improve from {best_f1: .4f} to {f1_test: .4f} in fold {fold} and epoch {epoch}')
                best_f1 = f1_test
                torch.save(model.state_dict(), PATH_WEIGHTED_F1)
                
            if loss_test < best_loss:
                print(f'Loss improve from {best_loss: .4f} to {loss_test: .4f} in fold {fold} and epoch {epoch}')
                best_loss = loss_test
                torch.save(model.state_dict(), PATH_WEIGHTED_LOSS)
            history[fold]['f1_score'].append(f1_test)
            history[fold]['loss'].append(loss_test)
    return  history          