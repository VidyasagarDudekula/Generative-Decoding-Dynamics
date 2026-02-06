from config import ModelArgs
from model import LLamaModel
from bpe_tokenizer import BPETokenizer
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import glob
import json

device = 'cpu'
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')



cfg_data = ModelArgs()
cfg_data.mode = 'Train'

ds = BPETokenizer(cfg_data)
dl_train = DataLoader(ds, shuffle=True, batch_size=cfg_data.max_batch_size)
print(f"Training Data loaded")
cfg_eval_data = ModelArgs()
cfg_eval_data.mode = 'Eval'
ds_eval = BPETokenizer(cfg_eval_data)
dl_eval = DataLoader(ds_eval, shuffle=True, batch_size=cfg_eval_data.max_batch_size)
print(f"Eval Data loaded")

x, y, mask = next(iter(dl_train))
print("x:", x.shape, x.dtype)
print("y:", y.shape, y.dtype)
print("mask:", mask.shape, mask.dtype)


@torch.no_grad()
def get_eval_loss(model):
    accumulate_loss = 0.0
    accumulate_n = 0
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=ds_eval.pad_token)
    for xb, yb, mask in dl_eval:
        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)
        out = model(xb, mask)
        loss = criterion(out.reshape(-1, ds_eval.vocab_size), yb.reshape(-1))
        accumulate_loss += loss.item()
        accumulate_n += 1
    model.train()

    return accumulate_loss/accumulate_n




def train():
    cfg = ModelArgs()
    cfg.mode = 'Train'
    model = LLamaModel(cfg)
    os.makedirs(cfg.model_folder, exist_ok=True)
    stats_file = f"{cfg.model_folder}/stats.jsonl"
    model = model.to(device)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
    total_parameters = sum([p.numel() for p in model.parameters()])
    print(f"Total Parameters:- {total_parameters}")
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=ds.pad_token, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs * len(dl_train))
    stepi = []
    step = 0
    lossi = []
    evali = []
    accumulate_loss = 0.0
    accumulate_n = 0
    start_epoch = 0
    
    if os.path.exists(stats_file):
        print(f"Loading stats from {stats_file}...")
        with open(stats_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                stepi.append(data['step'])
                lossi.append(data['train_loss'])
                evali.append(data['eval_loss'])
    checkpoints = glob.glob(f"{cfg.model_folder}/checkpoint_*.pth")
    
    if checkpoints:
        latest_checkpoint_path = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"--> Resuming from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        step = checkpoint['global_step']
    else:
        print("--> No checkpoint found. Starting from scratch.")
    print(f"From Epoch:- {start_epoch}, step:-{step}")
    for epoch in range(cfg.num_epochs):
        for i, (xb, yb, mask) in enumerate(dl_train):
            if epoch == start_epoch:
                batches_processed_in_epoch = step % len(dl_train)
                if i < batches_processed_in_epoch:
                    continue
            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)
            accumulate_n +=1
            optimizer.zero_grad(set_to_none=True)
            out = model(xb, mask)
            out = out.reshape(-1, ds.vocab_size)
            yb = yb.reshape(-1)
            loss = criterion(out, yb)
            accumulate_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if step%200==0:
                train_loss = accumulate_loss / accumulate_n
                accumulate_loss = 0.0
                accumulate_n = 0
                eval_loss = get_eval_loss(model)
                with open(stats_file, 'a') as f:
                    f.write(json.dumps({'step': step, 'train_loss': train_loss, 'eval_loss': eval_loss}) + "\n")
                stepi.append(step)
                lossi.append(train_loss)
                evali.append(eval_loss)
                print(f"Epoch: {epoch}, step:- {step}, Loss:- {train_loss}, eval_loss:- {eval_loss}")
            step += 1
        file_name = f"{cfg.model_folder}/checkpoint_{step}.pth"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "global_step": step, "scheduler_state_dict": scheduler.state_dict()}, file_name)
    
    plt.figure(figsize=(10, 6))
    plt.plot(stepi, lossi, label='Training Loss', color='blue')
    plt.plot(stepi, evali, label='Validation Loss', color='orange')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    print("Plot saved as training_loss_plot.png")



if __name__ == '__main__':
    train()