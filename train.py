from torch._dynamo.polyfills.itertools import accumulate
from config import ModelArgs
from model import LLamaModel
from bpe_tokenizer import BPETokenizer
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = 'cpu'
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('mps')



cfg_data = ModelArgs()
cfg_data.mode = 'Train'

ds = BPETokenizer(cfg_data)
dl_train = DataLoader(ds, shuffle=True, batch_size=16)
print(f"Training Data loaded")
cgf_data = 'Eval'
ds_eval = BPETokenizer(cfg_data)
dl_eval = DataLoader(ds_eval, shuffle=True, batch_size=16)
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
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=ds.pad_token, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    stepi = []
    step = 0
    lossi = []
    evali = []
    accumulate_loss = 0.0
    accumulate_n = 0
    for epoch in range(cfg.num_epochs):
        for xb, yb, mask in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            accumulate_n +=1
            optimizer.zero_grad()
            out = model(xb, mask)
            out = out.reshape(-1, ds.vocab_size)
            yb = yb.reshape(-1)
            loss = criterion(out, yb)
            accumulate_loss += loss.item()
            loss.backward()
            optimizer.step()
            if step%200==0:
                train_loss = accumulate_loss / accumulate_n
                accumulate_loss = 0.0
                accumulate_n = 0
                eval_loss = get_eval_loss(model)
                stepi.append(step)
                lossi.append(train_loss)
                evali.append(eval_loss)
                print(f"Epoch: {epoch}, step:- {step}, Loss:- {train_loss}, eval_loss:- {eval_loss}")
            step += 1
        file_name = f"{cfg['model_folder']}/checkpoint_{step}.pth"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "global_step": step}, file_name)
    
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