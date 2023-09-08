import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import sqrt
from tqdm import tqdm

class grad_accumulator:
    def __init__(self, EEG_ch, num_window):

        self.num_win = num_window
        self.grad_acc = {}
        self.grad_acc["drowsy"] = torch.zeros((self.num_win, EEG_ch, 750))
        self.grad_acc["all"] = torch.zeros((self.num_win, EEG_ch, 750))
        self.grad_acc["alert"] = torch.zeros((self.num_win, EEG_ch, 750))

    def update(self, grad):
        self.grad_acc["all"] += torch.sum(grad[:, self.num_win:, :].cpu(), 0)

"""# Training setup"""

def train_model(model, train_dl, val_dl, save_path=None, model_name=None, epochs=50, **cfg):
    
    optimizer = getattr(optim, cfg['optimizer'])(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = nn.MSELoss(reduction='mean')

    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    record = {'train loss': [], 'val loss':[], 'val cc':[]}
    total_loss = 0
    mini = 1e8
    obt_grad = True if cfg['save_grad'] else False
    grad = grad_accumulator(cfg['EEG_ch'], cfg['num_window'])
    
    print("==========Start training==========")
    for epoch in range(epochs):
        model.train()
        print(f"[{epoch+1}/{epochs}]")

        with tqdm(train_dl, unit="batch") as tepoch:
            for b, (x_train, y_train) in enumerate(tepoch):
                
                optimizer.zero_grad()
                
                ## If we use Siamese architecture
                ## we need to merge the multiple pairs into mini-batch
                ## The ground truth become Delta DI (current DI - baseline DI)
                if cfg["method"] == 'siamese':
                    x_train, y_train = torch.flatten(x_train, 0, 1), torch.flatten(y_train, 0, 1)
                    y_train = y_train[:, 0] - y_train[:, 1]

                x_train, y_train = x_train.to(cfg['device']), y_train.to(cfg['device'])
                x_train.requires_grad = obt_grad

                _, pred = model(x_train)
                loss = criterion(pred, y_train)
                total_loss += loss.detach().cpu().item()

                loss.backward(retain_graph=obt_grad)

                if(obt_grad):
                    grad.update(x_train.grad)

                optimizer.step()
                
                tepoch.set_postfix(loss = total_loss/(b+1))
                tepoch.update(1)

        del x_train, y_train

        ### Validation
        val_loss, rmse, cc = val_model(model, val_dl, cfg["method"], cfg['device'])
        record["train loss"].append(total_loss/len(train_dl.dataset))
        record["val loss"].append(val_loss)
        record["val cc"].append(cc)

        print(f"val loss-> {val_loss} rmse -> {rmse} cc -> {cc}")
        matrice = 1.0* rmse + 0.0*(1-cc)
        if matrice < mini:
            mini = matrice
            model_save_path = f'{save_path}{model_name}_model.pt' # Use test subject to name the model
            torch.save(model.state_dict(), model_save_path)
        
        total_loss = 0

    torch.cuda.empty_cache()
    return record, grad.grad_acc

def val_model(model, test_dl, method='siamese', device='cpu'):
    criterion_mse = nn.MSELoss(reduction='mean')
    total_loss = 0
    output = []

    with torch.no_grad():
        model.eval()
        for x_test, y_test in test_dl:

            if method == 'siamese':
                x_test, y_test = torch.flatten(x_test, 0, 1), torch.flatten(y_test, 0, 1)
                y_test = y_test[:,0] - y_test[:,1]

            x_test, y_test = x_test.to(device), y_test.to(device)

            _, pred = model(x_test)
            
            mse = criterion_mse(pred, y_test)
            rmse = sqrt(mse)
            cc = np.corrcoef(y_test.cpu().detach().numpy().reshape(-1), pred.cpu().detach().numpy().reshape(-1))[0, 1]

        del x_test, y_test

    return mse, rmse, cc

def test_model(model, test_dl, method='siamese', device='cpu'):
    
    model.eval()
    with torch.no_grad():
        for x_test, y_test in test_dl:

            if method == 'siamese':
                x_test, y_test = torch.flatten(x_test, 0, 1), torch.flatten(y_test, 0, 1)
                y_test = y_test[:,0] - y_test[:,1]
            
            x_test, y_test = x_test.to(device), y_test.to(device)

            latent, delta_di = model(x_test)

        del x_test, y_test

    return latent, delta_di