import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import math
import subprocess
from ptflops import get_model_complexity_info

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, scheduler=None, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'PatchTST_type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'SparseTSF_type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == 'FITS_type3':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** ((epoch - 1) // 1))}
    elif args.lradj == 'FITS1':
        lr_adjust = {epoch: args.learning_rate * (0.6 ** ((epoch - 1) // 1))} 
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'fixed':
        print('Constant learning rate {}'.format(args.learning_rate))
        return
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def test_params_flop(model,x_shape,input_constructor=None):
    """ SparseTSF
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    
    VE: This does not work for VarEmbed as torch.einsum is not supported
    """
    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
    #     # print('Flops:' + flops)
    #     # print('Params:' + params)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # from ptflops import get_model_complexity_info
    # print(x_shape)
    macs, params = get_model_complexity_info(model, 
                                             input_res=x_shape, 
                                             input_constructor=input_constructor, 
                                             as_strings=True, print_per_layer_stat=False)
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('{:<30}  {:<8}'.format('Trainable parameter count:', model_params))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs, params

# Function to get GPU usage using nvidia-smi
def get_gpu_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                            stdout=subprocess.PIPE)
    usage, memory = result.stdout.decode('utf-8').strip().split(',')
    # return int(usage), int(memory)
    return float(usage), float(memory)
