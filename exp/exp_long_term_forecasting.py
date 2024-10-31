from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, get_gpu_usage
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag=None):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        print(self.model)

        path = os.path.join('./results/checkpoints/', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.lradj == 'TST':    
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                steps_per_epoch = train_steps,
                                                pct_start = self.args.pct_start,
                                                epochs = self.args.train_epochs,
                                                max_lr = self.args.learning_rate)
        
        gpu_usages = []
        gpu_memories = []
        t1 = time.time()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break


            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            # adjust_learning_rate(model_optim, epoch + 1, self.args)


        t2 = time.time()
        self.Time_Train_pEpoch = (t2-t1)/(epoch + 1)
        self.Mem_Train = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / (1024 * 1024)
        print(f"Mem_Train Usage: {self.Mem_Train} MB")
        usage, memory = get_gpu_usage()
        gpu_usages.append(usage)
        gpu_memories.append(memory)
        print(f"GPU Usage: {sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0}%")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        vs_save_increment = np.floor(len(test_data)/20)
        
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./results/checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './results/visual/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('Testing model')
        self.model.eval()
        t1 = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % vs_save_increment == 0:
                # if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        t2 = time.time()
        self.Time_Infer_pSample = (t2-t1)*1000.0/len(test_data)     # ms/sample
        self.Mem_Infer = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / (1024 * 1024)
        print(f"Mem_Infer Usage: {self.Mem_Infer} MB")
        usage, memory = get_gpu_usage()
        print(f"GPU Usage: {usage}%")
        
        print('Calculating Model Complexity')
        def input_constructor(input_res):
            batch_x, batch_x_mark, dec_inp, batch_y_mark = input_res
            return dict(x_enc=batch_x.to(self.device), 
                        x_mark_enc=batch_x_mark.to(self.device), 
                        x_dec=dec_inp.to(self.device), 
                        x_mark_dec=batch_y_mark.to(self.device))
        temp_input_shape = (batch_x.unsqueeze(1)[0], 
                            batch_x_mark.unsqueeze(1)[0], 
                            dec_inp.unsqueeze(1)[0], 
                            batch_y_mark.unsqueeze(1)[0])
        print(batch_x.shape)
        self.Macs, self.Params = test_params_flop(self.model, temp_input_shape, input_constructor)

        # preds = np.array(preds)
        # trues = np.array(trues)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)

        rst = [mse, 
               mae, 
               self.Time_Train_pEpoch, 
               self.Time_Infer_pSample, 
               self.Mem_Train, 
               self.Mem_Infer,
               self.Macs,
               self.Params]
        
        parts = setting.split("_")[-2:]
        parts[1] = parts[1].replace('seed', '')
        itr_seed = "_".join(parts)
        metircs = '{:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<10} {:<10}'.format(
                    'mse', 'mae', 'trainT (s/ep)', 'inferT (ms/sp)', 'trainMem (MB)', 'inferMem (MB)', 'macs', 'params')
        message = '{:<15.10f} {:<15.10f} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f} {:<10} {:<10}'.format(
                    rst[0], rst[1], rst[2], rst[3], rst[4], rst[5], rst[6], rst[7])
        
        with open("result_long_term_forecast.txt", 'a') as f:
            if parts[0]=='0' or self.args.is_training==0:
                f.write(setting + "  \n")
                f.write('{:<8}  {}\n'.format(' ', metircs))
                print(metircs)
            f.write('{:<8}  {}\n'.format(itr_seed, message))
        print(message)

        return rst

    def log_mean(self, rsts=[]):
        
        import math
        transposed_rsts = list(zip(*rsts))
        list_mean = []
        list_std = []
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
        for lst in transposed_rsts:
            numeric_lst = [float(x) for x in lst if is_number(x)]
            if len(numeric_lst) != len(lst):
                lst_mean = lst[-1]
                lst_std  = 0
            else:
                lst_mean = sum(numeric_lst) / len(numeric_lst)
                variance = sum((x - lst_mean) ** 2 for x in numeric_lst) / len(numeric_lst)
                lst_std  = math.sqrt(variance)
            list_mean.append(lst_mean)
            list_std.append(lst_std)
        
        rst = list_mean
        formatted_mean = '{:<15.10f} {:<15.10f} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f} {:<10} {:<10}'.format(
                        rst[0], rst[1], rst[2], rst[3], rst[4], rst[5], rst[6], rst[7])
        rst = list_std
        formatted_std = '{:<15.10f} {:<15.10f} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f} {:<10} {:<10}'.format(
                        rst[0], rst[1], rst[2], rst[3], rst[4], rst[5], rst[6], rst[7])

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(formatted_mean)
        print(formatted_std)
        print(timestamp)
        
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write('{:<8}  {}\n'.format('Mean:', formatted_mean))
            f.write('{:<8}  {}\n'.format('Std:', formatted_std))
            f.write('{:<8}  {}\n'.format('Time:', timestamp))
            f.write('\n\n')


  