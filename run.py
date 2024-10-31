import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.model_log import log_config, model_log
import random
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VE Pipeline')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--test_ii', type=int, default=0, help='random seed index when loading model in testing')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    
    # custom   
    parser.add_argument('--debug', type=int,default=0, help='Debug mode')

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
    
    # FITS
    parser.add_argument('--cut_freq', type=int,default=0)
    parser.add_argument('--base_T', type=int,default=24)
    parser.add_argument('--H_order', type=int,default=2)
    
    # VE
    parser.add_argument('--use_bias', type=int,default=1,help="whether to use bias term; True 1 False 0")
    parser.add_argument('--VE_len', type=int,default=0,help="variate embedding length; '0' means VE is disabled.")
    parser.add_argument('--ParamRatio', type=float,default=0,help="Model complexity of VE Pipeline; '0' means LoRA is disabled")
    parser.add_argument('--usePreprocess', type=str,default='softmax',help="The nonlinearity applied to VE; ['none', 'softmax']")

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    fix_seed_list = range(2021, 2021+20)

    if args.cut_freq == 0:
        args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10
        
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.debug:
        print('>>>>>>>Debug mode. Setting train_epochs=1 and itr=1 >>>>>>>>>>>>>>>>>>>>>>>>>>')
        args.train_epochs = 1
        args.itr = 1
    else:
        log_config(f"{args.is_training}_{model_log(args)}.txt", args)

    Exp = Exp_Long_Term_Forecast

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    rsts = []
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_seed{}'.format(
                        model_log(args), 
                        ii, 
                        fix_seed_list[ii]
            )
            print(f"args.use_gpu: {args.use_gpu}")
            print('Args in experiment:')
            print(args)

            random.seed(fix_seed_list[ii])
            torch.manual_seed(fix_seed_list[ii])
            np.random.seed(fix_seed_list[ii])

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if args.debug:
                print('>>>>>>>Debuging. Skip test : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            else:
                rst = exp.test(setting)
                rsts.append(rst)
            torch.cuda.empty_cache()
            
        if args.itr > 1:
            exp.log_mean(rsts)
    else:
        ii = args.test_ii
        # setting record of experiments
        setting = '{}_{}_seed{}'.format(
                    model_log(args), 
                    ii, 
                    fix_seed_list[ii]
        )
        if args.debug==0:
            log_config(f"{args.is_training}_{setting}.txt", args)
        print(f"args.use_gpu: {args.use_gpu}")
        print('Args in experiment:')
        print(args)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        rst = exp.test(setting)
        torch.cuda.empty_cache()
