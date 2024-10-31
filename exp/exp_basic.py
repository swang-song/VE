import os
import torch
from models import FITS, Linear, DLinear, PatchTST, iTransformer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'FITS': FITS,
            'Linear': Linear,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        self.Time_Train_pEpoch = 0
        self.Time_Infer_pSample = 0
        self.Mem_Train = 0
        self.Mem_Infer = 0
        self.Macs = 0
        self.Params = 0

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
