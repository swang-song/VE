import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def get_WeightedParams(weights=None, params=None, is_complex=0):
    if is_complex:
        w_ = torch.view_as_real(params)
        w_real = w_[:,:,:,0]
        w_imag = w_[:,:,:,1]
        w_real = torch.einsum('ck,kio->cio', weights, w_real)
        w_imag = torch.einsum('ck,kio->cio', weights, w_imag)
        w = w_real + w_imag * 1j
    else:
        w = torch.einsum('ck,kio->cio', weights, params)
    return w


class VarEmbed(nn.Module):

    # Variate Embedding: Variate Embedding

    def __init__(self, C, k, din, dout, p=0, use_bias=1, useComplexLayer=0, usePreprocess='softmax', useExternalVE=0):
        """
        :param C: the number of channels
        :param k: the number of experts or the dimension of VE
        :param din: the dimension of the input in the linear layer
        :param dout: the dimension of the output in the linear layer
        :param p: the parameter size expansion ratio
        :param use_bias: [0, 1] if use_bias==1, add bias term
        :param useComplexLayer: [0, 1] if useComplexLayer==1, use complex-value weights
        :param usePreprocess: ['none', 'softmax']
        :param useExternalVE: [0, 1] For example, for DLinear, use one VE for both projection weights and this VE is initialized in DLinear
        """
        super(VarEmbed, self).__init__()
        assert k >= 1
        assert use_bias in [0, 1]
        assert useComplexLayer in [0, 1]
        assert usePreprocess in ['none', 'softmax']

        self.use_bias = use_bias
        self.useComplexLayer = useComplexLayer
        self.mdl_msg = ""   #for print(model)

        din = din + use_bias

        r = int((din * dout) * p * 1.0 / (k * (din + dout)))      # (di*do)/(k*(di+do))
        assert (p==0) or (r>0)
        self.useLoRA = False if r==0 else True

        param_dtype = torch.cfloat if self.useComplexLayer else torch.float32    # complex layer if use FFT

        self.useExternalVE = useExternalVE
        if self.useExternalVE==0:
            self.weights = nn.Parameter(torch.randn(C, k, dtype=torch.float32) / torch.sqrt(torch.tensor(k, dtype=torch.float32)))  # Our VE
            self.mdl_msg = self.mdl_msg + f"(weights): channel={C}, k={k}, dtype=torch.float32\n"
        if usePreprocess == 'softmax':
            self.preprocess = nn.Softmax(dim=-1)
        else:
            self.preprocess = nn.Identity()
        
                
        
        # LoRA of Weights
        if self.useLoRA:
            self.paramA = nn.Parameter(torch.randn(k, din, r, dtype=param_dtype) / torch.sqrt(torch.tensor(din, dtype=param_dtype)))
            self.paramB = nn.Parameter(torch.randn(k, r, dout, dtype=param_dtype) / torch.sqrt(torch.tensor(r, dtype=param_dtype)))
            self.mdl_msg = self.mdl_msg + f"(paramA): k={k}, din={din}, r={r}, dtype={param_dtype}\n"
            self.mdl_msg = self.mdl_msg + f"(paramB): k={k}, r={r}, dout={dout}, dtype={param_dtype}"
        else:
            self.params = nn.Parameter(torch.randn(k, din, dout, dtype=param_dtype) / torch.sqrt(torch.tensor(din, dtype=param_dtype)))
            self.mdl_msg = self.mdl_msg + f"(params): k={k}, din={din}, dout={dout}, dtype={param_dtype}"

    def forward(self, x, w=None):
        if self.use_bias:
            x = torch.concat([x, torch.ones(x.shape[0], x.shape[1], 1, device=x.device)], dim=-1)

        if self.useExternalVE==0:
            w = self.weights
        w = self.preprocess(w)
        if self.useLoRA:
            params = torch.einsum('kir,kro->kio', self.paramA, self.paramB)
        else:
            params = self.params
        w = get_WeightedParams(w, params, self.useComplexLayer)
        x = torch.einsum('bci,cio->bco', x, w)  # linear projection

        return x
    
    def extra_repr(self):
        return self.mdl_msg