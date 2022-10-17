import torch
from .mingpt.model import GPT, GPTConfig
from torch import nn

class gpt_args:
    pass

class GPUModel:
    def __init__(self, self_sup_model) -> None:
        vocab_size = 42336
        args = gpt_args()
        args.block_size = 24
        args.one_hot_len = 12
        args.type_loss_factor = 10
        args.emb_size = 23

        mconf = GPTConfig(vocab_size, args.block_size,
                    embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                    n_layer=12, n_head=8, n_embd=512)
        mconf.one_hot_len = args.one_hot_len
        mconf.type_loss_factor = args.type_loss_factor
        mconf.emb_size = args.emb_size
        model = GPT(mconf)
        print('load self_sup_model', self_sup_model)
        if len(self_sup_model) > 0:
            checkpoint = torch.load(self_sup_model)
            model.load_state_dict(checkpoint)

        self.model = model