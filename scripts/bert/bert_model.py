import torch
from transformers import RobertaConfig, RobertaForMaskedLM, AdamW
from torch import nn


class BertModel:
    def __init__(self, self_sup_model) -> None:
        
        ########### RobertaConfig
        config = RobertaConfig(
            vocab_size=42335 + 3,  # we align this to the tokenizer vocab_size
            max_position_embeddings=25 + 2 + 1,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1
        )

        ########### net
        model = RobertaForMaskedLM(config)
        print('load self_sup_model', self_sup_model)
        if len(self_sup_model) > 0:
            checkpoint = torch.load(self_sup_model)
            model.load_state_dict(checkpoint)


        model.lm_head = nn.Identity()

        self.model = model
        
        
        