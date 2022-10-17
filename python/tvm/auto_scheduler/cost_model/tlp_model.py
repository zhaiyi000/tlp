import pickle
import torch
import logging
from .cost_model import PythonBasedModel
from tvm.auto_scheduler.feature import get_per_store_features_from_states_tlp, get_per_store_features_from_states_tlp_init


logger = logging.getLogger("auto_scheduler")


class TLPModel(PythonBasedModel):
    """The wrapper of MLPModelInternal. So we can use it in end-to-end search."""
    def __init__(self, target, max_line_len=25, max_vec_len=22):
        super().__init__()
        
        get_per_store_features_from_states_tlp_init('cpu' if 'llvm' in str(target) else 'gpu')
        self.max_line_len = max_line_len
        self.max_vec_len = max_vec_len


    def update(self, inputs, results):
        pass


    def predict(self, task, states):
        file_vec = []
        file_vec = get_per_store_features_from_states_tlp(
            states, task, self.max_vec_len, self.max_line_len)
        file_vec = torch.FloatTensor(file_vec).to('cuda:0')
        with torch.no_grad():
            ret = self.model(file_vec)
        if isinstance(ret, list) and len(ret) > 0:
            ret = ret[0]
        return ret.cpu().detach().numpy()


    def load(self, file_name: str):
        with open(file_name, 'rb') as f:
            self.model = pickle.load(f).module.to('cuda:0')
        self.model.eval()
