import pickle
import logging
import torch
import torch.nn as nn
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import set_seed
import argparse


set_seed(42)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class SegmentDataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.iter_order = self.pointer = None

        datas_steps = []
        labels = []
        min_latency = []
        for data_idx, data in enumerate(dataset):
            datas_step, label, min_lat = data
            for step in datas_step:
                step.insert(0, 1 - sum(step[:11]))

            datas_steps.append(datas_step)
            labels.append(label)
            min_latency.append(min_lat)

        self.datas_steps = torch.FloatTensor(datas_steps)
        self.labels = torch.FloatTensor(labels)
        self.min_latency = torch.FloatTensor(min_latency)

        self.number = len(self.datas_steps)

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.randperm(self.number)
        else:
            self.iter_order = torch.arange(self.number)
        self.pointer = 0

        return self

    def __next__(self):
        if self.pointer >= self.number:
            raise StopIteration

        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)

    def _fetch_indices(self, indices):
        batch_datas_steps = self.datas_steps[indices]
        batch_datas_steps = nn.utils.rnn.pad_sequence(
            batch_datas_steps, batch_first=True)
        return batch_datas_steps[:, :-1, :], batch_datas_steps[:, 1:, :]

    def __len__(self):
        return self.number



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='/root/tenset/scripts/tlp_dataset_platinum_8272_100_train_and_val.pkl')
    parser.add_argument("--train_size_per_gpu", type=int, default=896)
    parser.add_argument("--block_size", type=int, default=24)  # seq_len - 1
    parser.add_argument("--one_hot_len", type=int, default=12)  # one_hot_len + 1
    parser.add_argument("--type_loss_factor", type=int, default=10)  
    parser.add_argument("--emb_size", type=int, default=23)  # emb_size + 1
    parser.add_argument("--n_epoch", type=int, default=200)
    args = parser.parse_args()


    with open(args.dataset, 'rb') as f:
        file_vecs = pickle.load(f)


    train_loader = SegmentDataLoader(
        file_vecs, args.train_size_per_gpu * torch.cuda.device_count(), True)
    del file_vecs
    vocab_size = 42336

    mconf = GPTConfig(vocab_size, args.block_size,
                    embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                    n_layer=12, n_head=8, n_embd=512)
    mconf.one_hot_len = args.one_hot_len
    mconf.type_loss_factor = args.type_loss_factor
    mconf.emb_size = args.emb_size
    model = GPT(mconf)

    tokens_per_epoch = len(train_loader) * args.block_size
    train_epochs = args.n_epoch

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=train_epochs, batch_size=-1, learning_rate=3e-5,
                        betas=(0.9, 0.95), weight_decay=0,
                        lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=20*tokens_per_epoch,
                        ckpt_path='gpt_model.pt',
                        num_workers=-1)
    trainer = Trainer(model, None, None, tconf, train_loader)
    trainer.train()
