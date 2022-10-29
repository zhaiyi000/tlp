import os
import pickle
import torch
import time
import numpy as np
import random
import math
from torch import nn
from torch import optim
import argparse


def get_cosine_schedule_with_warmup(
	optimizer: optim.Optimizer,
	num_warmup_steps: int,
	num_training_steps: int,
	num_cycles: float = 0.5,
	last_epoch: int = -1,
):
	"""
	Create a schedule with a learning rate that decreases following the values of the cosine function between the
	initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
	initial lr set in the optimizer.

	Args:
		optimizer (:class:`~torch.optim.Optimizer`):
		The optimizer for which to schedule the learning rate.
		num_warmup_steps (:obj:`int`):
		The number of steps for the warmup phase.
		num_training_steps (:obj:`int`):
		The total number of training steps.
		num_cycles (:obj:`float`, `optional`, defaults to 0.5):
		The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
		following a half-cosine).
		last_epoch (:obj:`int`, `optional`, defaults to -1):
		The index of the last epoch when resuming training.

	Return:
		:obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
	"""
	def lr_lambda(current_step):
		# Warmup
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		# decadence
		progress = float(current_step - num_warmup_steps) / float(
			max(1, num_training_steps - num_warmup_steps)
		)
		return max(
			0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
		)

	return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class AttentionModule(nn.Module):  
    def __init__(self):
        super().__init__()
        self.fea_size = args.fea_size
        self.step_size = args.step_size
        self.res_block_cnt = args.res_block_cnt

        in_dim = self.fea_size
        hidden_dim = args.hidden_dim
        out_dim = args.out_dim
        hidden_dim_1 = hidden_dim[-1]

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim_1, args.attention_head)

        self.l0 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
        )
        self.l_list = []
        for i in range(self.res_block_cnt):
            self.l_list.append(nn.Sequential(
                nn.Linear(hidden_dim_1, hidden_dim_1), 
                nn.ReLU()
            ))
        self.l_list = nn.Sequential(*self.l_list)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim_1, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
            nn.ReLU(),
            nn.Linear(out_dim[2], out_dim[3]),
        )
        

    def forward(self, batch_datas_steps):

        batch_datas_steps = batch_datas_steps[:, :self.step_size, :self.fea_size]
        encoder_output = self.encoder(batch_datas_steps)

        encoder_output = encoder_output.transpose(0, 1)
        output = self.attention(encoder_output, encoder_output, encoder_output)[0] + encoder_output

        for l in self.l_list:
            output = l(output) + output

        output = self.decoder(output).sum(0)

        return output.squeeze()


class TransformerEncoderLayerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fea_size = args.fea_size
        self.step_size = args.step_size

        in_dim = self.fea_size
        hidden_dim = [64, 128, 256, 256]
        out_dim = [256, 128, 64, 1]
        hidden_dim_1 = hidden_dim[-1]

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(),
        )

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim_1, dim_feedforward=256, nhead=args.attention_head
        )

        self.l0 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
        )
        self.l1 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim_1, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
            nn.ReLU(),
            nn.Linear(out_dim[2], out_dim[3]),
        )

    def forward(self, batch_datas_steps):
        batch_datas_steps = batch_datas_steps[:,
                                              :self.step_size, :self.fea_size]
        encoder_output = self.encoder(batch_datas_steps)
        encoder_output = encoder_output.transpose(0, 1)
        output = self.transformer_encoder_layer(encoder_output)
        output = self.l0(output) + output
        output = self.l1(output) + output
        output = self.decoder(output).sum(0)
        return output.squeeze()



class TransformerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fea_size = args.fea_size
        self.step_size = args.step_size

        in_dim = self.fea_size
        hidden_dim = [64, 128, 256, 256]
        out_dim = [256, 128, 64, 1]
        hidden_dim_1 = hidden_dim[-1]

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(),
        )

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim_1, dim_feedforward=256, nhead=args.attention_head
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=2)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim_1, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
            nn.ReLU(),
            nn.Linear(out_dim[2], out_dim[3]),
        )

    def forward(self, batch_datas_steps):

        batch_datas_steps = batch_datas_steps[:, :self.step_size, :self.fea_size]
        encoder_output = self.encoder(batch_datas_steps)

        encoder_output = encoder_output.transpose(0, 1)
        output = self.transformer(encoder_output)

        output = self.decoder(output).sum(0)

        return output.squeeze()


class LSTMModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fea_size = args.fea_size
        self.step_size = args.step_size

        lstm_linar_in_dim = self.fea_size
        lstm_linar_hidden_dim = [64, 128, 256, 256]
        out_dim = [256, 128, 64, 1]
        hidden_dim = lstm_linar_hidden_dim[-1]

        self.lstm_linar_encoder = nn.Sequential(
            nn.Linear(lstm_linar_in_dim, lstm_linar_hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(lstm_linar_hidden_dim[0], lstm_linar_hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(lstm_linar_hidden_dim[1], lstm_linar_hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(lstm_linar_hidden_dim[2], lstm_linar_hidden_dim[3]),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            lstm_linar_hidden_dim[-1], lstm_linar_hidden_dim[-1])


        self.l0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.l1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
            nn.ReLU(),
            nn.Linear(out_dim[2], out_dim[3]),
        )

    def forward(self, batch_datas_steps):

        batch_datas_steps = batch_datas_steps[:, :self.step_size, :self.fea_size]

        batch_datas_steps = batch_datas_steps.transpose(0, 1)
        lstm_output = self.lstm_linar_encoder(batch_datas_steps)
        _, (h, c) = self.lstm(lstm_output)
        lstm_output = h[0]

        output = lstm_output

        output = self.l0(output) + output
        output = self.l1(output) + output

        output = self.decoder(output)

        return output.squeeze()


class GPTModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_sup_model = args.self_sup_model

        from minGPT.gpt_model import GPUModel
        self.gpt = GPUModel(self.self_sup_model).model
        out_dim = [23, 256, 1]

        self.decoder = nn.Sequential(
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
        )

    def forward(self, batch_datas_steps):

        output = self.gpt(batch_datas_steps)[0].mean(1)
        output = self.decoder(output)
        return output.squeeze()


class BertModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_sup_model = args.self_sup_model

        from bert.bert_model import BertModel
        self.bert = BertModel(self.self_sup_model).model

        self.decode = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, batch_datas_steps):

        output = self.bert(batch_datas_steps).logits[:, 0, :]
        output = self.decode(output)
        return output.squeeze()


class LambdaRankLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
            G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10., sigma=1.):
        device = self.device
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :,
                                          None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros(
            (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(
            ((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (
            y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(
            sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss


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
            data = data[:3]
            datas_step, label, min_lat = data
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
        batch_labels = self.labels[indices]

        return (batch_datas_steps, batch_labels)

    def __len__(self):
        return self.number


class GPTSegmentDataLoader:
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
            datas_step_new = []
            for step in datas_step:
                step_new = step.copy()
                step_new.insert(0, 1 - sum(step_new[:11]))
                datas_step_new.append(step_new)

            datas_steps.append(datas_step_new)
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
        batch_labels = self.labels[indices]
        return (batch_datas_steps[:, :-1, :], batch_labels)

    def __len__(self):
        return self.number


class BertSegmentDataLoader:
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
            datas_steps.append(datas_step)
            labels.append(label)
            min_latency.append(min_lat)

        self.datas_steps = torch.LongTensor(datas_steps)
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
        batch_labels = self.labels[indices]
        return batch_datas_steps, batch_labels

    def __len__(self):
        return self.number



def load_datas(datasets_global):

    datasets = np.array(datasets_global, dtype=object)
    if args.data_cnt > 0:
        train_len = int(args.data_cnt * 1000 * 0.9)

        perm = np.random.permutation(len(datasets))
        train_indices, val_indices = perm[:train_len], perm[train_len:args.data_cnt * 1000]
    else:
        train_len = int(len(datasets) * 0.9)

        perm = np.random.permutation(len(datasets))
        train_indices, val_indices = perm[:train_len], perm[train_len:]

    train_datas, val_datas = datasets[train_indices], datasets[val_indices]

    n_gpu = torch.cuda.device_count()
    if args.attention_class == 'gpt':
        train_dataloader = GPTSegmentDataLoader(train_datas, args.train_size_per_gpu*n_gpu, True)
        val_dataloader = GPTSegmentDataLoader(val_datas, args.train_size_per_gpu*n_gpu, False)
    elif args.attention_class == 'bert':
        train_dataloader = BertSegmentDataLoader(train_datas, args.train_size_per_gpu*n_gpu, True)
        val_dataloader = BertSegmentDataLoader(val_datas, args.train_size_per_gpu*n_gpu, False)
    else:
        train_dataloader = SegmentDataLoader(
            train_datas, args.train_size_per_gpu*n_gpu, True)
        val_dataloader = SegmentDataLoader(
            val_datas, args.val_size_per_gpu*n_gpu, False)

    return train_dataloader, val_dataloader


def validate(model, valid_loader, loss_func, device):
    model.eval()
    valid_losses = []

    for batch_datas_steps, batch_labels in valid_loader:
        batch_datas_steps = batch_datas_steps.to(device)
        batch_labels = batch_labels.to(device)

        preds = model(batch_datas_steps)
        valid_losses.append(loss_func(preds, batch_labels).item())

    return np.sum(valid_losses)


def train(train_loader, val_dataloader, device):
    # n_epoch = 50
    if args.attention_class == 'default':
        args.hidden_dim = [64, 128, 256, 256]
        args.out_dim = [256, 128, 64, 1]
        net = AttentionModule().to(device)
        net = torch.nn.DataParallel(net).to(torch.cuda.current_device())
    elif args.attention_class == 'transformer':
        print('TransformerModule')
        net = TransformerModule().to(device)
        net = torch.nn.DataParallel(net).to(torch.cuda.current_device())
    elif args.attention_class == 'attention_encoder_layer':
        print('TransformerEncoderLayerModule')
        net = TransformerEncoderLayerModule().to(device)
        net = torch.nn.DataParallel(net).to(torch.cuda.current_device())
    elif args.attention_class == 'lstm':
        print('LSTMModule')
        net = LSTMModule().to(device)
        net = torch.nn.DataParallel(net).to(torch.cuda.current_device())
    elif args.attention_class == 'gpt':
        print('GPTModule')
        net = GPTModule().to(device)
        net = torch.nn.DataParallel(net).to(torch.cuda.current_device())
    elif args.attention_class == 'bert':
        print('BertModule')
        net = BertModule().to(device)
        net = torch.nn.DataParallel(net).to(torch.cuda.current_device())
    elif args.attention_class == 'attention_512':
        args.hidden_dim = [64, 128, 256, 512]
        args.out_dim = [256, 128, 64, 1]
        print('Attention512Module')
        net = AttentionModule().to(device)
        net = torch.nn.DataParallel(net).to(torch.cuda.current_device())
    elif args.attention_class == 'attention_768':
        args.hidden_dim = [64, 256, 512, 768]
        args.out_dim = [512, 256, 128, 1]
        print('Attention768Module')
        net = AttentionModule().to(device)
        net = torch.nn.DataParallel(net).to(torch.cuda.current_device())
    elif args.attention_class == 'attention_1024':
        args.hidden_dim = [64, 256, 512, 1024]
        args.out_dim = [512, 256, 128, 1]
        print('Attention1024Module')
        net = AttentionModule().to(device)
        net = torch.nn.DataParallel(net).to(torch.cuda.current_device())

    if args.rank_mse == 'rank':
        loss_func = LambdaRankLoss(device)
    else:
        loss_func = nn.MSELoss()

    n_epoch = args.n_epoch
    if args.optimizer == 'default':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=1)
    elif args.optimizer == 'decrease_per_17_0.8':
        print('optimizer', 'decrease_per_17_0.8')
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=0.8)
    elif args.optimizer == 'decrease_per_17_0.5':
        print('optimizer', 'decrease_per_17_0.5')
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=0.5)
    elif args.optimizer == 'decrease_per_12_0.5':
        print('optimizer', 'decrease_per_12_0.5')
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 4, gamma=0.5)
    elif args.optimizer == 'decrease_per_10_0.5':
        print('optimizer', 'decrease_per_10_0.5')
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 5, gamma=0.5)
    elif args.optimizer == 'decrease_per_17_0.5_no_decay':
        print('optimizer', 'decrease_per_17_0.5')
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=0.5)

    train_loss = None
    print('start train...')
    print(len(train_loader), len(val_dataloader))
    for epoch in range(n_epoch):
        tic = time.time()

        net.train()
        train_loss = 0
        for batch, (batch_datas_steps, batch_labels) in enumerate(train_loader):
            batch_datas_steps = batch_datas_steps.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            loss = loss_func(net(batch_datas_steps), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
        lr_scheduler.step()

        train_time = time.time() - tic

        if epoch % 5 == 0 or epoch == n_epoch - 1 or True:

            valid_loss = validate(net, val_dataloader,
                                  loss_func, device=device)
            loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (
                train_loss, valid_loss)
            print("Epoch: %d\tBatch: %d\t%s\tTrain Speed: %.0f" % (
                epoch, batch, loss_msg, len(train_loader) / train_time,))

        model_save_file_name = '%s/tlp_model_%d.pkl' % (args.save_folder, epoch)
        with open(model_save_file_name, 'wb') as f:
            pickle.dump(net.cpu(), f)
        net = net.to(device)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default='.')
    parser.add_argument("--cuda", type=str, default='cuda:0')
    parser.add_argument("--dataset", type=str, default='tlp_dataset_platinum_8272_100_train_and_val.pkl')
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--rank_mse", type=str, default='rank')
    parser.add_argument("--optimizer", type=str, default='default')
    parser.add_argument("--attention_head", type=int, default=8)
    parser.add_argument("--attention_class", type=str, default='default')
    parser.add_argument("--step_size", type=int, default=25)
    parser.add_argument("--fea_size", type=int, default=22)
    parser.add_argument("--res_block_cnt", type=int, default=2)
    parser.add_argument("--self_sup_model", type=str, default='')
    parser.add_argument("--data_cnt", type=int, default=-1)  # data_cnt * 1000

    parser.add_argument("--train_size_per_gpu", type=int, default=1024)
    parser.add_argument("--val_size_per_gpu", type=int, default=1024)
    parser.add_argument("--n_epoch", type=int, default=50)
    args = parser.parse_args()

    print(args)

    if os.path.exists(args.save_folder) is False:
        print('create folder', args.save_folder)
        os.makedirs(args.save_folder, exist_ok=True)

    print('load data...')
    with open(args.dataset, 'rb') as f:
        datasets_global = pickle.load(f)
    print('load pkl done.')
    datas = load_datas(datasets_global)
    print('create dataloader done.')
    del datasets_global
    print('load data done.')
    train(*datas, device=args.cuda)
