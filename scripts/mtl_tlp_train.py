import os
import pickle
import torch
import time
import numpy as np
import random
from torch import nn
from torch import optim
import argparse


class MTLTLPAttentionModule(nn.Module):  
    def __init__(self):
        super().__init__()
        self.fea_size = args.fea_size
        self.step_size = args.step_size
        self.res_block_cnt = args.res_block_cnt
        self.mtl_head_list = args.mtl_head_list

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

        self.mtl_heads = []
        for i in self.mtl_head_list:
            self.mtl_heads.append(nn.Sequential(
                nn.Linear(hidden_dim_1, out_dim[0]),
                nn.ReLU(),
                nn.Linear(out_dim[0], out_dim[1]),
                nn.ReLU(),
                nn.Linear(out_dim[1], out_dim[2]),
                nn.ReLU(),
                nn.Linear(out_dim[2], out_dim[3]),
            ))
        self.mtl_heads = nn.Sequential(*self.mtl_heads)
        

    def forward(self, batch_datas_steps):

        batch_datas_steps = batch_datas_steps[:, :self.step_size, :self.fea_size]
        encoder_output = self.encoder(batch_datas_steps)

        encoder_output = encoder_output.transpose(0, 1)
        output = self.attention(encoder_output, encoder_output, encoder_output)[0] + encoder_output

        for l in self.l_list:
            output = l(output) + output

        outputs = []
        for head in self.mtl_heads:
            outputs.append(head(output).sum(0).squeeze())

        return outputs


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
            is_train,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.iter_order = self.pointer = None
        self.mtl_head_list = args.mtl_head_list
        self.data_cnt = args.data_cnt

        datas_steps = []
        labels_list = [[] for i in self.mtl_head_list]
        min_latency_list = [[] for i in self.mtl_head_list]
        for data_idx, data in enumerate(dataset):
            datas_steps.append(data[0])
            data = data[1:]
            for i_index, i in enumerate(self.mtl_head_list):
                labels_list[i_index].append(data[2 * i])
                min_latency_list[i_index].append(data[2 * i + 1])

        self.datas_steps = torch.FloatTensor(datas_steps)
        self.labels_list = [torch.FloatTensor(labels) for labels in labels_list]
        self.min_latency_list = [torch.FloatTensor(min_latency) for min_latency in min_latency_list]

        self.number = len(self.datas_steps)


        print('mask...')
        data_cnt = int(self.data_cnt * 1000 * (0.9 if is_train else 0.1))
        all_index = torch.ones(self.number, dtype=torch.bool)
        remain_index = torch.randperm(self.number)[:data_cnt]
        all_index[remain_index] = False
        self.labels_list[0][all_index] = -100
        
        

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
        batch_datas_steps = nn.utils.rnn.pad_sequence(batch_datas_steps, batch_first=True)
        batch_labels_list = [i[indices] for i in self.labels_list]

        return (batch_datas_steps, batch_labels_list)

    def __len__(self):
        return self.number


def load_datas(datasets_global):

    datasets = np.array(datasets_global, dtype=object)
    train_len = int(len(datasets) * 0.9)

    perm = np.random.permutation(len(datasets))
    train_indices, val_indices = perm[:train_len], perm[train_len:]

    train_datas, val_datas = datasets[train_indices], datasets[val_indices]

    n_gpu = torch.cuda.device_count()
    train_dataloader = SegmentDataLoader(train_datas, args.train_size_per_gpu*n_gpu, shuffle=True, is_train=True)
    val_dataloader = SegmentDataLoader(val_datas, args.val_size_per_gpu*n_gpu, shuffle=False, is_train=False)

    return train_dataloader, val_dataloader


def validate(model, valid_loader, loss_func, device):
    model.eval()
    valid_losses_list = [[] for i in args.mtl_head_list]

    for batch_datas_steps, batch_labels_list in valid_loader:
        batch_datas_steps = batch_datas_steps.to(device)
        batch_labels_list = [batch_labels.to(device) for batch_labels in batch_labels_list]

        preds_list = model(batch_datas_steps)

        target_index = batch_labels_list[0] != -100
        preds_list[0] = preds_list[0][target_index]
        batch_labels_list[0] = batch_labels_list[0][target_index]

        for valid_losses, preds, batch_labels in zip(valid_losses_list, preds_list, batch_labels_list):
            valid_losses.append(loss_func(preds, batch_labels).item())

    return [np.sum(valid_losses) for valid_losses in valid_losses_list]


def train(train_loader, val_dataloader, device):
    # n_epoch = 50
    args.hidden_dim = [64, 128, 256, 256]
    args.out_dim = [256, 128, 64, 1]
    net = MTLTLPAttentionModule().to(device)
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

    print('start train...')
    print(len(train_loader), len(val_dataloader))
    for epoch in range(n_epoch):
        tic = time.time()

        net.train()
        train_loss_list = [0] * len(args.mtl_head_list)
        for batch, (batch_datas_steps, batch_labels_list) in enumerate(train_loader):
            batch_datas_steps = batch_datas_steps.to(device)
            batch_labels_list = [i.to(device) for i in batch_labels_list]

            optimizer.zero_grad()
            preds_list = net(batch_datas_steps)

            target_index = batch_labels_list[0] != -100
            preds_list[0] = preds_list[0][target_index]
            batch_labels_list[0] = batch_labels_list[0][target_index]


            loss_list = [loss_func(preds, labels) for preds, labels in zip(preds_list, batch_labels_list)]
            loss = sum(loss_list)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            train_loss_list = [train_loss + loss.item() for train_loss, loss in zip(train_loss_list, loss_list)]
        lr_scheduler.step()

        train_time = time.time() - tic

        if epoch % 5 == 0 or epoch == n_epoch - 1 or True:

            valid_losses_list = validate(net, val_dataloader,
                                  loss_func, device=device)

            loss_msg = ''.join(["Train Loss %d: %.4f\tValid Loss %d: %.4f\n" % (i, train_loss, i, valid_loss) for i, (train_loss, valid_loss) in enumerate(zip(train_loss_list, valid_losses_list))])
            print("Epoch: %d\tBatch: %d\tTrain Speed: %.0f\n%s" % (
                epoch, batch, len(train_loader) / train_time, loss_msg))

        model_save_file_name = '%s/mtl_tlp_model_%d.pkl' % (args.save_folder, epoch)
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
    parser.add_argument("--dataset", type=str, default='mtl_tlp_dataset_llvm_100.pkl')
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--rank_mse", type=str, default='rank')
    parser.add_argument("--optimizer", type=str, default='default')
    parser.add_argument("--attention_head", type=int, default=8)
    parser.add_argument("--attention_class", type=str, default='default')
    parser.add_argument("--step_size", type=int, default=25)
    parser.add_argument("--fea_size", type=int, default=22)
    parser.add_argument("--res_block_cnt", type=int, default=2)

    parser.add_argument("--train_size_per_gpu", type=int, default=1024)
    parser.add_argument("--val_size_per_gpu", type=int, default=1024)
    parser.add_argument("--n_epoch", type=int, default=50)

    parser.add_argument("--mtl_head_list", type=str, default='0,1,2,3,4')  # the first head corresponds to the target platform
    parser.add_argument("--data_cnt", type=int, default=500)  # data_cnt * 1000

    args = parser.parse_args()
    args.mtl_head_list = [int(i) for i in args.mtl_head_list.strip().split(',')]

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
