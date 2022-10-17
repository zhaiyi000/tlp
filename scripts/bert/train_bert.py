import argparse
import pickle
from transformers import RobertaConfig, RobertaForMaskedLM, AdamW
import torch
from torch import nn
from tqdm.auto import tqdm
import math

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
        # datas_step_seg = []
        labels = []
        min_latency = []
        # print('len_dataset', len(dataset))
        for data_idx, data in enumerate(dataset):

            datas_step, label, min_lat = data
            datas_steps.append(datas_step)
            labels.append(label)
            min_latency.append(min_lat)



        datas_steps = torch.LongTensor(datas_steps)
        self.labels = datas_steps.detach().clone()

        rand = torch.rand(datas_steps.shape)
        mask_arr = (rand < .15) * (datas_steps != 1) * (datas_steps != 2) * (datas_steps != 3)
        for i in range(datas_steps.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(torch.nonzero(mask_arr[i])).tolist()
            # mask datas_steps
            datas_steps[i, selection] = 0  # our custom [MASK] token == 0

        # self.labels = torch.FloatTensor(labels)
        # self.min_latency = torch.FloatTensor(min_latency)
        self.datas_steps = datas_steps
        self.mask_arr = mask_arr


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

        input_ids = self.datas_steps[indices]
        mask = self.mask_arr[indices]
        labels = self.labels[indices]

        return {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}

    def __len__(self):
        return self.number

def save_checkpoint(model, ckpt_path, epoch):
    # DataParallel wrappers keep raw model object in .module attribute
    raw_model = model.module if hasattr(model, "module") else model
    save_name = ckpt_path[:-3] + f'_{epoch}' + ckpt_path[-3:]
    print("saving %s", save_name)
    torch.save(raw_model.state_dict(), save_name)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default='../tlp_dataset_bert_platinum_8272_100_train_and_val.pkl')
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--decay_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=7e-4)
    args = parser.parse_args()
    print(args)

    ########### load data
    print('load data...')
    with open(args.datasets, 'rb') as f:
        datasets = pickle.load(f)
    cuda_count = torch.cuda.device_count()
    dataloader = BertSegmentDataLoader(datasets, args.batch_size * cuda_count, True)
    del datasets
    print('load data done')


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
    # checkpoint = torch.load('/root/tenset/scripts/train_bert/bertmodel.pkl2')
    # model.load_state_dict(checkpoint)
    model = nn.DataParallel(model).to(torch.cuda.current_device())

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ########### train
    epochs = 200
    batch_cnt = math.ceil(len(dataloader) / (args.batch_size * cuda_count))
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(dataloader, leave=True)
        for batch_idx, batch in enumerate(loop):
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            # extract loss
            loss = outputs.loss.sum()
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            # loop.set_description(f'Epoch {epoch}')

            if epoch == 0:
                # linear warmup
                lr_mult = (batch_idx + 1) / batch_cnt
            elif epoch <= args.decay_epoch:
                # cosine learning rate decay
                progress = ((epoch - 1) * batch_cnt + (batch_idx + 1)) / (args.decay_epoch * batch_cnt)
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            else:
                lr_mult = 0.1
            lr = args.lr * lr_mult
            for param_group in optim.param_groups:
                param_group['lr'] = lr


            loop.set_description(f"epoch {epoch} train loss {loss.item():.5f}. lr {lr:e}")
            # loop.set_postfix(loss=loss.item())


        save_checkpoint(model, 'bertmodel.pt', epoch)


        