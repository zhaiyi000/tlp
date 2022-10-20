import pickle
import numpy as np
import torch
import argparse
from tlp_train import *
from mtl_tlp_train import MTLTLPAttentionModule


top_ks = [1, 5, 10, 20]


def pred_a_dataset(datas, task_pred_dict, model):

    datas_new = []
    for data_idx, data in enumerate([datas]):
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        datas_new.extend(line_vecs)

    if isinstance(model, BertModule):
        test_loader = BertSegmentDataLoader(datas_new, 512, False)
    elif isinstance(model, GPTModule):
        test_loader = GPTSegmentDataLoader(datas_new, 512, False)
    else:
        test_loader = SegmentDataLoader(datas_new, 4000, False)
    assert test_loader.min_latency.min() == test_loader.min_latency.max()

    preds_all = []
    labels_all = []

    for batch_datas_steps, batch_labels in test_loader:
        batch_datas_steps = batch_datas_steps.to(device)
        preds = model(batch_datas_steps)
        if isinstance(preds, list) and len(preds) > 1:
            preds = preds[0]
        preds_all.append(preds.detach().cpu())
        labels_all.append(batch_labels.detach().cpu())

    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    task_pred_dict[workloadkey] = (preds_all.detach().cpu().numpy(
    ), test_loader.min_latency.min().numpy(), labels_all.numpy())


def eval_model(model_file):

    with open(model_file, 'rb') as f:
        model = pickle.load(f).module.to(device)
    model.eval()
    task_pred_dict = {}

    pred_a_dataset_dict = {}
    for data_idx, data in enumerate(test_datasets):
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        pred_a_dataset_dict[workloadkey] = data

    files = [
        'dataset/network_info/((resnet_50,[(1,3,224,224)]),%s).task.pkl' % args.platform,
        'dataset/network_info/((mobilenet_v2,[(1,3,224,224)]),%s).task.pkl' % args.platform,
        'dataset/network_info/((resnext_50,[(1,3,224,224)]),%s).task.pkl' % args.platform,
        'dataset/network_info/((bert_base,[(1,128)]),%s).task.pkl' % args.platform,
        'dataset/network_info/((bert_tiny,[(1,128)]),%s).task.pkl' % args.platform
    ]
    top_1_total = []
    top_5_total = []
    top_10_total = []
    top_20_total = []
    best_latency_total_list = []
    best_latency_total = 0
    top1_total = 0
    top5_total = 0
    top10_total = 0
    top20_total = 0
    for file in files:
        tasks, task_weights = pickle.load(open(file, "rb"))
        latencies = [0] * len(top_ks)
        best_latency = 0

        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                print('error task.workload_key not in pred_a_dataset_dict')
                continue
            pred_a_dataset(
                pred_a_dataset_dict[task.workload_key], task_pred_dict, model)
            preds, min_latency, labels = task_pred_dict[task.workload_key]

            real_values = labels[np.argsort(-preds)]
            real_latency = min_latency / np.maximum(real_values, 1e-5)

            for i, top_k in enumerate(top_ks):
                latencies[i] += np.min(real_latency[:top_k]) * weight
            best_latency += min_latency * weight

        top_1_total.append(best_latency/latencies[0])
        print(f"top 1 score: {best_latency/latencies[0]}")
        top_5_total.append(best_latency / latencies[1])
        print(f"top 5 score: {best_latency / latencies[1]}")

        best_latency_total_list.append(best_latency)
        best_latency_total += best_latency
        top1_total += latencies[0]
        top5_total += latencies[1]
        top10_total += latencies[2]
        top20_total += latencies[3]


    print(f"average top 1 score is {best_latency_total / top1_total}")
    top_1_total.append(best_latency_total / top1_total)
    print(f"average top 5 score is {best_latency_total / top5_total}")
    top_5_total.append(best_latency_total / top5_total)
    print(f"average top 10 score is {best_latency_total / top10_total}")
    top_10_total.append(best_latency_total / top1_total)
    print(f"average top 20 score is {best_latency_total / top20_total}")
    top_20_total.append(best_latency_total / top5_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default='cuda:0')
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--test_dataset_name", type=str, default='tlp_dataset_platinum_8272_2308_test.pkl')
    parser.add_argument("--load_name", type=str, default='tlp_i7/tlp_model_0.pkl') 
    parser.add_argument("--platform", type=str, default='llvm')  # or cuda
    args = parser.parse_args()
    print(args)

    device = args.cuda

    with open(args.test_dataset_name, 'rb') as f:
        test_datasets = pickle.load(f)

    eval_model(args.load_name)
