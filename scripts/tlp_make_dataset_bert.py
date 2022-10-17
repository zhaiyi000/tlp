import torch
import os
import glob
import json
import pickle
from random import random
from tvm import auto_scheduler
from common import (load_and_register_tasks, get_measure_record_filename, get_to_measure_filename)
import threading
import multiprocessing
from tvm.tir.expr import FloatImm
import numpy as np
import random
import argparse


def handle_file(file_idx, file):
    print(file_idx)

    with open(file, 'r') as f:
        lines = f.read().strip().split('\n')

    inputs, outputs = auto_scheduler.RecordReader(file).read_lines()
    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task

    workloadkey_idx = workloadkey_to_index[task.workload_key[len('["'): len('["6b7583cf23c7c37d3212cad9d06e58c1')]]
    workload_args = [int(i) for i in task.workload_key[len('["6b7583cf23c7c37d3212cad9d06e58c1", '): -1].split(', ')]

    line_vecs = []

    min_cost = 1000000
    for line_idx, line in enumerate(lines):

        inp = json.loads(line)
        steps = inp['i'][1][1]

        step_vecs = []
        for st in steps:
            vec = []
            vec.extend(stepname_to_idx_one_hot[st[0]])

            for i, it in enumerate(st):
                if i == 0:
                    continue
                if isinstance(it, int):
                    vec.append(it)
                elif isinstance(it, list):
                    for ii in it:
                        assert isinstance(ii, int)
                        vec.append(ii)
                elif isinstance(it, str):
                    if st[0] == 'PR' and 'auto_unroll_max_step' in it:
                        vec.append(auto_unroll_max_step_to_idx[it])
                    elif st[0] == 'CHW':
                        vec.append(chw_dict[it])
                    elif st[0] == 'CHR' and it == 'shared':
                            vec.append(1)
                    else:
                        assert False
                else:
                    assert False

            assert len(vec) <= max_emb_size
            for i in range(len(vec), max_emb_size, 1):
                vec.append(0)

            vec = vec[:args.crop_emb_size]
            step_vecs.append(vec)

        # assert len(step_vecs) <= max_seq_len
        # vec = [0] * args.crop_emb_size
        # for i in range(len(step_vecs), max_seq_len, 1):
        #     step_vecs.append(vec.copy())
        step_vecs = step_vecs[:args.crop_seq_len]

        costs = [x.value for x in outputs[line_idx].costs if isinstance(x, FloatImm)]
        cost = np.mean(costs)
        line_vecs.append((step_vecs, cost))
        min_cost = min(min_cost, cost)
    line_vecs_new = []
    for line_vec in line_vecs:
        step_vecs, cost = line_vec
        score = min_cost / cost
        line_vecs_new.append((step_vecs, score, min_cost))
    line_vecs = line_vecs_new

    return (file, file_idx, workloadkey_idx, task.workload_key, workload_args, task.compute_dag.flop_ct, line_vecs)


def make_all_dataset(json_files_path):
    tasks = load_and_register_tasks()
    json_files = sorted(glob.glob(json_files_path + '/' + '*.json'))
    json_files = random.sample(json_files, args.files_cnt)

    multiprocessing_pool = multiprocessing.Pool()
    que_res_list = []
    # json_files = json_files[1471:]
    for file_idx, file in enumerate(json_files):
        que_res_list.append(multiprocessing_pool.apply_async(handle_file, args=(file_idx, file)))
        # handle_file(file_idx, file)

    multiprocessing_pool.close()
    multiprocessing_pool.join()

    file_vecs = []
    for que_res in que_res_list:
        file_vecs.append(que_res.get())

    return file_vecs


def to_token(file_vecs, step_vec_to_token_dict):
    file_vecs_new = []
    for file_vec_idx, file_vec in enumerate(file_vecs):
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = file_vec

        line_vecs_new = []
        for line_vec in line_vecs:
            step_vecs, score, min_cost = line_vec

            step_vecs_new = [1]
            for st in step_vecs:
                step_vecs_new.append(step_vec_to_token_dict[str(st)] + 3)
            step_vecs_new.append(2)

            for i in range(len(step_vecs_new), args.crop_seq_len + 2, 1):
                step_vecs_new.append(3)

            line_vec = (step_vecs_new, score, min_cost)
            line_vecs_new.append(line_vec)

        file_vec = (file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs_new)
        file_vecs_new.append(file_vec)

    return file_vecs_new


def aya_token(file_vecs):
    step_vec_to_token_set = set()

    for file_vec_idx, file_vec in enumerate(file_vecs):
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = file_vec

        for line_vec in line_vecs:
            step_vecs, score, min_cost = line_vec

            for st in step_vecs:
                step_vec_to_token_set.add(str(st))

    step_vec_to_token_list = sorted(list(step_vec_to_token_set))
    step_vec_to_token_dict = {}
    for st_idx, st in enumerate(step_vec_to_token_list):
        step_vec_to_token_dict[str(st)] = st_idx + 1

    return step_vec_to_token_dict
    

def split_dataset(file_vecs):
    train_and_val_dataset = []
    test_data = []

    for file_vec_idx, file_vec in enumerate(file_vecs):

        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = file_vec
        print(file_vec_idx, len(line_vecs))

        if workloadkey in hold_out_tasks_set:
            test_data.append(file_vec)
        else:
            train_and_val_dataset.append(file_vec)

    train_and_val_dataset_new = []
    for data_idx, data in enumerate(train_and_val_dataset):
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        train_and_val_dataset_new.extend(line_vecs)
        # for line_index, line in enumerate(line_vecs):
        #     train_and_val_dataset_new.append([file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line, line_index])


    with open(f'{args.save_name}_{args.files_cnt}_test.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    with open(f'{args.save_name}_{args.files_cnt}_train_and_val.pkl', 'wb') as f:
        pickle.dump(train_and_val_dataset_new, f)


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
    parser.add_argument("--json_files_path", type=str, default='dataset/measure_records/platinum-8272')
    parser.add_argument("--files_cnt", type=int, default=2308)
    parser.add_argument("--save_name", type=str, default='')
    parser.add_argument("--platform", type=str, default='llvm')  # or cuda
    parser.add_argument("--crop_seq_len", type=int, default=-1)
    parser.add_argument("--crop_emb_size", type=int, default=-1)
    args = parser.parse_args()

    if args.save_name == '':
        args.save_name = 'tlp_dataset_bert_' + os.path.basename(os.path.normpath(args.json_files_path)).replace('-', '_')

    hold_out_tasks = []
    files = [
        'dataset/network_info/((resnet_50,[(1,3,224,224)]),%s).task.pkl',
        'dataset/network_info/((mobilenet_v2,[(1,3,224,224)]),%s).task.pkl',
        'dataset/network_info/((resnext_50,[(1,3,224,224)]),%s).task.pkl',
        'dataset/network_info/((bert_base,[(1,128)]),%s).task.pkl',
        'dataset/network_info/((bert_tiny,[(1,128)]),%s).task.pkl'
    ]
    for file in files:
        tasks_part, task_weights = pickle.load(open(file % args.platform, "rb"))
        hold_out_tasks.extend(tasks_part)

    hold_out_tasks_set = set([task.workload_key for task in hold_out_tasks])

    with open('tlp_make_dataset_str_to_idx_%s.pkl' % args.platform, 'rb') as f:
        workloadkey_to_index, stepname_to_idx, auto_unroll_max_step_to_idx = pickle.load(f)

    stepname_to_idx_one_hot = {}
    for key, value in stepname_to_idx.items():
        one_hot = [0] * 11
        one_hot[stepname_to_idx[key]-1] = 1
        stepname_to_idx_one_hot[key] = one_hot

    chw_dict = {
        'local': 1,
        'shared': 2,
        'global': 3,
    }

    if args.platform == 'llvm':
        max_seq_len = 54
        max_emb_size = 40
    else:
        max_seq_len = 69
        max_emb_size = 49

    if args.crop_seq_len == -1 or args.crop_emb_size == -1:
        if args.platform == 'llvm':
            args.crop_seq_len = 25
            args.crop_emb_size = 22
        else:
            args.crop_seq_len = 40
            args.crop_emb_size = 20

    print(args)

    file_vecs = make_all_dataset(args.json_files_path)

    step_vec_to_token_dict = aya_token(file_vecs)
    file_vecs = to_token(file_vecs, step_vec_to_token_dict)
    split_dataset(file_vecs)

    print('make dataset tlp done.')
