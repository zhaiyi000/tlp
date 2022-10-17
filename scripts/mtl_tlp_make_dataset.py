import pickle
import os
import argparse



def union_all():

    def merge(datasets_global1, datasets_global2):
        datasets_global_new = []

        for file_idx__, (data1, data2) in enumerate(zip(datasets_global1, datasets_global2)):

            assert (data1[0] == data2[0])
            # assert (data1[6][0] == data2[6][0])
            datas = list(data1)
            datas.append(data2[-2])
            datas.append(data2[-1])
            datasets_global_new.append(datas)

        return datasets_global_new

    with open(args.union_datasets[0], 'rb') as f:
        datasets_global1 = pickle.load(f)
    print('load 1 done.')

    for i in range(1, len(args.union_datasets)):

        with open(args.union_datasets[i], 'rb') as f:
            datasets_global2 = pickle.load(f)
        print(f'load {i+1} done.')
        datasets_global3 = merge(datasets_global1, datasets_global2)
        del datasets_global1
        del datasets_global2
        datasets_global1 = datasets_global3

    with open(args.save_name, 'wb') as f:
        pickle.dump(datasets_global1, f)

    print('done.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--union_datasets", type=str, nargs='+')
    parser.add_argument("--save_name", type=str, default='')
    args = parser.parse_args()
    
    if args.save_name == '':
        args.save_name = f'mtl_tlp_dataset_{len(args.union_datasets)}.pkl'

    print(args)
    union_all()