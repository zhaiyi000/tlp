# TLP: A Deep Learning-based Cost Model for Tensor Program Tuning

This repo is based on a fork of [tenset](https://github.com/tlc-pack/tenset).

[tlp slides](./tlp%20slides.pptx)

## Installation

Build and install this repo following the [guide](https://github.com/zhaiyi000/tlp/blob/main/docs/install/from_source.rst).

Version information can refer to [here](version.log).

## Download the TenSet and TenSet-TLP datasets

1. Download

   You can download [tenset_cpu_v3.3.zip](https://drive.google.com/file/d/1JQwGEe8jCpuhZPnUxO0Sb1CJJ06uevy6/view?usp=sharing), [tenset_gpu_v3.3.zip](https://drive.google.com/file/d/1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK/view?usp=sharing), [tenset_tlp_v0.1.zip](https://drive.google.com/file/d/1WVNbmha3jjlqAAX-81N_doJ5IFihGfSK/view?usp=sharing) from google drive. And put these zip files under `tlp/scripts`.

2. Unzip

   ```shell
   cd scripts
   unzip dataset_cpu_v3.3.zip
   unzip dataset_gpu_v3.3.zip
   unzip dataset_tlp_v0.1.zip
   mv i7 dataset_cpu/measure_records
   ```

3. There are some errors when training MTL-TLP. Execution following cmd to avoid them.

   ```shell
   python tlp_preprocess_dataset_gpu.py
   ```


## Train a TLP cost model

#### CPU 

1. Make a dataset.

   ```shell
   rm -f dataset
   ln -s dataset_cpu dataset
   
   # This will take a long time. If you are just trying it out, you can set `--files_cnt` to a small value, such as 100.
   python tlp_make_dataset.py --files_cnt=2308 --json_files_path=dataset/measure_records/i7 --platform=llvm
   
   # python tlp_make_dataset.py --files_cnt=2308 --json_files_path=dataset/measure_records/platinum-8272 --platform=llvm  
   # python tlp_make_dataset.py --files_cnt=2308 --json_files_path=dataset/measure_records/e5-2673 --platform=llvm
   # python tlp_make_dataset.py --files_cnt=2308 --json_files_path=dataset/measure_records/epyc-7452 --platform=llvm
   # python tlp_make_dataset.py --files_cnt=2308 --json_files_path=dataset/measure_records/graviton2 --platform=llvm
   ```

2. Train. Then pick a model based on the validation set loss.

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python tlp_train.py --save_folder=tlp_i7 --dataset=tlp_dataset_i7_2308_train_and_val.pkl
   ```

3. Eval

   ```shell
   python tlp_eval.py --test_dataset_name=tlp_dataset_i7_2308_test.pkl --load_name=tlp_i7/tlp_model_43.pkl
   ```

#### GPU 

1. Make a dataset.

   ```shell
   rm -f dataset
   ln -s dataset_gpu dataset
   
   python tlp_make_dataset.py --files_cnt=2308 --json_files_path=dataset/measure_records/t4 --platform=cuda
   
   # python tlp_make_dataset.py --files_cnt=2308 --json_files_path=dataset/measure_records/k80 --platform=cuda
   ```

2. Train. Then pick a model based on the validation set loss.

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python tlp_train.py --save_folder=tlp_t4 --dataset=tlp_dataset_t4_2308_train_and_val.pkl --step_size=40 --fea_size=20
   ```

3. Eval

   ```shell
   python tlp_eval.py --test_dataset_name=tlp_dataset_t4_2308_test.pkl --load_name=tlp_t4/tlp_model_45.pkl --platform=cuda
   ```

## Train a MTL-TLP cost model

#### CPU 

1. Make a dataset

   ```shell
   rm -f dataset
   ln -s dataset_gpu dataset
   
   python mtl_tlp_make_dataset.py --union_datasets tlp_dataset_platinum_8272_2308_train_and_val.pkl \
                                                   tlp_dataset_e5_2673_2308_train_and_val.pkl \
                                                   tlp_dataset_epyc_7452_2308_train_and_val.pkl \
                                                   tlp_dataset_graviton2_2308_train_and_val.pkl \
                                                   tlp_dataset_i7_2308_train_and_val.pkl
   ```

2. Train. Then pick a model based on the validation set loss.

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python mtl_tlp_train.py --save_folder=mtl_tlp_i7 --dataset=mtl_tlp_dataset_5.pkl --mtl_head_list=4,3,2,1,0
   ```

3. Eval

   ```shell
   python tlp_eval.py --test_dataset_name=tlp_dataset_i7_2308_test.pkl --load_name=mtl_tlp_i7/mtl_tlp_model_49.pkl
   ```

#### GPU 

1. Make a dataset

   ```shell
   rm -f dataset
   ln -s dataset_gpu dataset
   
   python mtl_tlp_make_dataset.py --union_datasets tlp_dataset_k80_2308_train_and_val.pkl \
                                                   tlp_dataset_t4_2308_train_and_val.pkl
   ```

2. Train. Then pick a model based on the validation set loss.

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python mtl_tlp_train.py --save_folder=mtl_tlp_t4 --dataset=mtl_tlp_dataset_2.pkl --mtl_head_list=1,0 --step_size=40 --fea_size=20
   ```

3. Eval

   ```shell
   python tlp_eval.py --test_dataset_name=tlp_dataset_t4_2308_test.pkl --load_name=mtl_tlp_t4/mtl_tlp_model_13.pkl --platform=cuda
   ```

## Use the model for search

#### CPU

```shell
rm -f dataset
ln -s dataset_cpu dataset

# TLP cost model
python tune_network.py --network=resnet_50 --n-trials=2000 --cost-model=tlp-no-update --load-model=tlp_i7/tlp_model_43.pkl --target='llvm -mcpu=core-avx2 -model=i7' --num_measures_per_round=10
# MTL-TLP cost model
python tune_network.py --network=resnet_50 --n-trials=2000 --cost-model=tlp-no-update --load-model=mtl_tlp_i7/mtl_tlp_model_49.pkl --target='llvm -mcpu=core-avx2 -model=i7' --num_measures_per_round=10
```

#### GPU

```shell
rm -f dataset
ln -s dataset_gpu dataset

# TLP cost model
python tune_network.py --network=resnet_50 --n-trials=2000 --cost-model=tlp-no-update --load-model=tlp_t4/tlp_model_45.pkl --target='cuda -model=t4' --num_measures_per_round=10 --step_size=40 --fea_size=20
# MTL-TLP cost model
python tune_network.py --network=resnet_50 --n-trials=2000 --cost-model=tlp-no-update --load-model=mtl_tlp_t4/mtl_tlp_model_13.pkl --target='cuda -model=t4' --num_measures_per_round=10 --step_size=40 --fea_size=20
```

## More experiments

1. fine-tuning

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python tlp_fine_tune.py --save_folder=tlp_i7_fine_tune --dataset=tlp_dataset_i7_2308_train_and_val.pkl --pre_train_model=tlp_platinum_8272/tlp_model_34.pkl
   ```

2. gpt

   The source code is a fork of this [commit](https://github.com/karpathy/minGPT/tree/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150) of minGPT.

   ```shell
   cd minGPT
   # 1. use unlabeled data to train the gpt model
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gpt.py --dataset=tlp_dataset_i7_2308_train_and_val.pkl --train_size_per_gpu=3072
   # 2. use labeled data to train the gpt and downstream model
   CUDA_VISIBLE_DEVICES=0,1,2,3 python tlp_train.py --save_folder=tlp_gpt --dataset=tlp_dataset_i7_2308_train_and_val.pkl --self_sup_model=minGPT/gpt_model_132.pt --attention_class=gpt --data_cnt=500 --train_size_per_gpu=384 --val_size_per_gpu=384
   # 3. eval
   python tlp_eval.py --test_dataset_name=tlp_dataset_i7_2308_test.pkl --load_name=tlp_gpt/tlp_model_29.pkl
   ```

3. bert

   ```shell
   # 1. make a dataset for bert
   python tlp_make_dataset_bert.py --json_files_path=dataset/measure_records/i7
   cd bert
   # 2. use unlabeled data to train the bert model
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train_bert.py --datasets=tlp_dataset_bert_platinum_8272_2308_train_and_val.pkl --batch_size=1152
   # 3. use labeled data to train the bert and downstream model
   CUDA_VISIBLE_DEVICES=0,1,2,3 python tlp_train.py --save_folder=tlp_bert --dataset=tlp_dataset_bert_i7_2308_train_and_val.pkl --self_sup_model=bert/bertmodel_78.pt --attention_class=bert --data_cnt=500 --train_size_per_gpu=384 --val_size_per_gpu=384
   # 4. eval
   python tlp_eval.py --test_dataset_name=tlp_dataset_bert_i7_2308_test.pkl --load_name=tlp_bert/tlp_model_33.pkl
   ```

## License
The code is licensed under an [Apache-2.0](LICENSE) license.  
The TenSet-TLP dataset is licensed under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.