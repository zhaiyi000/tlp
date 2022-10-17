import os


os.chdir('dataset_gpu/measure_records')
files = os.listdir('k80')

for file in files:
    with open(f'k80/{file}', 'r') as f:
        lines1 = f.read().strip().split('\n')

    with open(f't4/{file}', 'r') as f:
        lines2 = f.read().strip().split('\n')


    # print(len(lines1), len(lines2))
    if len(lines1) != len(lines2):
        print(file, len(lines1), len(lines2))

        min_len = min(len(lines1), len(lines2))

        lines1 = lines1[:min_len]
        lines2 = lines2[:min_len]

        with open(f'k80/{file}', 'w') as f:
            f.write('\n'.join(lines1))

        with open(f't4/{file}', 'w') as f:
            f.write('\n'.join(lines2))


with open('k80/([8a830aeb28cc45b189200eb204fd8825,8,32,32,512,1,1,512,256,1,1,1,256,8,32,32,256],cuda).json', 'r') as f:
    lines1 = f.read().strip().split('\n')

lines1[2395] = '{"i": [["[\\"8a830aeb28cc45b189200eb204fd8825\\", 8, 32, 32, 512, 1, 1, 512, 256, 1, 1, 1, 256, 8, 32, 32, 256]", "cuda -keys=cuda,gpu -arch=sm_37 -max_num_threads=1024 -max_threads_per_block=1024 -registers_per_block=65536 -shared_memory_per_block=49152 -thread_warp_size=32", [-1, 16, 64, 49152, 2147483647, 1024, 8, 32], "", 0, []], [[], [["CI", 7], ["SP", 5, 0, 8, [1, 2, 4, 1], 1], ["SP", 5, 5, 32, [8, 2, 1, 1], 1], ["SP", 5, 10, 32, [1, 8, 1, 1], 1], ["SP", 5, 15, 256, [1, 4, 4, 16], 1], ["SP", 5, 20, 1, [1, 1], 1], ["SP", 5, 23, 1, [1, 1], 1], ["SP", 5, 26, 512, [4, 1], 1], ["RE", 5, [0, 5, 10, 15, 1, 6, 11, 16, 2, 7, 12, 17, 20, 23, 26, 21, 24, 27, 3, 8, 13, 18, 22, 25, 28, 4, 9, 14, 19]], ["FSP", 8, 0, 1, 3], ["FSP", 8, 4, 2, 3], ["FSP", 8, 8, 3, 3], ["FSP", 8, 12, 4, 3], ["RE", 8, [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]], ["CA", 5, 8, 11], ["CHR", 4, "shared", [5]], ["CA", 5, 6, 14], ["CI", 4], ["CHR", 2, "shared", [6]], ["CA", 3, 7, 14], ["CI", 2], ["CI", 1], ["FU", 10, [0, 1, 2, 3]], ["AN", 10, 0, 5], ["FU", 10, [1, 2, 3, 4]], ["AN", 10, 1, 4], ["FU", 10, [2, 3, 4, 5]], ["AN", 10, 2, 6], ["FU", 6, [0, 1, 2, 3]], ["SP", 6, 0, 128, [1], 1], ["AN", 6, 1, 2], ["FFSP", 6, 0, [4, 3, 2, 1], 1, 1], ["AN", 6, 1, 6], ["FU", 3, [0, 1, 2, 3]], ["SP", 3, 0, 16, [1], 1], ["AN", 3, 1, 2], ["FFSP", 3, 0, [4, 3, 2, 1], 1, 1], ["AN", 3, 1, 6], ["PR", 7, 0, "auto_unroll_max_step$0"]]]], "r": [[0.176353, 0.176359], 0, 3.90316, 1626295478], "v": "v0.6"}'
with open('k80/([8a830aeb28cc45b189200eb204fd8825,8,32,32,512,1,1,512,256,1,1,1,256,8,32,32,256],cuda).json', 'w') as f:
    f.write('\n'.join(lines1))