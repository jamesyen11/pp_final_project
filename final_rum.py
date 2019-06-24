# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ runner.py ]
#   Synopsis     [ Implement of two basic algorithms to perform frequent pattern mining: 1. Apriori, 2. Eclat. 
#                  Find all itemsets with support > min_support. ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import csv
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------#
from apriori import apriori
from eclat import eclat
#-----------------------#
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})


##################
# CONFIGURATIONS #
##################
def get_config():
    parser = argparse.ArgumentParser(description='frequent itemset mining argument parser')
    parser.add_argument('mode', type=str, choices=['apriori', 'eclat', '1', '2'], help='algorithm mode')
    parser.add_argument('--min_support', type=float, default=0.6, help='minimum support of the frequent itemset')
    parser.add_argument('--toy_data', action='store_true', help='use toy data for testing')
    parser.add_argument('--iterative', action='store_true', help='run eclat in iterative method, else use the recusrive method')
    
    cuda_parser = parser.add_argument_group('Cuda settings')
    cuda_parser.add_argument('--use_CUDA', action='store_true', help='run eclat with GPU to accelerate computation')
    cuda_parser.add_argument('--block', type=int, default=16, help='block number for Cuda GPU acceleration')
    cuda_parser.add_argument('--thread', type=int, default=16, help='thread number for Cuda GPU acceleration')
    
    plot_parser = parser.add_argument_group('Plot settings')
    plot_parser.add_argument('--plot_campare', action='store_true', help='Run all the values in the block list and plot runtime')
    
    
    io_parser = parser.add_argument_group('IO settings')
    io_parser.add_argument('--input_path', type=str, default='./data/data.txt', help='input data path')
    io_parser.add_argument('--output_path', type=str, default='./data/output.txt', help='output data path')
    args = parser.parse_args()
    if args.mode == '1': args.mode = 'apriori'
    elif args.mode == '2': args.mode = 'eclat'
    return args

def proccess_mushroom_row(row):
    mapping_table = {"a":1, "b":2, "c":3, "d":4, "e":5, "f":6, "g":7, "h":8, "i":9, "j":10, "k":11, "l":12, "m":13
    , "n":14, "o":15, "p":16, "q":17, "r":18, "s":19, "t":20, "u":21, "v":22, "w":23, "x":24, "y":25, "z":26, "?":27}
    new_row = []
    for item in row:

        new_row.append(mapping_table[item])
    return new_row

#############
# READ DATA #
#############
def read_data(data_path, skip_header=False, toy_data=False):
    if toy_data:
        return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    data = []
    if not os.path.isfile(data_path): raise ValueError('Invalid data path.')
    with open(data_path, 'r', encoding='utf-8') as f:
        if 'mushroom' in data_path : 
            file = csv.reader(f, delimiter=',', skipinitialspace=True, quotechar='\r')
        else:
            file = csv.reader(f, delimiter=' ', skipinitialspace=True, quotechar='\r')
        if skip_header: next(file, None)  # skip the headers
        for row in file:
            if 'retail' in data_path:
                row.pop()
            elif 'mushroom' in data_path:
                row = proccess_mushroom_row(row)
            data.append(row)
    return data


#################
# RUN ALGORITHM #
#################
def run_algorithm(data, mode, support, iterative, use_CUDA, block, thread):
    if mode == 'apriori':
        print('Running Apriori algorithm with %f support and data shape: ' % (support), np.shape(data))
        result = apriori(data, support, use_CUDA)
        return result
    elif mode == 'eclat':
        print('Running eclat algorithm with %f support and data shape: ' % (support), np.shape(data))
        result = eclat(data, support, iterative, use_CUDA, block, thread)
        return result
    else:
        raise NotImplementedError('Invalid algorithm mode.')


################
# WRITE RESULT #
################
def write_result(result, result_path):
    if len(result[0]) == 0: print('Found 0 frequent itemset, please try again with a lower minimum support value!')
    with open(result_path, 'w', encoding='big5') as file:
        file_data = csv.writer(file, delimiter=',', quotechar='\r')
        for itemset_K in result[0]:
            for itemset in itemset_K:
                output_string = ''
                for item in itemset: output_string += str(item)+' '
                output_string += '(' + str(result[1][itemset]) +  ')'
                file_data.writerow([output_string])
    print('Results have been successfully saved to: %s' % (result_path))
    return True

def assert_at_most_one_is_true(*args):
    return sum(args) <= 1

########
# MAIN #
########
"""
    main function that runs the two algorithms, 
    and plots different experiment results.
"""
def main():
    args = get_config()
    data = read_data(args.input_path, toy_data=args.toy_data)
    
    #---argument error handling---#
    # if args.use_CUDA and args.mode != 'eclat':
    #     raise NotImplementedError()
    # assert assert_at_most_one_is_true(args.plot_support, args.plot_support_gpu, args.plot_block, args.plot_thread, args.compare_gpu)
    # if args.plot_support_gpu or args.compare_gpu or args.plot_block or args.plot_thread:
    #     try: assert args.use_CUDA
    #     except: raise ValueError('Must use Cuda for these experiments!')
                
    #---ploting mode handling---#
    if not args.plot_campare : 
        experiment_list = [args.min_support]
    elif args.plot_campare:
        experiment_list = (0.35, 0.3, 0.25, 0.2, 0.15)
    else:
        raise NotImplementedError()

    duration_eclat = []
    duration_eclat_gpu = []
    duration_apriori = []

    # for v in experiment_list:
    #     start_time = time.time()
    #     result = run_algorithm(data, 'eclat', v, args.iterative, args.use_CUDA, args.block, args.thread)
    #     duration_eclat_gpu.append(time.time() - start_time)
    #     print("Time duration eclat w gpu: %.5f" % (duration_eclat_gpu[-1]))

    for v in experiment_list:
        print('-'*77)
        start_time = time.time()
        result = run_algorithm(data, 'eclat', v, args.iterative, False, args.block, args.thread)
        duration_eclat.append(time.time() - start_time)
        print("Time duration eclat w/o gpu: %.5f" % (duration_eclat[-1]))


    for v in experiment_list:
        start_time = time.time()
        result = run_algorithm(data,  'apriori', v, args.iterative, False, args.block, args.thread)
        duration_apriori.append(time.time() - start_time)
        print("Time duration apriori w/o gpu: %.5f" % (duration_apriori[-1]))

    if args.plot_campare:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        title = 'apriori vs eclat'
        if args.plot_campare: 
            line3, = plt.plot(experiment_list, duration_apriori, marker='o', label='apriori w/o gpu')
            line2, = plt.plot(experiment_list, duration_eclat, marker='o', label='eclat w/o gpu')
            # line1, = plt.plot(experiment_list, duration_eclat_gpu, marker='o', label='eclat w gpu')
            plt.legend(handles=[line2, line3])
            for xy in zip(experiment_list, duration_apriori):
                ax.annotate('(%s, %.5s)' % xy, xy=xy, textcoords='data')
            for xy in zip(experiment_list, duration_eclat):
                ax.annotate('(%s, %.5s)' % xy, xy=xy, textcoords='data')
            # for xy in zip(experiment_list, duration_eclat_gpu):
      #   ax.annotate('(%s, %.5s)' % xy, xy=xy, textcoords='data')
            
            # for xy in zip(experiment_list, duration2):
            #     ax.annotate('(%s, %.5s)' % xy, xy=xy, textcoords='data')
        else:
            plt.plot(experiment_list, duration_eclat, marker='o')
        # for xy in zip(experiment_list, duration):
        #     ax.annotate('(%s, %.5s)' % xy, xy=xy, textcoords='data')
        plt.ylabel('execution time (seconds)')
        plt.xlabel('minimum support')
        plt.title(title)
        plt.grid()
        fig.savefig('./data/' + title + '.jpeg')
    else:
        done = write_result(result, args.output_path)


if __name__ == '__main__':
    main()

