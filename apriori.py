# -*- coding: utf-8 -*- #

import numpy as np
try:
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import skcuda.misc
    CUDA_FLAG = True
except:
    CUDA_FLAG = False
    print("Failed to import Pycuda! Machine does not support GPU acceleration.")

def compute_vertical_bitvector_data(data, use_CUDA):
    #---build item to idx mapping---#
    idx = 0
    item2idx = {}
    for transaction in data:
        for item in transaction:
            if not item in item2idx:
                item2idx[item] = idx
                idx += 1
    idx2item = { idx : str(int(item)) for item, idx in item2idx.items() }
    #---build vertical data---#
    vb_data = np.zeros((len(data), len(item2idx)), dtype=int)
    for trans_id, transaction in enumerate(data):
        for item in transaction:
            vb_data[trans_id, item2idx[item]] = 1
    if use_CUDA:
        vb_data = gpuarray.to_gpu(vb_data.astype(np.uint16))
    print('Data transformed into vertical bitvector representation with shape: ', np.shape(vb_data))
    return vb_data, idx2item

#############################
# COMPUTE C1 AND L1 ITEMSET #
#############################

def compute_C1_and_L1_itemset(data, num_trans, min_support):
    #---compute C1---#
    C1 = {}
    for transaction in data:
        for item in transaction:
            if not item in C1:
                C1[item] = 1
            else: C1[item] += 1
    #---compute L1---#
    L1 = []
    support1 = {}
    for candidate, count in sorted(C1.items(), key=lambda x: x[0]):
        support = count / num_trans
        if support >= min_support:
            L1.insert(0, [candidate])
            support1[frozenset([candidate])] = count
    return list(map(frozenset, sorted(L1))), support1, C1



def compute_CK(LK_, k):
    CK = []
    for i in range(len(LK_)):
        for j in range(i+1, len(LK_)): # enumerate all combinations in the Lk-1 itemsets
            L1 = sorted(list(LK_[i]))[:k-2]
            L2 = sorted(list(LK_[j]))[:k-2]
            if L1 == L2: # if the first k-1 terms are the same in two itemsets, merge the two itemsets
                new_candidate = frozenset(sorted(list(LK_[i] | LK_[j]))) # set union
                CK.append(new_candidate) 
    return sorted(CK)



def compute_LK(D, CK, num_trans, min_support):
    support_count = {}
    for item in D: # traverse through the data set
        for candidate in CK: # traverse through the candidate list
            if candidate.issubset(item): # check if each of the candidate is a subset of each item
                if not candidate in support_count:
                    support_count[candidate] = 1
                else: support_count[candidate] += 1
    LK = []
    supportK = {}
    for candidate, count in sorted(support_count.items(), key=lambda x: x[0]):
        support = count / num_trans
        if support >= min_support:
            LK.append(candidate)
            supportK[candidate] = count
    return sorted(LK), supportK




def apriori(data, min_support, use_CUDA=False):
    # if use_CUDA:
    #     vb_data, idx2item = compute_vertical_bitvector_data(data, use_CUDA)
    #     print(type(vb_data))
    #     # print(vb_data.shape)
    #     # print(vb_data[100])
    #     N = np.int32(vb_data.shape[1])
    #     GPU_memory = cuda.mem_alloc(N.nbytes)
    #     item_support = gpuarray.sum(vb_data, axis=0).get()
        
    #     print(item_support.shape)
    #     print("item_support  ", item_support)
    D = sorted(list(map(set, data)))
    # print(D)
    num_trans = float(len(D))
    L1, support_list, CK = compute_C1_and_L1_itemset(data, num_trans, min_support)
    # print("L1   ", L1)
    # print("support_list   ", support_list)
    # print("CK   ", CK)
    L = [L1]
    k = 1

    while (True): # create superset k until the k-th set is empty (ie, len == 0)
        print('Running Apriori: the %i-th iteration with %i candidates...' % (k, len(CK)))
        k += 1
        # print("L[-1]   ", L[-1])
        CK = compute_CK(LK_=L[-1], k=k)
        # print("CKKKK   ", CK)
        LK, supportK = compute_LK(D, CK, num_trans, min_support)
        
        if len(LK) == 0:
            L = [sorted([tuple(sorted(list(itemset), key=lambda x: int(x))) for itemset in LK]) for LK in L]
            support_list = dict((tuple(sorted(list(k), key=lambda x: int(x))), v) for k, v in support_list.items())
            print('Running Apriori: the %i-th iteration. Terminating ...' % (k-1))
            break
        else:
            L.append(LK)
            support_list.update(supportK)
    return L, support_list

