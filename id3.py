from __future__ import division
import numpy as np
import copy
from tree_node import treeNode

global TREE
TREE = np.array([])

# ex. A: [[overcast, sunny, rain], [very, medium, not], 
# ...]
# ex. D: [[overcast, hot , high, not, N], ..., [rain, ..., N]]

def decision_tree(D, parent, A, attr_val):
    global TREE
    if not (D.size == 0):
        dclass = D[0][D.shape[1] - 1]
        node = treeNode(idx=TREE.size, split_val=attr_val, parent=parent)
        if len(A) == 0:
            node.mark_leaf(dclass)
            TREE = np.append(TREE, node)
            return TREE

        #check if a vals are all equal
        a_check = True
        arr1 = D[0]
        arr1 = np.delete(arr1, arr1.shape[0] - 1)
        for i in range(1, D.shape[0]):
            arr2 = D[i]
            arr2 = np.delete(arr2, arr2.shape[0] - 1)
            if not np.array_equal(arr1, arr2):
                a_check = False

        check = True
        for i in range(0, D.shape[0]):
            cl = D[i][D.shape[1] - 1]
            if cl != dclass:
                check = False 

        if check or a_check:
            node.mark_leaf(dclass)
            TREE = np.append(TREE, node)
            return TREE

        (split_attr, new_a) = max_info_gain(D, A)
        node.set_split(split_attr)
        TREE = np.append(TREE, node)
        opts = get_opts_a(A, split_attr)

        for av in range(len(opts)):
            #for each value of Ai, create dv
            dv = get_si_dt(opts[av], D, split_attr)
            decision_tree(dv, node, new_a, opts[av])

def max_info_gain(D, A):
    max_i_gain = info_gain(D, 0)
    #max_i_gain = info_gain_v2(D, 0)
    #max_i_gain = info_gain_v3(D, 0, A)
    split_attr = 0
    new_a = copy.deepcopy(A)
    for a in range(1, len(A)): 
        if info_gain(D, a) > max_i_gain:
        #if info_gain_v2(D, a) > max_i_gain:
        #if info_gain_v3(D, a, A) > max_i_gain:
            max_i_gain = info_gain(D, a)
            #max_i_gain = info_gain_v2(D, a)
            #max_i_gain = info_gain_v3(D, a, A)
            split_attr = a
    del new_a[split_attr]
    return (split_attr, new_a)


#original ID3 information gain
def info_gain(s, a_col):
    sum_e = 0
    for i in range(s.shape[1] - 1):
        sum_e = sum_e + entropy(s, i)
    gain = sum_e - entropy(s, a_col)
    return gain

def get_opts_a(s, a_col):
    dup_s = copy.deepcopy(s)
    new_s = dup_s[:][a_col]
    opts = list(new_s)
    return opts

def get_opts(s, a_col):
    dup_s = copy.deepcopy(s)
    new_s = dup_s[:, a_col].flat
    opts = list(set(new_s))
    return opts

def get_si_dt(i, s, a_col):
    dup_s = np.array(s)
    ind = dup_s[:, a_col] == i 
    si = dup_s[ind, :]
    si = strip_data_attr(si, a_col)
    return si

def strip_data_attr(si, col):
    new_si = copy.deepcopy(si)
    if si.shape[0] > 0:
        arr = np.delete(new_si, col, 1)
        print "ARR " + str(arr)
        return arr
    else:
        return si

def get_si(i, s, a_col):
    dup_s = np.array(s)
    ind = dup_s[:, a_col] == i 
    si = dup_s[ind, :]
    return si

#entropy calculation
def entropy(s, a_col):
    opts = get_opts(s, a_col)
    c = len(opts)
    ent = 0
    #for each option
    for o in range(c):
        sa = get_si(opts[o], s, a_col)
        acc = 0
        for i in range(2):
            si = get_si(str(i), sa, s.shape[1] - 1)
            if (len(si) != 0):
                acc = acc - (len(si)/len(sa))*np.log2(len(si)/len(sa))
        ent = ent + (len(sa)/len(s))*acc
    return ent

def calc_num_pos(s):
    dup_s = np.array(s)
    ind = dup_s[:, s.shape[1] - 1] == '1'
    pos = dup_s[ind, :]
    return len(pos)

#Taylor Information Gain
def info_gain_v2(s, a_col):
    p = calc_num_pos(s)
    total = len(s)
    n = total - p
    part1 = (2*p*n)/(p+n)
    opts = get_opts(s, a_col)
    c = len(opts)
    acc = 0
    for a in range(c):
        si = get_si(opts[a], s, a_col)
        if (len(si) !=0):
            pi = calc_num_pos(si)
            ti = len(si)
            ni = ti - pi
            acc = acc + (2*pi*ni)/(pi + ni)
    gain = part1 - acc
    return gain

def info_gain_v3(s, a_col, A):
    #gain = entropy(system) - entropy(single) * normalization factor
    sum_e = 0
    sum_af = 0
    for i in range(s.shape[1] - 1):
        sum_e = sum_e + entropy(s, i)
        sum_af = sum_af + normalization_factor(s, i, A)
    if sum_af > 0:
        gain = sum_e - entropy(s, a_col) * (normalization_factor(s, a_col, A) / sum_af)
    else:
        gain = sum_e - entropy(s, a_col) * (normalization_factor(s, a_col, A) / .001)
    return gain


def normalization_factor(s, a_col, A):
    nf = 0
    n = len(A[a_col])
    if n > 0:
        for i in range(n):
            si = get_si(A[a_col][i], s, a_col)
            tot = len(si)
            x1 = calc_num_pos(si)
            x2 = tot - x1
            nf = nf + (x1 - x2)
        return nf / n
    return 1

def main():
    tennis = np.genfromtxt('tennis.csv', delimiter=',', skip_header=1, dtype='S')
    ten_labels = [['overcast', 'sunny', 'rain'],
                        ['hot', 'cool', 'mild'],
                        ['high', 'normal'],
                        ['FALSE', 'TRUE']]

    titanic = np.genfromtxt('train.csv', delimiter=',', skip_header=1, dtype='S')
    titanic_labels = [np.array(map(str, line.split(','))) for line in open('labels.csv')]

    #decision_tree(tennis, None, ten_labels, None)
    decision_tree(titanic, None, titanic_labels, None)
    node1 = TREE[0]
    node1.print_tree(0, TREE)
    return TREE


main()

