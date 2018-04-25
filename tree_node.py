import numpy as np

class treeNode:
    def __init__(self, idx=0, is_leaf=True, split_attr=None, split_val=None,
            classification=None, parent=None, child_array=np.array([])):
        self.is_leaf = is_leaf
        self.idx = idx
        self.split_attr = split_attr
        self.split_val = split_val
        self.classification = classification
        self.parent = parent
        self.child_array = child_array

    def mark_leaf(self, classification):
        self.classification = classification
        self.is_leaf = True
    
    def set_split(self, split_attribute):
        self.is_leaf = False
        self.split_attr = split_attribute

    def print_tree(self, index, tree):
        if tree[index].is_leaf:
            self.print_node(tree[index])
        else:
            self.print_branch(index, tree)
        if tree.size > 1:
            newtree = np.delete(tree, 0, 0)
            self.print_tree(index, newtree)

    def print_node(self, node):
        print "\nLeaf Index " + str(node.idx)
        print "Classification " + str(node.classification)
        print "Split val " + str(node.split_val)

    def print_branch(self, index, tree):
        node = tree[index]
        print "\nBranch Index " + str(node.idx)
        print "Split on " + str(node.split_attr)
        print "Split val " + str(node.split_val)
