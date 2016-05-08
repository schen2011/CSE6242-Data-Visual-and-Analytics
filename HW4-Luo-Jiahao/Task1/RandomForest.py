import csv
import numpy as np  # http://www.numpy.org
from random import shuffle
from CSE6242HW4Tester import generateSubmissionFile

"""
Here, X is assumed to be a matrix with n rows and d columns
where n is the number of samples
and d is the number of features of each sample

Also, y is assumed to be a vector of n labels
"""

# Enter your name here
myname = "Jiahao Luo"

# Calculate entropy value
def entropy(dataset):
    instance_num = len(dataset)
    if instance_num == 0:
        return 0
    class0_num = 0
    class1_num = 0
    for label in dataset:
        if label[-1] == 0:
            class0_num += 1.0
        else:
            class1_num += 1.0
    prob0 = float(class0_num)/instance_num
    prob1 = float(class1_num)/instance_num
    if prob0 == 0 or prob1 == 0:
        return 0
    else:
    # Formula to calculate entropy
        return - prob0 * np.log2(prob0) - prob1 * np.log(prob1)

# Using information gain to find out the best value and best attribute to split the data
def information_gain(dataset, feat_col):
    instance_num = dataset.shape[0]
    feature_num = dataset.shape[1]
    X = dataset[:,0:-1]

# Find out all the specific values without duplicate
    m_feature = int(np.ceil(np.sqrt(feature_num)))
    shuffle_feature = np.transpose(np.random.permutation(np.transpose(X)))
    for i in range(m_feature):
        feature_value_list = [sample[i] for sample in shuffle_feature]
        unique_values = set(feature_value_list)

# Set the best_value and best_info_gain as 0 at first
    best_value, best_info_gain = 0.0, 0.0
    # The base entropy
    base_entropy = entropy(dataset)
    # Sort the attribute value
    sorted_values = sorted(unique_values)
    if len(sorted_values) > max_iteration:
        sorted_values = np.linspace(min(sorted_values), max(sorted_values), max_iteration + 1)
    sorted_values = [(temp1 + temp2) / 2 for (temp1, temp2) in zip(sorted_values[:-1], sorted_values[1:])]

    for value in sorted_values:
        new_entropy = 0.0
        left_branch, right_branch = split_dataset(dataset, feat_col, value)
        prob1 = len(left_branch) / float(instance_num)
        prob2 = len(right_branch) / float(instance_num)
        new_entropy = prob1 * entropy(left_branch) + prob2 * entropy(right_branch)
        info_gain = base_entropy - new_entropy
        info_gain -= prob1 * entropy(left_branch)
        info_gain -= prob2 * entropy(right_branch)
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_value = value
    return (best_info_gain, best_value)

# Split data based on feature column index and value
def split_dataset(dataset, feat_col, value):
    left_branch = []
    right_branch = []
    for feature in dataset:
        if feature[feat_col] <= value:
            left_branch.append(feature)
        else:
            right_branch.append(feature)
    return (left_branch, right_branch)

# Vote and find the find_majority_class class (binary: 0 or 1)
def find_majority_class(dataset):
    instance_num = len(dataset)
    class1_num = sum(dataset[:,-1])
    class0_num = instance_num - class1_num
    if class1_num >= class0_num:
        return 1
    else:
        return 0

class DecisionNode:
    def __init__(self):
        self.label = -1
        self.attribute = 0
        self.best_split_value = 0
        self.depth = 0
        self.leaf = True
        self.left_branch = None
        self.right_branch = None

class RandomForest(object):
# -------Decision Tree Part Start-------- #
    class __DecisionTree(object):
        tree = {}

        def learn(self, X, y):
            attribute_value_list = X.shape[1]
            self.X = X
            self.y = y
            self.attribute_value_list = X.shape[1]

            y = y.reshape([y.shape[0], 1])
            dataset = np.concatenate((X, y), axis=1)

            self.tree["root"] = DecisionNode()
            root = self.tree["root"]

            candidates = set(range(num_of_attr_in_tree))
            # Set up the threshold as 5, when the number of instances is 5, then stop spliting the data
            self.create_tree(root, dataset, -1, candidates, threshold=5, e_th=0.0001)
            self.root = root
            root.depth = 0

        def create_tree(self, decision_node, dataset, target, candidates, threshold=0, e_th=1e-8):
            best_attribute = 0
            best_split_value = 0.0
            max_info_gain = 0.0
            ## Class label list: class_list=[0,1,0,1,1,1,0..]
            class_list = [label[-1] for label in dataset]
            # If all the samples are label 0 or label 1
            if class_list.count(class_list[0]) == len(class_list):
                decision_node.leaf = True
                decision_node.label = class_list[0]
                return decision_node.label
            # If the remain subtree is one sample only, then return
            if len(class_list) == 1:
                decision_node.leaf = True
                decision_node.label = class_list[0]
                return decision_node.label
            if threshold > 0 and len(dataset) <= threshold:
                decision_node.leaf = True
                decision_node.label = find_majority_class(dataset)
                return decision_node.label
            if not candidates:
                decision_node.leaf = True
                decision_node.label = find_majority_class(dataset)
                return decision_node.label
            # Generates a random sample from self.attribute_value_list given 1-D array,
            #the return size is num_of_attr_in_tree and sampling without replacement
            random_attribute = np.random.choice(self.attribute_value_list, num_of_attr_in_tree, False)
            for attribute in random_attribute:
                info_gain, value = information_gain(dataset, attribute)
                if info_gain > max_info_gain:
                    best_attribute = attribute
                    max_info_gain = info_gain
                    best_split_value = value
            attribute = best_attribute

            if max_info_gain > e_th:
                left_branch, right_branch = split_dataset(dataset, attribute, best_split_value)
                decision_node.leaf = False
                decision_node.attribute = attribute
                decision_node.best_split_value = best_split_value
                new_candidates = candidates
                decision_node.left_branch = DecisionNode()
                decision_node.left_branch.depth = decision_node.depth + 1
                decision_node.right_branch = DecisionNode()
                decision_node.right_branch.depth = decision_node.depth + 1
                self.create_tree(decision_node.left_branch, left_branch, target, new_candidates, threshold)
                self.create_tree(decision_node.right_branch, right_branch, target, new_candidates, threshold)
            else:
                decision_node.leaf = True
                decision_node.label = find_majority_class(dataset)

        def classify(self, test_instance):
            # TODO: return predicted label for a single instance using
            node = self.root
            while not node.leaf:
                node_attribute = node.attribute
                if (test_instance[node_attribute] <= node.best_split_value):
                    node = node.left_branch
                else:
                    node = node.right_branch
            return int(node.label)

    decision_trees = []

    def __init__(self, num_trees):
        # according to your need, in this case, I set the number of trees as 10
        self.num_trees = num_trees
        self.decision_trees = [self.__DecisionTree()] * num_trees

    # You MUST NOT change this signature
    def fit(self, X, y):
        instance_num, attribute_num = X.shape
        # print instance_num, attribute_num
        for i in range(self.num_trees):
            index = np.random.choice(instance_num, instance_num, True)
            # index = np.arange(0, instance_num)
            X_sample = X[index]
            y_sample = y[index]
            self.decision_trees[i].learn(X_sample, y_sample)
        pass

    # You MUST NOT change this signature
    def predict(self, X):
        y = np.array([], dtype=int)
        for instance in X:
            votes = np.array([decision_tree.classify(instance) for decision_tree in self.decision_trees])
            counts = np.bincount(votes)
            y = np.append(y, np.argmax(counts))
        return y

# according to your need, in this case, I set the number of trees as 10
num_of_trees = 10
num_of_attr_in_tree = 4
max_iteration = 10

# main function
def main():
    X = []
    y = []

    # Load data set
    with open("hw4-data.csv") as f:
        next(f, None)

        for line in csv.reader(f, delimiter=","):
            X.append(line[:-1])
            y.append(line[-1])

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    # Implemented K-fold cross-validation
    K = 10
    cross_validation(K, X, y)

def cross_validation(K, X, y):
    # Split training/test sets
    # You need to modify the following code for cross-validation
    result = 0
    for num_of_fold in range(K):
        X_train = np.array([x for i, x in enumerate(X) if i % K != num_of_fold], dtype=float)
        y_train = np.array([z for i, z in enumerate(y) if i % K != num_of_fold], dtype=int)
        X_test = np.array([x for i, x in enumerate(X) if i % K == num_of_fold], dtype=float)
        y_test = np.array([z for i, z in enumerate(y) if i % K == num_of_fold], dtype=int)

        randomForest = RandomForest(num_of_trees)  # Initialize according to your implementation

        randomForest.fit(X_train, y_train)

        y_predicted = randomForest.predict(X_test)

        results = [prediction == truth for prediction,truth in zip(y_predicted, y_test)]

        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        print "accuracy: %.4f" % accuracy
        result += results.count(True)

    print "Average accuracy: %.4f" % (float(result) / len(y))
    generateSubmissionFile(myname, randomForest)

main()
