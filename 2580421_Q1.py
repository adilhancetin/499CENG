import os
import random
import math

FEATURE_NAMES = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

class TreeNode:
    def __init__(self, attribute=None, prediction=None, majority_label=None):
        self.attribute = attribute
        self.prediction = prediction
        self.majority_label = majority_label
        self.children = {}

    def is_leaf(self):
        return self.prediction is not None

    def add_child(self, attribute_value, child_node):
        self.children[attribute_value] = child_node


class CarDataset:

    def __init__(self, values):
        self.values = values

    def feature(self, index):
        return self.values[index]

    def label(self):
        return self.values[-1]


def load_car_dataset(file_path):
    dataset = []

    with open(file_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 7:
                continue

            example = CarDataset(parts)
            dataset.append(example)

    return dataset


def train_test_split(examples, test_size=0.2, random_state=42):
    shuffled = examples[:]
    random.Random(random_state).shuffle(shuffled)

    test_count = int(len(shuffled) * test_size)
    test_set = shuffled[:test_count]
    train_set = shuffled[test_count:]
    return train_set, test_set

def split_dataset_on_attribute_value(dataset, attribute_index, attribute_value):
    subset = []
    for example in dataset:
        if example.feature(attribute_index) == attribute_value:
            subset.append(example)
    return subset

def check_same_labels(train_set):
    if not train_set:
        return True
    
    first_label = train_set[0].label()
    for example in train_set:
        if example.label() != first_label:
            return False
    return True

def calculate_entropy(dataset):
    if not dataset:
        return 0.0

    count = {}
    for example in dataset:
        label = example.label()
        if label not in count:
            count[label] = 0
        count[label] += 1

    total = len(dataset)
    sigma = 0.0
    for label in count:
        p = count[label] / total
        sigma -= p * math.log2(p)
    return sigma   

def attr_values_helper(dataset, attribute_index):
    att_values = []
    for example in dataset:
        att_value = example.feature(attribute_index)
        if att_value not in att_values:
            att_values.append(att_value)
    return att_values    

def calculate_information_gain(dataset, attribute_index):
    if not dataset:
        return 0.0

    att_values = attr_values_helper(dataset, attribute_index)
    total = len(dataset)
    sigma = 0.0
    for att_value in att_values:
        splitted_dataset = split_dataset_on_attribute_value(dataset, attribute_index, att_value)
        sigma += (len(splitted_dataset) / total) * calculate_entropy(splitted_dataset) 

    return calculate_entropy(dataset) - sigma   

def find_majority_label(dataset):
    if not dataset:
        return None

    count = {}
    for example in dataset:
        label = example.label()
        if label not in count:
            count[label] = 0
        count[label] += 1

    majority_label = max(count, key=count.get)
    return majority_label


def id3_training_loop(current_node, train_set, attributes, fallback_label):

    if not train_set:
        current_node.prediction = fallback_label
        current_node.majority_label = fallback_label
        return current_node

    if check_same_labels(train_set):
        current_node.prediction = train_set[0].label()
        current_node.majority_label = train_set[0].label()
        return current_node

    if not attributes:
        majority = find_majority_label(train_set)
        current_node.prediction = majority
        current_node.majority_label = majority
        return current_node
    
    best_attribute = 0
    for attribute_index in attributes:
        if calculate_information_gain(train_set, attribute_index) > calculate_information_gain(train_set, best_attribute):
            best_attribute = attribute_index

    current_node.attribute = best_attribute
    current_node.majority_label = find_majority_label(train_set)
    attr_values = attr_values_helper(train_set, best_attribute)
    for attr_value in attr_values:
        child_node = TreeNode()
        current_node.add_child(attr_value, child_node)      
        splitted_dataset = split_dataset_on_attribute_value(train_set, best_attribute, attr_value)
        if not splitted_dataset:
            child_node.prediction = fallback_label
            child_node.majority_label = fallback_label
        else:
            new_attributes = [attr for attr in attributes if attr != best_attribute]
            id3_training_loop(child_node, splitted_dataset, new_attributes, current_node.majority_label)
    
    return current_node


def predict_example(root_node, example):
    current_node = root_node

    while current_node is not None and not current_node.is_leaf():
        attribute_index = current_node.attribute
        attribute_value = example.feature(attribute_index)

        if attribute_value not in current_node.children:
            return current_node.majority_label

        current_node = current_node.children[attribute_value]

    if current_node is None:
        return None

    return current_node.prediction


def predict_dataset(root_node, dataset):
    predictions = []
    for example in dataset:
        predictions.append(predict_example(root_node, example))
    return predictions


def accuracy_score(y_true, y_pred):
    if not y_true:
        return 0.0

    correct = 0
    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == predicted_label:
            correct += 1

    return correct / len(y_true)


def precision_recall_f1_macro(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0, 0.0, 0.0

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for label in labels:
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for true_label, predicted_label in zip(y_true, y_pred):
            if predicted_label == label and true_label == label:
                true_positive += 1
            elif predicted_label == label and true_label != label:
                false_positive += 1
            elif predicted_label != label and true_label == label:
                false_negative += 1

        if true_positive + false_positive == 0:
            precision = 0.0
        else:
            precision = true_positive / (true_positive + false_positive)

        if true_positive + false_negative == 0:
            recall = 0.0
        else:
            recall = true_positive / (true_positive + false_negative)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    count = len(labels)
    return precision_sum / count, recall_sum / count, f1_sum / count
    



# Implementation of the ID3 algorithm similar to the slides provided
def main():
    data_path = os.path.join(os.path.dirname(__file__), "car.data")
    dataset = load_car_dataset(data_path)

    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    
    root_node = TreeNode()
    attributes = list(range(len(FEATURE_NAMES)))
    initial_majority_label = find_majority_label(train_set)
    decision_tree_root_node = id3_training_loop(root_node, train_set, attributes, initial_majority_label)

    y_true = [example.label() for example in test_set]
    y_pred = predict_dataset(decision_tree_root_node, test_set)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1_macro(y_true, y_pred)

    print(f"accuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1: {f1:.4f}")
    

    


if __name__ == "__main__":
    main()