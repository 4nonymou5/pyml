import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score

#get the distance between two items
def distance(item1,item2):
    item1 = np.array(item1)
    item2 = np.array(item2)
    return np.linalg.norm(item1-item2)


#returns the k neighbors to test_instance
def get_neighbors(training_set,labels,test_instance,k):
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance,training_set[index])
        distances.append((training_set[index],dist,labels[index]))
    distances.sort(key=lambda x:x[1])
    neighbors = distances[:k]
    return neighbors

#get the label of the instance with the help of neighbors
def predict(neighbours):
    n_labels = [x[2] for x in neighbours]
    return max(n_labels, key=n_labels.count)

#return the accuracy score between two lists
def get_accuracy(actual_labels,predicted_labels):
    return accuracy_score(actual_labels,predicted_labels)


if __name__ == "__main__":
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_labels = iris.target
    
    # randomizing the indices of dataset
    indices = np.random.permutation(len(iris_data))
    
    n_test_samples = 12
    learnset_data = iris_data[indices[:-n_test_samples]]
    learnset_labels = iris_labels[indices[:-n_test_samples]]

    test_data = iris_data[indices[n_test_samples:]]
    test_labels = iris_labels[indices[n_test_samples:]]

    k=4
    pred_labels = []
    for i in range(len(test_data)):
        n = get_neighbors(learnset_data,learnset_labels,test_data[i],k)
        pred_labels.append(predict(n))
    
    print(get_accuracy(test_labels,pred_labels))
    



