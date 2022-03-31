import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Futura']


def load_data():
    """
    Loads the data from the csv file, split to dataset and obj_name and returns the numpy array
    :return:
        x: features by class
        obj_name: labels
    """
    class_name = ['animals', 'countries', 'fruits', 'veggies']
    data, name = None, None
    for i in range(len(class_name)):
        features = []
        labels = []
        with open('CA2data/' + class_name[i]) as F:
            for line in F:
                instance = line.strip().split(' ')
                name = instance[0]
                del instance[0]
                instance = [float(i) for i in instance]
                features.append(instance)
                labels.append(name)
        features = np.array(features)
        features = np.column_stack((features, [i] * features.shape[0]))
        labels = np.array(labels)
        data = features if data is None else np.vstack((data, features))
        name = labels if name is None else np.hstack((name, labels))
    return data, name


def random_centroids(data, k):
    """
    Randomly select k centroids from the dataset
    :param data: dataset
    :param k: number of centroids
    :return: centroids
    """
    index = np.random.choice(data.shape[0], k, replace=False)
    return data[index]


class KMeans:
    """
    KMeans class
    """

    def __init__(self, k, epsilon=0.0001):
        """
        Initialize KMeans
        :param k: k value, the number of clusters
        :param epsilon: convergence criteria
        """

        self.k = k
        self.centroids = []
        self.epsilon = epsilon

    def fit(self, data, seed=None):
        classifications = {}
        if seed is not None:
            np.random.seed(seed)
        # Randomly select k centroids from the dataset
        self.centroids = np.array(random_centroids(data, self.k))

        # Run the algorithm
        while True:

            for i in range(self.k):
                classifications[i] = []

            for feature_set in data:
                distances = [self.l2_distance(feature_set, centroid) for centroid in self.centroids]
                classification = distances.index(min(distances))
                classifications[classification].append(feature_set)

            prev_centroids = np.array(self.centroids)

            for classification in classifications:
                self.centroids[classification] = np.mean(classifications[classification], axis=0)

            optimized = True

            for i, c in enumerate(self.centroids):
                original_centroid = prev_centroids[i]
                current_centroid = self.centroids[i]
                if self.l2_distance(original_centroid, current_centroid) > self.epsilon:
                    optimized = False

            if optimized:
                labels = []
                for feature_set in data:
                    distances = [self.l2_distance(feature_set, centroid) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    labels.append(classification)
                return labels

    @staticmethod
    def l2_distance(x, y):
        # Return the Euclidean distance between X and Y
        return np.linalg.norm(x - y)


class KMedians:
    """
    KMedians class
    """

    def __init__(self, k, epsilon=0.0001):
        """
        Initialize KMedians
        :param k: k value, the number of clusters
        :param epsilon: convergence criteria
        """

        self.k = k
        self.centroids = []
        self.epsilon = epsilon

    def fit(self, data, seed=None):
        classifications = {}
        if seed is not None:
            np.random.seed(seed)
        # Randomly select k centroids from the dataset
        self.centroids = np.array(random_centroids(data, self.k))

        # Run the algorithm
        while True:

            for i in range(self.k):
                classifications[i] = []

            for feature_set in data:
                distances = [self.l1_distance(feature_set, centroid) for centroid in self.centroids]
                classification = distances.index(min(distances))
                classifications[classification].append(feature_set)

            prev_centroids = np.array(self.centroids)

            for classification in classifications:
                self.centroids[classification] = np.median(classifications[classification], axis=0)

            optimized = True

            for i, c in enumerate(self.centroids):
                original_centroid = prev_centroids[i]
                current_centroid = self.centroids[i]
                if self.l1_distance(original_centroid, current_centroid) > 0:
                    optimized = False

            if optimized:
                labels = []
                for feature_set in data:
                    distances = [self.l1_distance(feature_set, centroid) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    labels.append(classification)
                return labels

    @staticmethod
    def l1_distance(x, y):
        # Return the Manhattan distance between X and Y
        return np.sum(np.abs(x - y))


def bcubed_score(y, predict):
    """
    Calculate the bcubed precision (Without Merging and labeling the clusters)
    :param y: the real label of the data
    :param predict: the predicted label of the data
    :return: the bcubed precision, bcubed recall, F-score
    """
    precision, recall, f_score = [], [], []
    for index, instance in enumerate(predict):
        # all indices of all instances in the cluster
        cluster_base_index = [i for i, x in enumerate(predict) if x == instance]
        # all instance that have same labels with p
        recall_base_index = [i for i, x in enumerate(y) if x == y[index]]
        # Intersection of the two sets
        intersection_index = list(set(cluster_base_index).intersection(set(recall_base_index)))
        p = len(intersection_index) / len(cluster_base_index)
        r = len(intersection_index) / len(recall_base_index)
        f = 2 * p * r / (p + r)
        precision.append(p)
        recall.append(r)
        f_score.append(f)

    return np.mean(precision), np.mean(recall), np.mean(f_score)


def plot_evaluation(k_schedule, precision, recall, f_score, title, save=False):
    """
    Plot the evaluation results
    :param k_schedule: the k value
    :param precision: the precision list
    :param recall: the recall list
    :param f_score: the f_score list
    :param title: the title of the plot
    :param save: whether to save the plot
    """
    plt.figure()
    plt.plot(k_schedule, precision, label='Precision')
    plt.plot(k_schedule, recall, label='Recall')
    plt.plot(k_schedule, f_score, label='F-Score')
    # plot the highest F-score with corresponding F-score
    plt.plot(k_schedule[np.argmax(f_score)], f_score[np.argmax(f_score)], 'r-o')
    show_max_f_score = 'Max F-Score: %.3f' % f_score[np.argmax(f_score)]
    plt.annotate(show_max_f_score,
                 xy=(k_schedule[np.argmax(f_score)] - 1, f_score[np.argmax(f_score)] - 0.05))
    # plot the highest precision
    plt.plot(k_schedule[np.argmax(precision)], precision[np.argmax(precision)], 'yo')
    show_max_precision = 'Max Precision: %.3f' % precision[np.argmax(precision)]
    plt.annotate(show_max_precision,
                 xy=(k_schedule[np.argmax(precision)] - 1, precision[np.argmax(precision)] + 0.05))
    # plot the highest recall
    plt.plot(k_schedule[np.argmax(recall)], recall[np.argmax(recall)], 'ro')
    plt.title(title)
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.legend()
    if save:
        plt.savefig(title + '.png')
    plt.show()


def l2_norm(x):
    """
    Normalize x with l2 distance
    """
    return x / (np.linalg.norm(x, axis=1, keepdims=True))


def run_k_cluster():
    data_set = load_data()[0]
    y = data_set[:, -1]
    y = np.array([int(i) for i in y])
    x = np.array(data_set[:, :-1])
    x_norm = l2_norm(np.copy(x))

    # Config for Four runs
    config = {
        "K-Means without normalization": [x, KMeans],
        "K-Means with normalization": [x_norm, KMeans],
        "K-Medians without normalization": [x, KMedians],
        "K-Medians with normalization": [x_norm, KMedians],
    }

    for name, (data, algorithm) in config.items():
        precision_list, recall_list, f_score_list = [], [], []

        for k in K_SCHEDULE:
            model = algorithm(k)
            predict = model.fit(data, seed=SEED)
            precision, recall, f_score = bcubed_score(y, predict)

            precision_list.append(precision)
            recall_list.append(recall)
            f_score_list.append(f_score)

            print(
                "-"*10 + " %s " % name + "-"*10 + "\n"
                "{} clusters\n"
                "Precision: {}\n"
                "Recall: {}\n"
                "F-Score: {}\n".format(k, precision, recall, f_score),
                "-"*50
            )

        plot_evaluation(K_SCHEDULE, precision_list, recall_list, f_score_list, title=name, save=True)


# Hyper-parameters
SEED = 1
K_SCHEDULE = [i for i in range(1, 10)]
np.random.seed(SEED)

if __name__ == '__main__':
    run_k_cluster()
