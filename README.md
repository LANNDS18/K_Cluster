# Data Cluster
## COMP338 Assignment 2: implementing the k-means and k-medians clustering algorithms

## 1. k-means and k-medians clustering with B-CUBED evaluation
This project is a simple implementation of k-means and k-medians. The k-means and k-median algorithm can be used to cluster data points into k clusters. 
The k-means and k-medians algorithm are greedy algorithms that assign each data point to the cluster that is closest to them. 
The dataset used is the unzipped CA2data.zip file. The CA2data folder contains four classes of files.

The structure of this project is as follows:
```
.
├── CA2data
│   ├── animals
│   ├── countries
│   ├── fruits
│   └── veggies
├── K-Means with normalization.png
├── K-Means without normalization.png
├── K-Median with normalization.png
├── K-Median without normalization.png
├── K_Cluster.py
└── README.md

```

Four .png files are created after running the B-CUBED evaluation algorithm on the k-means and k-medians algorithm.


## 2. Dependencies

- Python 3.6 or later (implemented on 3.9)

Install python: `sudo apt-get install python3.9`

Install pip: `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`

- numpy 1.14.x or later

Install numpy: `python3 -m pip install numpy`

-matplotlib 1.5.x or later

Install matplotlib: `python3 -m pip install matplotlib`

## 3. Run
After setting up the environment as last section shows, run the following command to run the program:

`python3 main.py`

The program will run in the background and will print the results to the console.

## 4. Default Hyperparameter

```
np.random.seed(1)                       # Shuffle the data with the same seed to make the results reproducible
LEARNING_RATE = 1e-2                    # Learning rate for training single-layer perceptron
EPOCH = 20                              # Number of epochs for training
L2_REG = [0.01, 0.1, 1.0, 10.0, 100.0]  # 5 different L2 regularization parameters
SHUFFLE = True                          # Shuffle the data before each epoch which is used to avoid overfitting
```

## 5. Results

The result is output to the console as the sequence of question. The format is as follows:

```
Information
 ----------------------------------------------------------------------------------------------------
The number of weight/bias updates: n
...
Train accuracy between a and b: xx.xx% (for binary classification)
...
Overall train accuracy: xx.xx% (for multiclass classification)
...
----------------------------------------------------------------------------------------------------
Test accuracy between a and b: xx.xx% (for binary classification)
...
Overall test accuracy: xx.xx% (for both binary and multiclass classification)
```
The number of weight/bias updates is the number of updates of the weight/bias in the training process, which is used to 
determine how difficult the training is. The accuracy is the percentage of the correct predictions. Training accuracy
can reflect the fitting of the model to the training data. The test accuracy is the accuracy of the model on the test.

For question 3, the overall test accuracy is the average accuracy for the all binary pairs. For question 4 and 5, overall
test accuracy is the average accuracy is the accuracy between all y_true and y_pred for all classes.

## 6. Author
Name: Wuwei Zhang ([@LANNDS18](https://github.com/LANNDS18))

Email: sgwzha23@liverpool.ac.uk

Student ID: 201522671
