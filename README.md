# Data Cluster
## COMP337 Assignment 2: implementing the k-means and k-medians clustering algorithms

## 1. k-means and k-medians clustering with B-CUBED evaluation
This project is a simple implementation of k-means and k-medians. The k-means and k-medians algorithm can be used to cluster data points into k clusters. 
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
├── K-Medians with normalization.png
├── K-Medians without normalization.png
├── Data_Cluster.py
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

`python3 Data_Cluster.py`

The program will run in the background and will print the results to the console.
The plot results will be saved into four .png files.

## 4. Default Hyperparameter

```
SEED = 1                                     # Random seed for reproducibility
K_SCHEDULE = [i for i in range(1, 10)]       # The K values will be used in the k-means and k-medians algorithm
```

## 5. Results

The terminal output for each algorithm in different K with or without normalization is as follows:

```
--- K-Medians(K-Means) with(without) normalization ---
n(1-9) clusters
Precision: 0.919546511597592
Recall: 0.516843360757367
F-Score: 0.6241404710659718
-------------------------------------------------------
```
Meanwhile, the plot results are saved into four .png files.

## 6. Author
Name: Wuwei Zhang ([@LANNDS18](https://github.com/LANNDS18))

Email: sgwzha23@liverpool.ac.uk

Student ID: 201522671
