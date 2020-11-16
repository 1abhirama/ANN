import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import csv

#load dataset

def load_data():
    dataset = []
    with open("E202-COMP7117-TD01-00 - clustering") as f:
        reader = csv.reader(f)
        next(reader)
        for data in reader:
            #feature selection
            #special day rate
            if (data[9] == "HIGH"):
                Special_Day_Rate = 2
            elif (data[9] == "NORMAL"):
                Special_Day_Rate = 1
            elif (data[9] == "LOW"):
                Special_Day_Rate = 0
            
            #visitor type
            if (data[15] == "Returning_Visitor"):
                Visitor_Type = 2
            elif (data[15] == "New_Visitor"):
                Visitor_Type = 1
            elif (data[15] =="Other"):
                Visitor_Type = 0

            #Weekend
            if ( data[16]== "TRUE"):
                Weekend = 1
            elif ( data[16]== "FALSE"):
                Weekend = 0

            #product related duration
            Product_Related_Duration = data[5]

            #exit rates
            ExitRates = data[7]

            feature = Special_Day_Rate, Visitor_Type,Weekend,Product_Related_Duration,ExitRates
            dataset.append(feature)
    return dataset

def normalize(dataset, min, max):
    normalized = [(i-min) / (max-min) for i in dataset] 
    return normalized

    max_value = max(dataset)
    min_value = min(dataset)
    dataset = normalize(dataset, max_value, min_value)


def pca(dataset):
    pca = PCA(n_components=3)
    dataset_pca = pca.fit_transform(dataset)
    return dataset_pca

class SOM:
    def __init__(self, width, height, in_dim):
        self.width = width
        self.height = height
        self.in_dim = in_dim

        node = [[x,y] #bikin utk nandain index2 dari setiap inputnya jd buat 2 dimensi(x,y)
            for y in range(self.height)
            for x in range(self.width)
        ]
        self.node = tf.to_float(node)

        #buat weight
        self.weight = tf.Variable(tf.random_normal(
            [self.width*self.height, in_dim] 
        ))

        self.x = tf.placeholder(tf.float32, [self.in_dim])
        winning_node = self.get_bmu(self.x) #bmu = best matching unit

        self.update = self.update_w(winning_node, self.x)
    
    def get_bmu(self, x): 
        expanded_x = tf.expand_dims(x, 0) #expand biar dia bisa dikurang dengan self weight, krn pd saat awal placeholdernya cuma 1D
        node_diff = tf.square(tf.subtract(expanded_x, self.weight))
        node_dist = tf.reduce_sum(node_diff, 1) #reduce sum kalo fiturnya lebih dari 1, tp kalo fiturnya kurang dari 1, dijumlahin semua
        winning_index = tf.argmin(node_dist, 0) #cari yang paling kecil, ini buat ngnitung distancenya
        winning_loc = tf.stack([
            tf.mod(winning_index, self.width),
            tf.div(winning_index, self.width)
        ])
        return tf.to_float(winning_loc)

    def update_w(self, winning_node, x):
        lr = .5
        expanded_bmu = tf.expand_dims(winning_node, 0) 
                                                        
        diff_to_winning = tf.square(tf.subtract(expanded_bmu, self.node))
        dist_to_winning = tf.reduce_sum(diff_to_winning, 1)
        sigma = tf.to_float(tf.maximum(self.width, self.height))/2. 
        NS = tf.exp(-tf.div(tf.square(dist_to_winning), 2*tf.square(sigma)))
		
        rate = tf.multiply(NS, lr)
        count_node = self.width*self.height
        rf = tf.stack(
            [tf.tile(tf.slice(rate, [i], [1]), [self.in_dim])   
                for i in range(count_node)
            ]
        )
        x_w = tf.subtract(tf.stack([x for i in range(count_node)]), self.weight)
        w_new = tf.add(self.weight, tf.multiply(rf, x_w))

        return tf.assign(self.weight, w_new)

    def train(self, dataset, epochs):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                for data in dataset:
                    feed = {
                        self.x: data
                    }
                    sess.run([self.update], feed)
                w = list(sess.run(self.weight))
                n = list(sess.run(self.node))
                cluster = [[] for i in range(self.width)]
                for i, loc in enumerate(n):
                    cluster[int(loc[0])].append(w[i])
                self.cluster = cluster 

dataset = load_data()
som = SOM(9, 9, 3)
som.train(dataset, 5000)

plt.imshow(som.cluster)
plt.show()    