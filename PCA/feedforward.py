import tensorflow as tf
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

#read csv
def load_data():
    df = pd.read_csv("LifeExpectancy.csv")
    target = df[["Life Expectancy"]]
    features = df[["Gender", "Residential","Physical Activity (times per week)", "Happiness"]]
    return features, target

features, target = load_data()
#buat object, utk ngubah yang ada di csv yang valuenya bukan angka menjadi angka
ordinal_encoder = OrdinalEncoder() 
features[["Gender","Residential"]] = ordinal_encoder.fit_transform(features[["Gender","Residential"]])

#normalisasi data
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

#convert target di csv yang bukan angka menjadi angka
one_hot_encoder = OneHotEncoder(sparse=False) #supaya array
target = one_hot_encoder.fit_transform(target)

#ngesplit train data dan test data
train_data, test_data, train_target, test_target = train_test_split(features, target, test_size = 0.2)

#model architecture
layer = {
    "input": 4,
    "hidden": 4,
    "output": 3,
} #init layer

weights = {
    "input_hidden" : tf.Variable(tf.random_normal([layer['input'], layer['hidden']])),
    "hidden_output" : tf.Variable(tf.random_normal([layer['hidden'],layer['output']]))
}#init weight

bias = {
    "input_hidden": tf.Variable(tf.random_normal([layer['hidden']])),
    "hidden_output": tf.Variable(tf.random_normal([layer['output']]))
} #bias

learning_rate = 0.1 #learning rate

#buat placeholder
input_placeholder = tf.placeholder(tf.float32, [None, layer['input']])
output_placeholder = tf.placeholder(tf.float32, [None, layer['output']])

#fungsi buat feedforward
def feed_forward():
    #input -> hidden
    Wx1 = tf.matmul(input_placeholder, weights['input_hidden'])
    Wx1b = Wx1 + bias['input_hidden']
    Wx1bA = tf.nn.sigmoid(Wx1b)

    #hidden -> output
    Wx2 = tf.matmul(Wx1bA, weights['hidden_output'])
    Wx2b = Wx2 + bias['hidden_output']
    Wx2bA = tf.nn.sigmoid(Wx2b)
    return Wx2bA

predict = feed_forward()

#hitung error MSE
loss = tf.reduce_mean(0.5*(output_placeholder - predict) ** 2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#update
train = optimizer.minimize(loss)

#training
epoch = 3000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #set data -> placeholder
    train_dict = {
        input_placeholder: train_data,
        output_placeholder: train_target,
    }

    for i in range(epoch):
        sess.run(train, feed_dict=train_dict)

        error = sess.run(loss, feed_dict=train_dict)
        #print error
        if i % 200 == 0:
            print("Iteration {} error: {}".format(i,error))
    #axis = 1 buat checknya row
    matches = tf.equal(tf.argmax(output_placeholder, axis = 1), tf.argmax(predict, axis = 1))
    floatMatches = tf.cast(matches, tf.float32)
    acc = tf.reduce_mean(floatMatches)

    test_dict = {
        input_placeholder: test_data,
        output_placeholder: test_target,
    }

    print("Accuracy {}".format(sess.run(acc, feed_dict = test_dict )*100))