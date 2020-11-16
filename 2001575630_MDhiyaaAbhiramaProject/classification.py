import tensorflow as tf
import numpy as np
import csv
from sklearn.decomposition import PCA
import random
 
result = []
def loadData(path):
	dataset = []
	label = []
	with open(path) as d:
		read = csv.reader(d)
		next(read)
		for data in read:
			features = data[5:11]
			features.append(data[12])
			features.extend(data[14:16])
			features.extend(data[17:21])
			features.append(data[23])

			result = data[4]
			label.append(result)
			dataset.append(features)
	return dataset,label
		
def applyPCA(data):
	pca = PCA(n_components = 5)
	dataset_pca = pca.fit_transform(data)
	return dataset_pca

def dataPCA_Label(dataPCA, label):
	dataBest5 = []
	for data in range(len(dataPCA)):
		#for dataresult in result:
		#	dataBest5.append((data,dataresult))
		dataBest5.append((dataPCA[data],label[data]))

	return dataBest5	

result_list = ['a','b','c','d','e'] # kalo misal dia high, jadinya begini outputnya -> [1,0,0]

def preProcessing_output(label):
	new_label = []
	for result in label:
		result_index = result_list.index(result)
		new_result = np.zeros(len(result_list), 'int')
		new_result[result_index] = 1
		new_label.append(new_result)
	return new_label

# = Load Data =
dataset,label = loadData("E202-COMP7117-TD01-00 - classification.csv")

#normalisasi
def normalize(dataset, min, max): #di normalize karena ingin di ubah nilainya dari 0 - 1. biar gak timpang
    temp_normalize = []
    dataset_new = []
    for data in dataset:
        for feature in range(len(data)):
            normalized = [(feature-float(min[feature])) / (float(max[feature])-float(min[feature]))] #cara comprehension 
            temp_normalize.append(normalized)
        dataset_new.append(temp_normalize)
    return dataset_new

max_value = max(dataset)
min_value = min(dataset)

#dataset = normalize(dataset, min_value, max_value)

#print(dataset)

# = PCA dan Label=
data = np.array(dataset)
dataset_pca = applyPCA(data)
label = preProcessing_output(label)

# = DataPCA - Label =
dataPCALabel = dataPCA_Label(dataset_pca, label)

#print(dataPCALabel)

random.shuffle(dataPCALabel)

counter = int(.7 * len(dataPCALabel))
counter2 = int(.2 * len(dataPCALabel))
slicing = counter + counter2

train_dataset = dataPCALabel[:counter] #dari data awal sampai counter
#print(" =================== TRAIN DATASET")
#print(train_dataset)
test_dataset = dataPCALabel[counter:slicing] #dari data counter sampai akhir
#print(" =================== TEST DATASET")
#print(test_dataset)
eval_dataset = dataPCALabel[slicing:]
#print(" ================== EVAL DATASET")
#print(eval_dataset)

#forward pass
def fully_connected(input, numinput, numoutput):
    #buat weight dan bias
    w = tf.Variable(tf.random_normal([numinput, numoutput])) #shape tergantung input sam output
    b = tf.Variable(tf.random_normal([numoutput])) #hanya bergantung sama output aja
    
    #weight akhir setelah dihitung // linier combination
    wx_b = tf.matmul(input,w)+b

    act = tf.nn.sigmoid(wx_b) # ini adalah fungsi aktifasi berupa sigmoid
    return act

#Architecture
num_of_input = 5
num_of_output = 5
num_of_hidden = [3,4]

#Membuat model // arsitektur
def build_model(input):
    layer1 = fully_connected(input, num_of_input, num_of_hidden[0]) #input ke hidden layer 1
    layer2 = fully_connected(layer1, num_of_hidden[0], num_of_hidden[1]) #hidden layer 1 ke hidden layer 2
    layer3 = fully_connected(layer2, num_of_hidden[1], num_of_output) #hidden layer 2 ke output
    return layer3

#variable untuk training input dan output
trn_input = tf.placeholder(tf.float32, [None, num_of_input])
trn_target = tf.placeholder(tf.float32, [None, num_of_output])

#learning rate, epoch, report between
learning_rate = .1 #kecil lama, besar gak akurat
num_of_epoch = 5000+1
report_between = 100 #report tiap interval 3000 kali

save_dir = "./ann-model/"
filename = "ann.ckpt"

#TRAINING
def optimize(model, dataset, test_dataset): #minta model juga
    #hitung error by rumus 
    error = tf.reduce_mean(.5 * (trn_target - model)**2) #training nya banyak, ambil rata2 dari training
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error) #ini backward nya // update weight nya
    correct_prediction = tf.equal(tf.argmax(model,1),tf.argmax(trn_target,1)) #ngecek result apa sama atau nggak
    #cek akurasi
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #di ubah tipe datanya bias bisa di reduce mean
    saver = tf.train.Saver(tf.global_variables())
    error_temp = 9999999999

    with tf.Session() as sess: 
        #harus validasi semua variabel tensor flow dulu
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_of_epoch):
            feature = [data[0] for data in dataset]
            result = [data[1] for data in dataset]
            feed = {trn_input: feature, trn_target:result}
            _,error_value,accuracy_value = sess.run([optimizer, error, accuracy], feed)

            feature2 = [data[0] for data in test_dataset]
            result2 = [data[1] for data in test_dataset]
            
            feed2 = {trn_input: feature2, trn_target:result2}
            error_value2,accuracy_value2 = sess.run([error, accuracy], feed2)

            if epoch % report_between == 0:
                print()
                print("Train - Epoch : ",epoch, ", Error : ",error_value*100,", Accuarcy : ", accuracy_value*100)

			#TEST/VALIDATION DATA
            if epoch >= 500:
                if epoch % 500 == 0:
                    if error_value2 < error_temp:
                        error_temp = error_value2
                        saver.save(sess, save_dir + filename + str(epoch))

                if epoch % report_between == 0:
                    print("Test - Epoch : ",epoch, ", Error : ",error_value2*100,", Accuarcy : ", accuracy_value2*100)


#validation -> dilakukan setiap iterasi cek nya
#testing, dilakkukan sekali aja cek nya

#TESTING

def testing_model(model, eval_dataset):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        correct_prediction = tf.equal(tf.argmax(model,1),tf.argmax(trn_target,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        error = tf.reduce_mean(.5 * (trn_target - model)**2) #training nya banyak, ambil rata2 dari training
    

        #load model
        saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(len(eval_dataset)):
            feature = [data[0] for data in eval_dataset]
            result = [data[1] for data in eval_dataset]
            feed = {trn_input : feature, trn_target:result}
            error_value,accuracy_value = sess.run([error, accuracy], feed)

            if i % report_between == 0:
                print("Epoch - ",i, ", Accuracy : ", accuracy_value*100)
            
         
model = build_model(trn_input)
optimize(model,train_dataset, test_dataset)
testing_model(model, eval_dataset)

#print(dataset)