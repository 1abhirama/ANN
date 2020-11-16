# PCA
# Principal Component Analysis
# 4000 feature -> 2 feature

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#column = faces
#row = pixels

#row = faces
#column = pixels
#before
#[1 , 2]
#[3 , 4]

#after transposed
#[1 , 3]
#[2 , 4]


#load dataset
def load_dataset():
    dataset = loadmat('olivettifaces.mat')
    face_images = dataset['faces']
    transpose = np.transpose(face_images)
    return face_images, transpose.astype(float)

#column = faces
#row = pixels

def imageshow(face):
    #4096
    # each image = 64 x 64
    image = face.reshape(64, 64)
    image = np.transpose(image)
    plt.imshow(image)
    plt.show()
    

#check data
original, dataset = load_dataset()
#imageshow(dataset[0])
#print(dataset)

#PCA
#     pixel
#face
#       |
#       V
# 1. dapatkan mean dari setiap pixel
def get_mean(dataset):
    return tf.reduce_mean(dataset, axis=0)

mean = get_mean(dataset)



# 2. kurangi setiap pixel dengan mean
# normalize
def normalize(dataset, mean):
    return dataset - mean

norm_dataset = normalize(dataset, mean)


# 3. dapatkan covariance
# formula covariance = A * AT (matrix * matrix Transpose)
def get_covariance(norm_dataset):
    # 400 * 4096 ->  400 * 400
    # 4096 * 400 - > 4096 * 4096
    #jgn kebalik dimensinya

    return tf.matmul(norm_dataset, tf.transpose(norm_dataset))
#size covariance matrix 400*400

covariance = get_covariance(norm_dataset)

# 4. dari covariance dapatkan eigen vector
def get_eigen_vector(covariance):
    # eigen_value, eigen_vector
    eigen_value, eigen_vector = tf.self_adjoint_eig(covariance)
    #sorting descending
    eigen_vector = tf.reverse(eigen_vector, [1])
    return eigen_vector

eigen_vector = get_eigen_vector(covariance)

# 5. dari eigen vector dapatkan eigen face
# 400 * 400
# 400 * 4096
# formula eigen_face = (A* eigen_vector)

def get_eigen_face(dataset, eigen_vector):
    #transpose dulu biar bs diitung
    dataset = tf.transpose(dataset)
    eigen_face = tf.matmul(dataset, eigen_vector)
    eigen_face = tf.transpose(eigen_face)
    return eigen_face

eigen_face = get_eigen_face(norm_dataset, eigen_vector)

with tf.Session() as sess: 
    result = sess.run(eigen_face)

imageshow(result[0])