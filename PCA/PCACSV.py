import pandas as pd 
from sklearn.decomposition import PCA

#load dataset
def load_dataset():
    df = pd.read_csv("Iris.csv")
    features = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]] 
    return features

#apply PCA
def apply_pca(dataset):
    pca = PCA(n_components=2)#reduksi ke 2 dimensi
    result = pca.fit_transform(dataset)
    return result
    

dataset = load_dataset()
print(dataset)
pca_dataset = apply_pca(dataset)
print(pca_dataset)
#print(pca_dataset.shape)