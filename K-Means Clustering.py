import numpy as np
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import time


def kmeans_images(img, k):
    # Reshape the image into a 2D array
    data = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    if data.shape[1] > 3:
        # slicing the image
        data = data[:, :3]
    # K-Means Clustering function call
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    # Transform the Clusters labels into a 2D array
    labels = kmeans.labels_.reshape(img.shape[0], img.shape[1])
    # Clusters centers
    centers = kmeans.cluster_centers_
    return labels, centers


def creat_image(labels, centers):
    # creat an image with zeros
    img = np.zeros(shape=(labels.shape[0], labels.shape[1], 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = centers[labels[i, j]]
    if(img.max() > 1):
        # matplot lib works only with pixels colors ranging from 0 to 1
        img /= 255
    # saving the resulting image
    mpimg.imsave('Image Result.jpg', img)
    return img


def compactness(img, labels, centers):
    WSS = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            WSS += np.sum(np.power(img[i, j]-centers[labels[i, j]], 2))
    return WSS/(labels.shape[0]*labels.shape[1])


def separation(labels, centers):
    BSS = 0
    cluster_size = np.zeros(centers.shape[0])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            cluster_size[labels[i, j]] += 1
    mean = np.mean(centers, axis=0)
    for k in range(centers.shape[0]):
            BSS += cluster_size[k]*np.sum(np.power(mean - centers[k], 2))
    return BSS/(labels.shape[0]*labels.shape[1])


clusters = 4
start_time = time.clock()
img = mpimg.imread('colorcube-1_66.jpg')
labels, centers = kmeans_images(img, clusters)
creat_image(labels, centers)
WSS = compactness(img, labels, centers)
BSS = separation(labels, centers)
elapsed_time = time.clock() - start_time
print("compactness = ", WSS, "| separation = ", BSS)
print("elapsed time : {:0.3f} seconds".format(elapsed_time))
