import os
from PIL import Image
from torchvision import transforms

# 1. Rescaling
# needed input dimensions for the CNN
inputDim = (224,224)
inputDir = "originalimgs"
inputDirCNN = "inputImagesCNN"

os.makedirs(inputDirCNN, exist_ok = True)

transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])

for imageName in os.listdir(inputDir):
    if imageName.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp')):
        I = Image.open(os.path.join(inputDir, imageName))
        # print(I)
        newI = transformationForCNNInput(I)

        # copy the rotation information metadata from original image and save, else your transformed images may be rotated
        exif = I.info['exif']
        newI.save(os.path.join(inputDirCNN, imageName), exif=exif)

        newI.close()
        I.close()
    else:
        print('non-jpg/png file format detected, skipped:', imageName)
        continue

import torch
from tqdm import tqdm
from torchvision import models

# 2. Creating the similarity matrix with Resnet18 (from generated imgs in part1)
class Img2VecResnet18():
    def __init__(self):

        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()

        # normalize the resized images as expected by resnet18
        # [0.485, 0.456, 0.406] --> normalized mean value of ImageNet, [0.229, 0.224, 0.225] std of ImageNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def getVec(self, img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)

        def copyData(m, i, o): embedding.copy_(o.data)

        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()

        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):

        cnnModel = models.resnet18(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512

        return cnnModel, layer


# generate vectors for all the images in the set
img2vec = Img2VecResnet18()

allVectors = {}
print("Converting images to feature vectors:")
for image in tqdm(os.listdir(inputDirCNN)):
    if image.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp')):
        I = Image.open(os.path.join(inputDirCNN, image))
        vec = img2vec.getVec(I)
    #     print(vec)
        allVectors[image] = vec
        I.close()
    else:
        print('non-jpg/png file format detected, skipped:', imageName)
        continue


# 3. Cosine Similarity
# now let us define a function that calculates the cosine similarity entries in the similarity matrix
import pandas as pd
import numpy as np

def getSimilarityMatrix(vectors):
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / ((np.linalg.norm(v, axis=0).reshape(-1,1)) * ((np.linalg.norm(v, axis=0).reshape(-1,1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim, columns = keys, index = keys)

    return matrix

similarityMatrix = getSimilarityMatrix(allVectors)

# 4. Top k lists

from numpy.testing import assert_almost_equal
import pickle

k = 11 # the number of top similar images to be stored

similarNames = pd.DataFrame(index = similarityMatrix.index, columns = range(k))
similarValues = pd.DataFrame(index = similarityMatrix.index, columns = range(k))

for j in tqdm(range(similarityMatrix.shape[0])):
    kSimilar = similarityMatrix.iloc[j, :].sort_values(ascending = False).head(k)
    similarNames.iloc[j, :] = list(kSimilar.index)
    similarValues.iloc[j, :] = kSimilar.values

similarNames.to_pickle("similarNames.pkl")
similarValues.to_pickle("similarValues.pkl")

# 5. Get & Visualize images
import matplotlib.pyplot as plt

def setAxes(ax, image, query = False, **kwargs):
    value = kwargs.get("value", None)
    if query:
        ax.set_xlabel("Selected photo\n{0}".format(image), fontsize = 8)
    else:
        ax.set_xlabel("Similarity score {1:1.3f}\n{0}".format( image,  value), fontsize = 8)
    ax.set_xticks([])
    ax.set_yticks([])

def getSimilarImages(image, simNames, simVals):
    if image in set(simNames.index):
        imgs = list(simNames.loc[image, :])
        vals = list(simVals.loc[image, :])
        if image in imgs:
            assert_almost_equal(max(vals), 1, decimal = 5)
            imgs.remove(image)
            vals.remove(max(vals))
        return imgs, vals
    else:
        print("'{}' Unknown image".format(image))


def plotSimilarImages(image, similarNames, similarValues, num_similar):
    simImages, simValues = getSimilarImages(image, similarNames, similarValues)
    total_images = num_similar + 1  # Including the query image
    fig = plt.figure(figsize=(2 * total_images, 4))  # Adjust figure size dynamically

    # Plot the query image
    ax = fig.add_subplot(1, total_images, 1)
    query_img = Image.open(os.path.join(inputDirCNN, image))
    setAxes(ax, image, query=True)
    plt.imshow(query_img.convert('RGB'))
    query_img.close()

    # Plot the similar images
    for j in range(1, num_similar + 1):
        if j < len(simImages):
            ax = fig.add_subplot(1, total_images, j + 1)
            similar_img = Image.open(os.path.join(inputDirCNN, simImages[j-1]))
            setAxes(ax, simImages[j-1], value=simValues[j-1])
            plt.imshow(similar_img.convert('RGB'))
            similar_img.close()

    plt.show()

# 5. Invoke code
import random

# Number of example images to plot
num_examples = 1

# Number of similar
num_similar = 10

# Get all processed image names from inputDirCNN
processed_images = os.listdir(inputDirCNN)

# Ensure that there are enough images for the examples
if len(processed_images) < num_examples:
    raise ValueError("Not enough processed images. Found {}, need {}".format(len(processed_images), num_examples))

# Randomly select a subset of images aka selected image
inputImages = random.sample(processed_images, num_examples)

# Plotting configuration
numCol = 11
numRow = 1

# Plot the similar images for the selected examples
for image in inputImages:
    plotSimilarImages(image, similarNames, similarValues, num_similar)