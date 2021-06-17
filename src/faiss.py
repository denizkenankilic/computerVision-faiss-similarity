import faiss
from PIL import Image
import numpy as np
import imagehash
import os

def collect_image_signatures(archivedImagesPath,
                             imagesToBeCheckedPath,
                             validImageExtensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"),
                             DIMENSION = 64,
                             hashfunc = imagehash.phash):
    """
    A function to collect hash signatures of images in the folder
    archivedImagesPath and imagesToBeCheckedPath.
    """
    imageNames = [f for f in os.listdir(archivedImagesPath) if not f.startswith('.')]
    imagePaths = [os.path.join(archivedImagesPath, k) for k in imageNames]

    hashArray = np.zeros(shape=(len(imagePaths), DIMENSION)).astype("float32")
    for i in range(len(imagePaths)):
        # imageName = os.path.splitext(imageNames[i])[0]
        imageExtension = os.path.splitext(imageNames[i])[1] # File extensions
        if imageExtension in validImageExtensions:
            hashSignature = hashfunc(Image.open(imagePaths[i]))
            hashSignature = ((hashSignature.hash).reshape(1, len(hashSignature))).astype("float32")
            hashArray[i] =  hashSignature

    imageNamesToBeChecked = [f for f in os.listdir(imagesToBeCheckedPath) if not f.startswith('.')]
    imagePathsToBeChecked = [os.path.join(imagesToBeCheckedPath, k) for k in imageNamesToBeChecked]

    hashArrayToBeChecked = np.zeros(shape=(len(imagePathsToBeChecked), DIMENSION)).astype("float32")
    for i in range(len(imagePathsToBeChecked)):
            # imageNameToBeChecked = os.path.splitext(imageNamesToBeChecked[i])[0]
            imageExtension = os.path.splitext(imageNamesToBeChecked[i])[1] # File extensions
            if imageExtension in validImageExtensions:
                hashSignature = hashfunc(Image.open(imagePathsToBeChecked[i]))
                hashSignature = ((hashSignature.hash).reshape(1, len(hashSignature))).astype("float32")
                hashArrayToBeChecked[i] =  hashSignature
    return hashArray, hashArrayToBeChecked, imageNames, imageNamesToBeChecked

def check_faiss_similarity(hashArray,
                           hashArrayToBeChecked,
                           imageNames,
                           imageNamesToBeChecked,
                           index_method = "Flat",
                           metric = faiss.METRIC_L2,
                           isDataBatchingEnabled = True,
                           isIndexWritingEnabled = True,
                           isSeperateBatchSearchingEnabled = True,
                           batchSize = 5000,
                           validImageExtensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"),
                           SIMILARITY_THR = 0.40,
                           NUMBER_OF_KNN = 10):
    """
    A function to compare hash signatures, which are collected in the
    'collect_image_signatures' function, with respect to the
    Faiss index.
    """
    size, dimension = hashArray.shape
    index = faiss.index_factory(dimension, index_method, metric)

    if isDataBatchingEnabled == True:
        # numberOfbatch = math.ceil(hashArray.shape[0] / batchSize)
        batcheslist = [hashArray[i:i + batchSize] for i in range(0, len(hashArray), batchSize)]

        similarImagesResult = {}
        if isSeperateBatchSearchingEnabled == True:
            for i in range(len(batcheslist)):
                index = faiss.index_factory(dimension, index_method, metric)
                index.train(batcheslist[0])
                index.add(batcheslist[i])
                if isIndexWritingEnabled == True:
                    faiss.write_index(index, "vector_" + str(i) + ".index")

                distances, indexes = index.search(hashArrayToBeChecked, NUMBER_OF_KNN)
                normDistances = distances / dimension
                indexes += (i * batchSize)

                thresholdedNormDistances = np.where(normDistances < SIMILARITY_THR)
                coordinatesInBatch = list(zip(thresholdedNormDistances[0], thresholdedNormDistances[1]))

                for j in range(len(imageNamesToBeChecked)):
                    similarImagesResult[os.path.splitext(imageNamesToBeChecked[j])[0] + "_batch_" + str(i)] = []
                    for k in range(len(coordinatesInBatch)):
                        if j == coordinatesInBatch[k][0]:
                            similarImagesResult[os.path.splitext(imageNamesToBeChecked[j])[0] + "_batch_" + str(i)].append(os.path.splitext(imageNames[indexes[coordinatesInBatch[k][0]][coordinatesInBatch[k][1]]])[0])
                            similarImagesResult[os.path.splitext(imageNamesToBeChecked[j])[0] + "_batch_" + str(i)].append(normDistances[coordinatesInBatch[k][0]][coordinatesInBatch[k][1]])

                index.reset()
        else:
            index.train(batcheslist[0])
            for i in range(len(batcheslist)):
                index.add(batcheslist[i])
            if isIndexWritingEnabled == True:
                faiss.write_index(index, "vector.index")
            distances, indexes = index.search(hashArrayToBeChecked, NUMBER_OF_KNN)
            normDistances = distances / dimension
            thresholdedNormDistances = np.where(normDistances < SIMILARITY_THR)
            coordinatesInBatch= list(zip(thresholdedNormDistances[0], thresholdedNormDistances[1]))

            for j in range(len(imageNamesToBeChecked)):
                similarImagesResult[os.path.splitext(imageNamesToBeChecked[j])[0]] = []
                for k in range(len(coordinatesInBatch)):
                    if j == coordinatesInBatch[k][0]:
                        similarImagesResult[os.path.splitext(imageNamesToBeChecked[j])[0]].append(os.path.splitext(imageNames[indexes[coordinatesInBatch[k][0]][coordinatesInBatch[k][1]]])[0])
                        similarImagesResult[os.path.splitext(imageNamesToBeChecked[j])[0]].append(normDistances[coordinatesInBatch[k][0]][coordinatesInBatch[k][1]])
    else:
        index.add(hashArray)
        if isIndexWritingEnabled == True:
            faiss.write_index(index, "vector.index")
        D, I = index.search(hashArrayToBeChecked, NUMBER_OF_KNN)
        normalizedDistances = D / dimension

        imageIndexesThresholded = np.where(normalizedDistances < SIMILARITY_THR)
        listOfCoordinates= list(zip(imageIndexesThresholded[0], imageIndexesThresholded[1]))
        similarImagesResult = {}
        for j in range(len(imageNamesToBeChecked)):
            similarImagesResult[os.path.splitext(imageNamesToBeChecked[j])[0]] = []
            for k in range(len(listOfCoordinates)):
                if j == listOfCoordinates[k][0]:
                    similarImagesResult[os.path.splitext(imageNamesToBeChecked[j])[0]].append(os.path.splitext(imageNames[I[listOfCoordinates[k][0]][listOfCoordinates[k][1]]])[0])
                    similarImagesResult[os.path.splitext(imageNamesToBeChecked[j])[0]].append(normalizedDistances[listOfCoordinates[k][0]][listOfCoordinates[k][1]])
    return similarImagesResult  
