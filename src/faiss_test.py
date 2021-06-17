import unittest
from unittest import TestCase
import faiss
import imagehash
import os, sys

from src.faiss import collect_image_signatures, check_faiss_similarity

class TestMethods(TestCase):

    longMessage = True

    def test_pHash(self):

        archivedImagesPath = "tests/archived_test_image"
        imagesToBeCheckedPath = "tests/archived_images_to_be_checked"

        hashArray, hashArrayToBeChecked, imageNames, imageNamesToBeChecked = collect_image_signatures(archivedImagesPath,
                                                                                              imagesToBeCheckedPath,
                                                                                              validImageExtensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"),
                                                                                              DIMENSION = 64,
                                                                                              hashfunc = imagehash.phash)

        similarImagesResult = check_faiss_similarity(hashArray, 
                           hashArrayToBeChecked,
                           imageNames, 
                           imageNamesToBeChecked,
                           index_method = "Flat",
                           metric = faiss.METRIC_L2,
                           isDataBatchingEnabled = False,
                           isIndexWritingEnabled = True,
                           isSeperateBatchSearchingEnabled = True,
                           batchSize = 5000,
                           validImageExtensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"), 
                           SIMILARITY_THR = 0.40,  
                           NUMBER_OF_KNN = 1)

        normalizedDistanceFloat = similarImagesResult['MonaLisa_WikiImages'][1]

        self.assertEqual(normalizedDistanceFloat, 0.0625, "Result should be 0.0625")

if __name__ == "__main__":
     unittest.main()
