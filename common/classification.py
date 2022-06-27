import numpy as np
import scipy.spatial.distance as ds
from common.features import BagOfWords, IdentityFeatureTransform


class KNNClassifier(object):

    def __init__(self, k_neighbors, metric):
        """Initialisiert den Klassifikator mit Meta-Parametern
        
        Params:
            k_neighbors: Anzahl der zu betrachtenden naechsten Nachbarn (int)
            metric: Zu verwendendes Distanzmass (string), siehe auch scipy Funktion cdist 
        """
        self.__k_neighbors = k_neighbors
        self.__metric = metric
        # Initialisierung der Membervariablen fuer Trainingsdaten als None. 
        self.__train_samples = None
        self.__train_labels = None

    def estimate(self, train_samples, train_labels):
        """Erstellt den k-Naechste-Nachbarn Klassfikator mittels Trainingdaten.
        
        Der Begriff des Trainings ist beim K-NN etwas irre fuehrend, da ja keine
        Modelparameter im eigentlichen Sinne statistisch geschaetzt werden. 
        Diskutieren Sie, was den K-NN stattdessen definiert.
        
        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing
        
        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        self.__train_samples = train_samples
        self.__train_labels = train_labels

    def classify(self, test_samples):
        """Klassifiziert Test Daten.
        
        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            
        Returns:
            test_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
        
            mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        distances = ds.cdist(self.__train_samples, test_samples, metric=self.__metric)
        # Indecies der k-kleinste Distances
        # https://www.pythonpool.com/numpy-argpartition/
        # argpartition partitioniert so, dass die k-klienste Elemente am anfang gespeichtert werden
        indeces = np.argpartition(distances, self.__k_neighbors, axis=0)[:self.__k_neighbors].T
        # Labels der Indeices
        k_test_labels = np.array(self.__train_labels[indeces])

        from common.features import BagOfWords
        # Remove single-dimensional entries from the shape of an array.
        # wegen most_freq_words vorbereiten
        test_labels = np.squeeze(k_test_labels.reshape(1, len(test_samples), self.__k_neighbors))

        if self.__k_neighbors == 1:
            return test_labels.reshape(len(test_samples), 1)
        else:
            return np.array([BagOfWords.most_freq_words(word_list=row, n_words=1) for row in test_labels])

    def classify1k(self, test_samples):
        distances = ds.cdist(self.__train_samples, test_samples, metric=self.__metric)
        test_labels_indices = np.argmin(distances, axis=0)
        return self.__train_labels[test_labels_indices]
