import numpy as np
import scipy.spatial.distance as ds
from common.features import BagOfWords, IdentityFeatureTransform
import scipy.spatial.distance


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
    from common.features import BagOfWords
    def classify2(self, test_samples):
        """Klassifiziert Test Daten.
        
        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            
        Returns:
            test_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
        
            mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        
       
        if self.__train_samples is None or self.__train_labels is None:
            raise ValueError('Classifier has not been "estimated", yet!')
        train_samples =self.__train_samples
        train_labels = self.__train_labels
        
        k = self.__k_neighbors
        metrik = self.__metric
        #print train_labels
        #cdist berechnet den abstand zwischen die paare vom array 
        #jedes array oder zeile wird mit anderen gepaart und abstand gerechnet 
        distanz = scipy.spatial.distance.cdist(test_samples, train_samples, metrik)
        # argsort gibt die index in sortierte reihenfolge 
        #in sortiert wird der erste index mit den kleinsten abstand gespeichert also index vom kleinsten abstand  
        sortiert = np.argsort(distanz, axis = 1)[:,:k]
        #copy_train_label als 1 D array alle klassifikationen von trainlabels in copy als 1 d array 
        copy_train_labels = train_labels.ravel()
        #jz nehmen wir train labels in den indexen wo wir den 1ten nachbarn fanden 
        test_labels = copy_train_labels[sortiert]
       # print(" test_label  :"+str(len(test_labels)) + " SOrtiert :" + str(len(sortiert))) ist gleich 
        list_test_labels = test_labels.tolist()

        list_copy_train_labels = copy_train_labels.tolist()
        #print ("list_copy_train_labels")
        #print (list_copy_train_labels)
        #print("fertig")
        Bag = BagOfWords(list_copy_train_labels)
        listreturn = []
        for i in list_test_labels:
            listreturn.append(Bag.most_freq_words(i))
            
        
        return np.asarray(listreturn)