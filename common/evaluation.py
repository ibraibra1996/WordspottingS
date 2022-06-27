import numpy as np
# from common.features import IdentityFeatureTransform
from collections import defaultdict


# most likely due to a circular import
class IdentityFeatureTransform(object):
    """Realisert eine Transformation auf die Identitaet, bei der alle Daten
    auf sich selbst abgebildet werden. Die Klasse ist hilfreich fuer eine
    softwaretechnisch elegante Realisierung der Funktionalitaet "keine Transformation
    der Daten durchfuehren" (--> Duck-Typing).
    """

    def estimate(self, train_data, train_labels):
        pass

    def transform(self, data):
        return data


class CrossValidation(object):

    def __init__(self, category_bow_dict, n_folds):
        """Initialisiert die Kreuzvalidierung ueber gegebnen Daten
            Kreuzvalidierung ist eine Vorgehensweise zur Bewertung der Leistung
            eines Algorithmus beim Machine Learning.
              Mit neuen Datensätzen, die nicht in der Trainingsphase genutzt wurden,
               wird die Güte der Vorhersage geprüft.
        
        Params:
            category_bow_dict: Dictionary, das fuer jede Klasse ein ndarray mit Merkmalsvektoren
                (zeilenweise) enthaelt.
            n_folds: Anzahl von Ausschnitten ueber die die Kreuzvalidierung berechnet werden soll.
            # n_folds=5 mit wird jedes fünftes Element als testdaten genommen

        """
        self.__category_bow_list = list(
            category_bow_dict.items())
        self.__n_folds = n_folds

    def validate(self, classifier, feature_transform=None):
        """Berechnet eine Kreuzvalidierung ueber die Daten,
        
        Params:
            classifier: Objekt, das die Funktionen estimate und classify implementieren muss.
            feature_transform: Objekt, das die Funktionen estimate und transform implementieren 
                muss. Optional: Falls None, wird keine Transformation durchgefuehrt.

        Returns:
            Dokumentiern!
            crossval_overall_result: durchschnittliche Gesamtfehlerrate
            crossval_class_results: durchschnittliche klassenspezifische Fehlerraten
        """
        if feature_transform is None:
            feature_transform = IdentityFeatureTransform()

        crossval_overall_list = []
        crossval_class_dict = defaultdict(list)
        for fold_index in range(self.__n_folds):
            train_bow, train_labels, test_bow, test_labels = self.corpus_fold(fold_index)
            feature_transform.estimate(train_bow, train_labels)
            train_feat = feature_transform.transform(train_bow)
            test_feat = feature_transform.transform(test_bow)
            classifier.estimate(train_feat, train_labels)
            estimated_test_labels = classifier.classify(test_feat)
            classifier_eval = ClassificationEvaluator(estimated_test_labels, test_labels)
            crossval_overall_list.append(list(classifier_eval.error_rate()))
            # category_error_rates()gibt: list von tuple: [ (category, error_rate, n_wrong, n_samlpes), ...]
            crossval_class_list = classifier_eval.category_error_rates()
            for category, err, n_wrong, n_samples in crossval_class_list:
                crossval_class_dict[category].append([err, n_wrong, n_samples])

        crossval_overall_mat = np.array(crossval_overall_list)
        crossval_overall_result = CrossValidation.__crossval_results(crossval_overall_mat)

        crossval_class_results = []
        for category in sorted(crossval_class_dict.keys()):

            crossval_class_mat = np.array(crossval_class_dict[category])
            crossval_class_result = CrossValidation.__crossval_results(crossval_class_mat)
            crossval_class_results.append((category, crossval_class_result))

        # durchschnittliche Ergebnisse
        return crossval_overall_result, crossval_class_results

    @staticmethod
    def __crossval_results(crossval_mat):
        """
        Beachten Sie, dass dabei ein gewichtetes Mittel gebildet werden muss,
         da die einzelnen Test Folds nicht unbedingt gleich gross sein muessen.
        """
        #
        #
        # error_rate, n_wrong, n_samples
        # [0.25,1,4]
        # [0,666,2,3]
        # --->crossval_weights: 4/7 , 3/7
        # crossval_result:

        # Relative number of samples
        crossval_weights = crossval_mat[:, 2] / crossval_mat[:, 2].sum()
        # Weighted sum over recognition rates for all folds
        crossval_result = (crossval_mat[:, 0] * crossval_weights).sum()
        return crossval_result

    def corpus_fold(self, fold_index):
        """Berechnet eine Aufteilung der Daten in Training und Test
        
        Params:
            fold_index: Index des Ausschnitts der als Testdatensatz verwendet werden soll.
            #Also
        Returns:
            Ergaenzen Sie die Dokumentation!
            training_bow_mat: eine Matrix von Trainingsdaten, in der jeder Vektor ein Dokument präsentiert, die zum Trainieren eines Klassifikators vwewenden werden sollen
            training_label_mat: eine Matrix von Trainingslabels, in der jeder Vektor die Klasse eines Dokumentes entspricht.
            test_bow_mat: eine Matrix von Testdaten, in der jeder Vektor ein Dokument präsentiert, die zum Test eines Klassifikators vwewenden werden sollen
            test_label_mat: Testlabels, die anhand Trainingsdaten bestimmt werden.
        """
        training_bow_mat = []
        training_label_mat = []
        test_bow_mat = []
        test_label_mat = []

        for category, bow_mat in self.__category_bow_list:
            # anzahl Dokumenten in dieser category
            n_category_samples = bow_mat.shape[0]
            #
            # Erklaeren Sie nach welchem Schema die Aufteilung der Daten erfolgt.
            # Die Daten von fold_index bis ende einer Kategorie (n_category_samples) um n_folds schritten werden als Testdaten gewählt
            # und das Rest als Trainingsdaten. Also wenn fold_index=0 und n_fold=2 und n_category_samples dann werden 0,2,4,..,8 als
            # test_indices gewählt und das Rest als Trainingsdaten
            #
            # Select indices for fold_index-th test fold, remaining indices are used for training
            test_indices = list(range(fold_index, n_category_samples, self.__n_folds))
            train_indices = [train_index for train_index in range(n_category_samples)
                             if train_index not in test_indices]
            category_train_bow = bow_mat[train_indices, :]
            category_test_bow = bow_mat[test_indices, :]
            # Construct label matrices ([x]*3 --> [x, x, x])
            category_train_labels = np.array([[category] * len(train_indices)])
            category_test_labels = np.array([[category] * len(test_indices)])

            training_bow_mat.append(category_train_bow)
            training_label_mat.append(category_train_labels.T)
            test_bow_mat.append(category_test_bow)
            test_label_mat.append(category_test_labels.T)

        # vstack:Stack arrays in sequence vertically (row wise).
        training_bow_mat = np.vstack(tuple(training_bow_mat))
        training_label_mat = np.vstack(tuple(training_label_mat))
        test_bow_mat = np.vstack(tuple(test_bow_mat))
        test_label_mat = np.vstack(tuple(test_label_mat))

        return training_bow_mat, training_label_mat, test_bow_mat, test_label_mat


class ClassificationEvaluator(object):

    def __init__(self, estimated_labels, groundtruth_labels):
        """Initialisiert den Evaluator fuer ein Klassifikationsergebnis 
        auf Testdaten.
        
        Params:
            estimated_labels: ndarray (N x 1) mit durch den Klassifikator 
                bestimmten Labels.
            groundtruth_labels: ndarray (N x 1) mit den tatsaechlichen Labels.
                
        """
        self.__estimated_labels = estimated_labels
        self.__groundtruth_labels = groundtruth_labels
        # 
        # Bestimmen Sie hier die Uebereinstimmungen und Abweichungen der
        # durch den Klassifikator bestimmten Labels und der tatsaechlichen 
        # Labels

        self.__matchingLabels = self.__estimated_labels == self.__groundtruth_labels

    def error_rate(self, mask=None):
        """Bestimmt die Fehlerrate auf den Testdaten.
        
        Params:
            mask: Optionale boolsche Maske, mit der eine Untermenge der Testdaten
                ausgewertet werden kann. Nuetzlich fuer klassenspezifische Fehlerraten.
                Bei mask=None werden alle Testdaten ausgewertet.
        Returns:
            tuple: (error_rate, n_wrong, n_samlpes)
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """
        matchingLabels = self.__matchingLabels
        if mask is not None:
            # nur mask indiecies aus matchingLabels nehemen
            matchingLabels = matchingLabels[mask.reshape(-1)]

        n_samples = len(matchingLabels)
        # count_nonzero(matchingLabels) bestimmt anzahl der Trues
        n_wrong = n_samples - np.count_nonzero(matchingLabels)
        error_rate = (n_wrong / n_samples) * 100

        return error_rate, n_wrong, n_samples

    def category_error_rates(self):
        """Berechnet klassenspezifische Fehlerraten
        
        Returns:
            list von tuple: [ (category, error_rate, n_wrong, n_samlpes), ...]
            category: Label der Kategorie / Klasse
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """

        ListOfTuple = []
        categories = np.unique(self.__groundtruth_labels)

        for category in categories:
            # Array der länge groundtruth_labels von Falses
            # wie np.zeros(shape=(3, 2), dtype=bool) gibt:
            # [[False False]
            # [False False]
            # [False False]]
            mask_array = np.zeros(shape=(len(self.__groundtruth_labels), 1), dtype=bool)

            # indecies der tatsaechlichen Labels.
            indecies = np.where(self.__groundtruth_labels == [category])[0]
            # mask für dieser Kategorie vorbereiten
            mask_array[indecies] = True
            # error_rate für dieser Kategorie aufrufen
            error_rate, n_wrong, n_samples = self.error_rate(mask=mask_array)
            ListOfTuple.append((category, error_rate, n_wrong, n_samples))

        return ListOfTuple

# %%
