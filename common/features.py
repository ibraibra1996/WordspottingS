import string
import numpy as np
from collections import defaultdict
from common.corpus import CorpusLoader
from nltk.stem.porter import PorterStemmer  # IGNORE:import-error
import cv2

from common.evaluation import CrossValidation


class AbsoluteTermFrequencies(object):
    """Klasse, die zur Durchfuehrung absoluter Gewichtung von Bag-of-Words
    Matrizen (Arrays) verwendet werden kann. Da Bag-of-Words zunaechst immer in
    absoluten Frequenzen gegeben sein muessen, ist die Implementierung dieser 
    Klasse trivial. Sie wird fuer die softwaretechnisch eleganten Unterstuetzung
    verschiedner Gewichtungsschemata benoetigt (-> Duck-Typing).   
    """

    @staticmethod
    def weighting(bow_mat):
        """Fuehrt die Gewichtung einer Bag-of-Words Matrix durch.
        
        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).
                
        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen 
                je Dokument (zeilenweise).
        """
        # Gibt das NumPy Array unveraendert zurueck, da die Bag-of-Words Frequenzen
        # bereits absolut sind.
        return bow_mat

    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'absolute'

    def __format__(self, format_spec):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentation des Objekts verwendet.
        Sie wird durch den Python Interpreter ausgefuehrt, wenn das Objekt einer
        'format' methode uebergeben wird."""
        return format(str(self), format_spec)


class RelativeTermFrequencies(object):
    """Realisiert eine Transformation von in absoluten Frequenzen gegebenen 
    Bag-of-Words Matrizen (Arrays) in relative Frequenzen.
    """

    @staticmethod
    def weighting(bow_mat):
        """Fuehrt die relative Gewichtung einer Bag-of-Words Matrix (relativ im 
        Bezug auf Dokumente) durch.
        
        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).
                
        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen 
                je Dokument (zeilenweise).
        """
        # divide each element in row by its sum
        return bow_mat / np.sum(bow_mat, axis=1)[:, None]

    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'relative'

    def __format__(self, format_spec):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentation des Objekts verwendet.
        Sie wird durch den Python Interpreter ausgefuehrt, wenn das Objekt einer
        'format' methode uebergeben wird."""
        return format(str(self), format_spec)


class RelativeInverseDocumentWordFrequecies(object):
    """Realisiert eine Transformation von in absoluten Frequenzen gegebenen 
    Bag-of-Words Matrizen (Arrays) in relative - inverse Dokument Frequenzen.
    """

    def __init__(self, vocabulary, category_wordlists_dict):
        """Initialisiert die Gewichtungsberechnung, indem die inversen Dokument
        Frequenzen aus dem Dokument Korpous bestimmt werden.
        
        Params:
            vocabulary: Python Liste von Woertern (das Vokabular fuer die 
                Bag-of-Words).
            category_wordlists_dict: Python dictionary, das zu jeder Klasse (category)
                eine Liste von Listen mit Woertern je Dokument enthaelt.
                Siehe Beschreibung des Parameters cat_word_dict in der Methode
                BagOfWords.category_bow_dict.
        """

        bow = BagOfWords(vocabulary)
        category_bow_dict = bow.category_bow_dict(category_wordlists_dict)

        # mit fold_index=0 und n_folds=1 werden alle zeilen in test_bow gespeichert
        # in test_bow sind Anzahl Dokumente die Term enthalten
        CV = CrossValidation(category_bow_dict, n_folds=1)
        train_bow, train_labels, test_bow, test_labels = CV.corpus_fold(fold_index=0)

        # count nonzeros in each column
        # dokumente die term enthalten, das sind diejenige bei denen die spalte ungleich 0 ist
        n_non_zeros = np.count_nonzero(test_bow != 0, axis=0)

        # IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
        self.__inverse_document_Frequencies = np.log(len(test_bow) / n_non_zeros)

    def weighting(self, bow_mat):
        """Fuehrt die Gewichtung einer Bag-of-Words Matrix durch.
        
        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).
                
        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen 
                je Dokument (zeilenweise).
        """
        # TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).

        term_Frequencies = RelativeTermFrequencies.weighting(bow_mat=bow_mat)
        return term_Frequencies * self.__inverse_document_Frequencies

    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'tf-idf'

    def __format__(self, format_spec):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentation des Objekts verwendet.
        Sie wird durch den Python Interpreter ausgefuehrt, wenn das Objekt einer
        'format' methode uebergeben wird."""
        return format(str(self), format_spec)


class BagOfWords(object):
    """Berechnung von Bag-of-Words Repraesentationen aus Wortlisten bei 
    gegebenem Vokabular.
    """

    def __init__(self, vocabulary, term_weighting=AbsoluteTermFrequencies()):
        """Initialisiert die Bag-of-Words Berechnung
        
        Params:
            vocabulary: Python Liste von Woertern / Termen (das Bag-of-Words Vokabular).
                Die Reihenfolge der Woerter / Terme im Vokabular gibt die Reihenfolge
                der Terme in den Bag-of-Words Repraesentationen vor.
            term_weighting: Objekt, das die weighting(bow_mat) Methode implemeniert.
                Optional, verwendet absolute Gewichtung als Default.
        """
        self.__vocabulary = vocabulary
        self.__term_weighting = term_weighting
        self.__vocabulary_index_lut = {vword: index for index, vword in enumerate(vocabulary)}

    def category_bow_dict(self, cat_word_dict):
        """Erzeugt ein dictionary, welches fuer jede Klasse (category)
        ein NumPy Array mit Bag-of-Words(Häufigkeiten) Repraesentationen enthaelt.
        
        Params:
            cat_word_dict: Dictionary, welches fuer jede Klasse (category)
                eine Liste (Dokumente) von Listen (Woerter) enthaelt.
                cat : [ [word1, word2, ...],  <--  doc1
                        [word1, word2, ...],  <--  doc2
                        ...                         ...
                        ]
        Returns:
            category_bow_mat: Ein dictionary mit Bag-of-Words Matrizen fuer jede
                Kategory. Eine Matrix enthaelt in jeder Zeile die Bag-of-Words 
                Repraesentation eines Dokuments der Kategorie. (d x t) bei d
                Dokumenten und einer Vokabulargroesse t (Anzahl Terme). Die
                Reihenfolge der Terme ist durch die Reihenfolge der Worter / Terme
                im Vokabular (siehe __init__) vorgegeben.
        """

        category_bow_mat = {}

        for catg, word_Lists in cat_word_dict.items():
            mat = np.zeros((len(word_Lists), len(self.__vocabulary)))
            for index, doc in enumerate(word_Lists):
                for word in doc:
                    try:
                        mat[index, self.__vocabulary_index_lut[word]] += 1
                    except KeyError:
                        pass
            category_bow_mat[catg] = self.__term_weighting.weighting(mat)

        return category_bow_mat

    @staticmethod
    def most_freq_words(word_list, n_words=None):
        """Bestimmt die (n-)haeufigsten Woerter in einer Liste von Woertern.
        
        Params:
            word_list: Liste von Woertern
            n_words: (Optional) Anzahl von haeufigsten Woertern (top n). Falls
                n_words mit None belegt ist, sollen alle vorkommenden Woerter
                betrachtet werden.
            
        Returns:
            words_topn: Python Liste, die (top-n) am haeufigsten vorkommenden 
                Woerter enthaelt. Die Sortierung der Liste ist nach Haeufigkeit
                absteigend.
        """

        from collections import Counter
        # (x[0],x[1]) : (wort, häufigkeit)
        sortedListOfFre = sorted(Counter(word_list).items(), key=lambda x: x[1], reverse=True)

        return [x[0] for x in sortedListOfFre[:n_words]]


class WordListNormalizer(object):

    def __init__(self, stoplist=None, stemmer=None):
        """Initialisiert die Filter
        
        Params: 
            stoplist: Python Liste von Woertern, die entfernt werden sollen
                (stopwords). Optional, verwendet NLTK stopwords falls None
            stemmer: Objekt, das die stem(word) Funktion implementiert. Optional,
                verwendet den Porter Stemmer falls None.
        """

        if stoplist is None:
            stoplist = CorpusLoader.stopwords_corpus()
        self.__stoplist = stoplist

        if stemmer is None:
            stemmer = PorterStemmer()
        self.__stemmer = stemmer
        self.__punctuation = string.punctuation
        self.__delimiters = ["''", '``', '--']

    def normalize_words(self, word_list):
        """Normalisiert die gegebenen Woerter nach in der Methode angwendeten
        Filter-Regeln (Gross-/Kleinschreibung, stopwords, Satzzeichen, 
        Bindestriche und Anführungszeichen, Stemming)
        
        Params: 
            word_list: Python Liste von Worten.
            
        Returns:
            word_list_filtered, word_list_stemmed: Tuple von Listen
                Bei der ersten Liste wurden alle Filterregeln, bis auch stemming
                angewandt. Bei der zweiten Liste wurde zusätzlich auch stemming
                angewandt.
        """
        lowerWord_List = [word.lower() for word in word_list]
        word_list_filtered = [word for word in lowerWord_List
                              if word not in self.__stoplist and
                              word not in list(self.__punctuation) and
                              word not in self.__delimiters]

        word_list_stemmed = [self.__stemmer.stem(word) for word in word_list_filtered]

        return word_list_filtered, word_list_stemmed

    def category_wordlists_dict(self, corpus):
        """Erstellt Python dictionary, das zu jeder Klasse (category)
            eine Liste von Listen mit Woertern, gefiltert und gestemmt, je Dokument enthaelt.

        Params:
                corpus:Sammlung sprachlicher Gegenstände(Wörter, Sätze, Texte), die verwendet werden sollen.
        Returns:
            cat_word_dict: dictionary, das zu jeder Klasse (category)
            eine Liste von Listen mit Woertern je Dokument enthaelt.
            { cat1 : [ [word1, word2, ...],  <--  doc1
                        [word1, word2, ...],  <--  doc2
                        ...                         ...
                        ] , cat2: [[word1, word2, ...],
                                 ...
                                 ,[word1, word2, ...]]
                        }


        """

        """
                cat_word_dict = {}
        for catg in corpus.categories():
            cat_word_dict[catg] = []
            for doc in corpus.fileids(catg):
                # normalize_words() für jedes Dokument aufrufen
                word_list_stemmed = self.normalize_words(word_list=corpus.words(doc))[1]
                cat_word_dict[catg].append(word_list_stemmed)
        return cat_word_dict
        
        """

        cat_word_dict = {}
        for catg in corpus.categories():
            # normalize_words() für jedes Dokument aufrufen
            cat_word_dict[catg] = np.array(
                [self.normalize_words(word_list=corpus.words(doc))[1] for doc in corpus.fileids(catg)])
        return cat_word_dict


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


class TopicFeatureTransform(object):
    """Realsiert statistische Schaetzung eines Topic Raums und Transformation
    in diesen Topic Raum.
    """

    def __init__(self, topic_dim):
        """Initialisiert die Berechnung des Topic Raums
        
        Params:
            topic_dim: Groesse des Topic Raums, d.h. Anzahl der Dimensionen.
        """
        self.__topic_dim = topic_dim
        # Transformation muss in der estimate Methode definiert werden.
        self.__T = None
        self.__S_inv = None

    def estimate(self, train_data, train_labels):  # IGNORE:unused-argument
        """Statistische Schaetzung des Topic Raums
        
        Params:
            train_data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
                Hinweis: Fuer den hier zu implementierenden Topic Raum werden die
                Klassenlabels nicht benoetigt. Sind sind Teil der Methodensignatur
                im Sinne einer konsitenten und vollstaendigen Verwaltung der zur
                Verfuegung stehenden Information.
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        raise NotImplementedError()

    def transform(self, data):
        """Transformiert Daten in den Topic Raum.
        
        Params:
            data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
        
        Returns:
            data_trans: ndarray der in den Topic Raum transformierten Daten 
                (d x topic_dim).
        """
        raise NotImplementedError()


def compute_sift_descriptors(im_arr, cell_size=5, step_size=20):
    """
    Berechnet SIFT Deskriptoren in einem regulären Gitter

    Params:
        im_array: ndarray, mit dem Eingabebild in Graustufen (n x n x 1)
        cell_size: int, Größe einer Zelle des SIFT Deskriptors in Pixeln
        step_size: int, Schrittweite im regulären Gitter in Pixeln

    Returns:
        frames: list, mit x,y Koordinaten der Deskriptor Mittelpunkte
        desc: ndarray, mit den berechneten SIFT Deskriptoren (N x 128)
         Deskriptor. desc enthaelt die 128 dimensionalen Vektoren.

    """
    # Generate dense grid
    frames = [(x, y) for x in np.arange(10, im_arr.shape[1], step_size, dtype=np.float32)
              for y in np.arange(10, im_arr.shape[0], step_size, dtype=np.float32)]

    # Note: In the standard SIFT detector algorithm, the size of the
    # descriptor cell size is related to the keypoint scal by the magnification factor.
    # Therefore the size of the is equal to cell_size/magnification_factor (Default: 3)
    kp = [cv2.KeyPoint(x, y, cell_size / 3) for x, y in frames]

    sift = cv2.SIFT_create()

    sift_features = sift.compute(im_arr, kp)
    desc = sift_features[1]
    return frames, desc
