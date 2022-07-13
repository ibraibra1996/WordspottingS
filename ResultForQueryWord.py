from bagOfFeatures.bofFunctions import ImagePrep


def main():
    histograms, namesOfWords = ImagePrep.bof(documentSegmentation='3000300.gtp',
                                             documentImage='3000300.png',
                                             step_size=5,
                                             cell_size=3,
                                             step_size_doc=15,
                                             cell_size_doc=5,
                                             n_centroids=250,
                                             iter=20,
                                             pyramide=False)

    # hier kann eine Anfrage gemacht werden
    EntriesWith01, Ruckgabeliste = ImagePrep.resultListForPrecision(histograms=histograms,
                                                                    namesOfWords=namesOfWords,
                                                                    queryIndex=33)
    print(EntriesWith01)
    print(Ruckgabeliste)


if __name__ == "__main__":
    main()
