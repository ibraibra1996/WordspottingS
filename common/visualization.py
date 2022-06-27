import numpy as np
import matplotlib.pyplot as plt


def hbar_plot(x_values, y_labels, x_err=None, title=None):
    """ Plottet ein vertikales Balkendiagramm
    Params:
        x_values: Liste von numerischen x Werten
        y_labels: Liste von labels. Ein label pro Balken.
        x_err: Abweichungen fuer Fehlerbalken
        title: Ueberschrift fuer den Plot
    """

    y_pos = np.arange(len(y_labels))

    # plt.figure() erzeugt ein Figure-Objekt. Eine Figure ist ein Container, der
    # die Plot-Elemente enthaelt.
    #
    # fig.add_subplot(r, c, i) gibt ein Axes-Objekt aus der Figure fig. r und c
    # definieren ein Grid von Axes mit r Reihen und c Spalten. Das Axes-Objekt
    # mit Index i soll zurueckgegeben werden. Der Index beginnt bei 1 und erhoeht
    # sich zuerst entlang der Reihen.
    # Wenn r, c und i einstellig sind, kann auch ein dreistelliger Parameter genutzte werden.
    # Bsp: add_subplot(3, 2, 1) entspricht add_subplot(321).
    #
    # Mit plt.show() werden alle bisher erzeugten Figure-Objekte angezeigt.
    #
    # Siehe auch:
    # http://matplotlib.org/api/figure_api.html
    # http://matplotlib.org/api/axes_api.html

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(y_pos, x_values, align='center', alpha=0.4, xerr=x_err)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)

    if title is not None:
        ax.set_title(title)
    plt.show()


def bar_plot(x_values, y_values, y_err=None, title=None, rotation=None, horizontalalignment='left'):
    """ Plottet ein vertikales Balkendiagramm
    Params:
        x_values: Liste von x Werten. Auf None setzen, um den Index aus y_values
            zu verwenden. (Automatische Anzahl / Platzierung der x-ticks).
        y_values: Liste von y Werten
        y_err: Abweichungen fuer Fehlerbalken
        title: Ueberschrift des Plots
    """
    x_pos = np.arange(len(y_values))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x_pos, y_values, width=0.9, align='center', alpha=0.4, yerr=y_err)
    # if x_values is not None:
    # ax.set_xticks(np.linspace(0, len(y_values), len(x_values)), x_values)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_values, rotation=rotation, horizontalalignment=horizontalalignment)
    plt.show()

def bar_plot_double(x_values1,x_values2, y_values, y_err=None, title=None,
                    rotation=None, horizontalalignment='left'):
    x_pos = np.arange(len(y_values))  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x_pos, x_values1, width, color='r')

    wordsNuProCatgInTsd = np.array(x_values2)/1000


    rects2 = ax.bar(x_pos + width, wordsNuProCatgInTsd, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Anzahl Wörter und Dokumente')
    ax.set_title('Anzahl Wörter und Dokumente pro Kategorie')
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(y_values , rotation = 45, horizontalalignment='right')

    ax.legend((rects1[0], rects2[0]), ( 'Dokumente','Wörter * 1000'))

    plt.show()
