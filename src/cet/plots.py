from ttta.methods.lda_prototype import LDAPrototype
from ttta.methods.LDA.importance_calculation import calculate_importance
from wordcloud import WordCloud
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def plot_wordcloud(ax: Axes, words: list[list[str]]):
    """Generates a global word cloud and plots it on the given axis object from a list of lists of words
    Args:
        ax (Axes): Matplotlib axis object where the word cloud will be plotted
        words (list[list[str]]): List of lists of words from which the word cloud will be generated
    """
    ## global word cloud
    word_list = []
    for l in words:
        word_list.extend(l)
    counter = Counter(word_list)
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(counter)

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")


def plot_ldaprototype_wordcloud(model: LDAPrototype, topic: int, number: int = 100):
    """Generates a LDA Prototype word cloud

    Args:
        model (LDAPrototype): LDA Prototype model from the ttta library
        topic (int): Topic number for which the word cloud is generated
        number (int, optional): Number of top words to include in the word cloud. Defaults to 100.

    Returns:
        WordCloud: Generated word cloud object
    """
    word_topic_matrix = model.get_word_topic_matrix()
    importance = calculate_importance(word_topic_matrix)
    word_weights = {
        model._vocab[index]: importance[topic, index]
        for index in np.argsort(-importance[topic, :])[:number]
    }

    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(word_weights)
    return wordcloud
