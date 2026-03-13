import pandas as pd
from cet.analysis import lda
from cet.utils import get_topic_cleartext
from ttta.methods.lda_prototype import LDAPrototype
from matplotlib import pyplot as plt
from typing import Optional
from cet.plots import plot_ldaprototype_wordcloud, plot_wordcloud
from cet.excel import add_topic_descriptions_to_excel
from wordcloud import WordCloud
import datetime


def create_lda_report(df: pd.DataFrame, K: list[int]):
    """
    Create LDA report with word clouds and topic explanations inside an excel file.

    Parameters:
    df (pd.DataFrame): DataFrame containing preprocessed text data.
    K (List[int]): List of topic numbers to analyze.

    Returns:
    None
    """
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"reports/{current_timestamp}", exist_ok=True)
    excel_writer = pd.ExcelWriter(
        f"reports/{current_timestamp}/lda_report.xlsx", engine="xlsxwriter"
    )

    ## global word cloud
    word_list = []
    for l in df["preprocessed_text"].to_list():
        word_list.extend(l)
    counter = Counter(word_list)
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(counter)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Global WordCloud", fontsize=16)
    plt.show()
    fig.savefig(
        f"reports/{current_timestamp}/wordcloud_filtered_data.png",
        dpi=300,
        bbox_inches="tight",
    )

    ## LDA Analysis with varying K
    for k in K:
        n_cols = 2
        n_rows = math.ceil(k / n_cols)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        lda_model = LDAPrototype(K=k, prototype=1)
        temp_data = df
        temp_data["text"] = temp_data["preprocessed_text"]
        lda_model.fit(temp_data["text"].to_list())
        topic_explanations = []
        top_words = lda_model.top_words(number=15).transpose()
        top20_words = lda_model.top_words(number=15).transpose()
        top_words["clear_description"] = top20_words.apply(
            lambda row: get_topic_cleartext(
                row.values.astype(str).tolist(), model="gpt-5.1"
            ),
            axis=1,
        )
        top_words.to_excel(excel_writer, sheet_name=f"Static LDA K-{k}", index=True)
        for topic in range(k):

            row = topic // n_cols
            col = topic % n_cols
            wordcloud = plot_wordcloud(lda_model, topic=topic)
            ax[row, col].imshow(wordcloud, interpolation="antialiased")
            ax[row, col].axis("off")
            ax[row, col].set_title(
                f'Topic {topic}: {top_words.iloc[topic]["clear_description"]}',
                fontsize=8,
                wrap=True,
            )

        for a in fig.axes:
            if not bool(a.has_data()):
                fig.delaxes(a)

        plt.suptitle(f"LDA Topic WordClouds K={k}", fontsize=16)
        plt.tight_layout(h_pad=1.10, w_pad=1.10)
        plt.savefig(f"reports/{current_timestamp}/lda_wordclouds_k_{k}.png", dpi=150)
        plt.close()

    excel_writer.close()
