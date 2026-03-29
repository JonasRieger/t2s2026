from cet.utils import get_topic_cleartext
from typing import Optional
import pandas as pd
from ttta.methods.lda_prototype import LDAPrototype
import pandas


def add_topic_descriptions_to_excel(
    lda_model: LDAPrototype,
    excel_writer: pd.ExcelWriter,
    sheet_name: Optional[str] = None,
    close_on_finish: bool = False,
    num_top_words: int = 15,
    **kwargs,
) -> pandas.DataFrame:
    """Creates an excel report with topic explanations for each topic in the LDA model.
    Args:
        lda_model (LDAPrototype): Trained LDA Prototype model from the ttta library
        excel_writer (pd.ExcelWriter): Pandas ExcelWriter object to write the report to
        sheet_name (Optional[str], optional): Name of the excel sheet. Defaults to None.
        close_on_finish (bool, optional): Whether to close the excel writer after writing. Defaults to False.
        num_top_words (int, optional): Number of top words to consider for topic explanation. Defaults to 15.
        **kwargs: Additional keyword arguments for pandas.DataFrame.to_excel(), i.e. startrow, startcol, etc.
    Returns:
        None
    """
    top_words = lda_model.top_words(number=num_top_words).transpose()  # type: ignore
    top20_words = lda_model.top_words(number=num_top_words).transpose()  # type: ignore
    top_words["clear_description"] = top20_words.apply(
        lambda row: get_topic_cleartext(
            row.values.astype(str).tolist(), model="gpt-5.1"
        ),
        axis=1,
    )
    top_words.to_excel(
        excel_writer,
        sheet_name=sheet_name or f"Static LDA K-{lda_model._K}",
        index=True,
        **kwargs,
    )
    if close_on_finish:
        excel_writer.close()

    return top_words
