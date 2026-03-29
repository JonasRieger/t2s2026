import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm

from matplotlib import pyplot as plt
from typing import Iterable
from spacy.language import Language
from spacy.tokens import Token
import json
import warnings
import logging
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import re
from typing import Callable

logger = logging.getLogger(__name__)

## %% Setup variables and other configurations
config = json.load(open("config.json"))

SPANISCH_STOPWORD = set(stopwords.words("spanish"))
pipeline_name = config.get("spacy_pipeline", None)
if pipeline_name is None:
    raise ValueError("Spacy pipeline name not found in config.json")
words_to_exclude = config.get("words_to_exclude", [])
if len(words_to_exclude) == 0:
    warnings.warn("No additional words to exclude found in config.json")


def custom_token_matching() -> Callable:
    """Custom token matching for the tokenizer which should never be tokenized"""
    special_cases = ["albemar", "albemar él"]
    return re.compile("|".join(special_cases)).match


def get_default_spacy_pipeline(pipeline_name: str) -> Language:
    """Load the spacy pipeline

    Args:
        pipeline_name (str): Name of the spacy pipeline to load

    Returns: Spacy Language Pipeline
    """

    pipeline = spacy.load(pipeline_name)

    ## modify the lemmatizer rule to handle some edge cases
    # albemar él -> albar
    # albemar -> albemarle
    # lemmatizer.add_special_case("albemar él", [{"LEMMA": "albemarle"}])
    # TODO: Leads to error
    # lemmatizer.add_special_case("albemar", [{"LEMMA": "albemarle"}])
    # lemmatizer.add_special_case("albemar él", [{"LEMMA": "albemarle"}])
    # pipeline.tokenizer.token_match = custom_token_matching()

    return pipeline


PIPELINE = get_default_spacy_pipeline(pipeline_name)


def is_valid_token(token: Token) -> bool:
    """Boolean function which defines if the token should be returned.
    Only tokens which are not part of the stopword list, not a digit and not a punctuation are returned.
    """
    ## all conditions must be true in order to return true and keep the token
    bools = [
        token.is_digit == False,
        token.is_punct == False,
        token.is_stop == False,
        token.is_alpha,
        token.text.strip() != "",
        token.text != "leer",
    ]
    if all(bools) == True:
        return True
    else:
        return False


def postprocess_token(token: Token) -> Token:
    """Postprocess a single token after spacy processing

    Args:
        token (Token): Token to postprocess

    Returns:
        Token: Postprocessed token
    """
    text = token.lemma_.lower()
    if text == "albemar":
        token.lemma_ = "albemarle"
    elif text == "albemar él":
        token.lemma_ = "albemarle"
    return token


def preprocess_batch(
    text: Iterable[str],
    pipeline: Language = PIPELINE,
    stopwords: set = SPANISCH_STOPWORD,
    additional_words_to_exclude: list = words_to_exclude,
    num_workers: int = max(1, cpu_count() - 2),
) -> Iterable[str]:
    """Preprocessing a list of text data

    Args:
        text (Iterable[str]): Iterable text to preprocess
        pipeline (Language, optional): Space Pipeline. "named entity recognition" is disabled. Defaults to PIPELINE.
        stopwords (set, optional): Additional stopword list to exclude from. Defaults to SPANISCH_STOPWORD.
        additional_words_to_exclude (list, optional): Additional words which should be omitted. Defaults to words_to_exclude.

    Returns:
        Iterable[str]: _description_
    """

    logger.info("Starting preprocessing of texts...")
    logger.info(f"Using pipeline: {pipeline.meta['name']}")

    logger.info(f"Beginning preprocessing of texts")

    # doc = pipeline(t, disable=["ner"])
    docs = list(pipeline.pipe(text, n_process=num_workers - 2, batch_size=500))

    preprocessed_texts = []
    for doc in tqdm(docs):
        if doc is None:
            preprocessed_texts.append([])
            continue
        else:
            t = doc.text
            if t is None or t == "":
                preprocessed_texts.append([])
            else:
                ## correct key errors
                postprocessed = [postprocess_token(token) for token in doc]
                ## adding lower case to the tokens
                pipeline_result = [
                    token.lemma_.lower()
                    for token in postprocessed
                    if is_valid_token(token)
                ]
                ## additional stopword removal
                pipeline_result = [
                    token for token in pipeline_result if token not in stopwords
                ]
                ## additional word removal from config.
                excluded_words = [
                    word
                    for word in pipeline_result
                    if word not in additional_words_to_exclude
                ]
                preprocessed_texts.append(excluded_words)

    return preprocessed_texts
