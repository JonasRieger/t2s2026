## This script preprocess the text data present in the excel file

# %%
import pandas as pd
from pathlib import Path
import nltk
import logging
import json
import sys
import spacy
from nltk.corpus import stopwords
from cet.preprocessing import preprocess_batch


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Load data from config file
    logger.info("Loading configuration...")

    config = json.load(open("config.json", "r", encoding="utf-8"))
    words_to_exclude = set(config["words_to_exclude"])
    logger.info(f"Creating Temp directory if it does not exist...")
    Path("temp").mkdir(parents=True, exist_ok=True)

    TEMP_FILE = Path("temp/database_preprocessed.feather")
    if TEMP_FILE.exists() == False:
        logger.info(f"Temp file {TEMP_FILE} not found. Starting preprocessing...")
        logger.info("Loading raw data...")
        df = pd.read_excel(
            "data/english_database.xlsx", sheet_name="Filtered_Conflicts"
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notnull()]
        logger.info("Filtering data based on start date from config...")
        start_date_data = config.get("start_date_data", "2011-01-01")
        logger.info(f"Filtering data from date: {start_date_data}")
        df = df[df["date"] >= pd.Timestamp(start_date_data)]

        nltk.download("stopwords")
        logger.info("Loading Spacy language model...")
        language_model = config.get("spacy_pipeline", "es_core_news_sm")
        logger.info(f"Using language model: {language_model}")
        nlp = spacy.load(language_model)

        logger.info("Loading stopwords...")

        stopwords_es = set(stopwords.words("spanish"))

        logger.info("Starting text preprocessing...")
        preprocessed_text = preprocess_batch(
            df["full_text"].tolist(),
            pipeline=nlp,
            stopwords=stopwords_es,
            additional_words_to_exclude=list(words_to_exclude),
        )
        logger.info("Text preprocessing completed.")

        ## Saving data to temp directory
        df["preprocessed_text"] = preprocessed_text
        logger.info(f"Saving preprocessed data to {TEMP_FILE}...")
        df.to_feather(TEMP_FILE)

    else:
        logger.info(f"Temp file {TEMP_FILE} found. Skipping preprocessing.")
        logger.info("No Preprocessing needed.")
