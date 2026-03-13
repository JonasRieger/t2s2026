### This script trains a single LDA and fine tunes it.
from tqdm import tqdm
from ttta.methods.lda_prototype import LDAPrototype
import pandas as pd
import logging
import sys
from pathlib import Path
import json
import datetime
from datetime import datetime
from cet.excel import add_topic_descriptions_to_excel
from cet.reports import create_topic_wordcloud_reports


config = json.load(open("config.json"))

K = 12
NUM_LDAs = 1
workers = config.get("num_workers", 4)
MODEL_SAVE_PATH = Path(config.get("model_save_path", "models"))
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


## LDA Default is 1/K
## Going to low is not good because it gives away the availibilty of soft clustering and leads to more hard clustering
alpha_values = [0.01, 0.05, 0.25, 0.5, 1]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Loading preprocessed data...")
    data_path = Path("temp/database_preprocessed.feather")
    if data_path.exists() is False:
        logger.error(f"Data file not found at {data_path}")
        sys.exit(1)

    df = pd.read_feather(data_path)

    ## train the models
    lda_models = {}
    logger.info("Training and tuning LDA model...")
    ## base params
    lda = LDAPrototype(K=K, prototype=NUM_LDAs)
    lda.fit(df["preprocessed_text"].tolist(), workers=workers)
    extended_path = MODEL_SAVE_PATH / f"{RUN_TIMESTAMP}_single_model_tuning"
    extended_path.mkdir(parents=True, exist_ok=True)
    path_to_str = extended_path / f"lda_finetuned_K{K}_standard_params.pkl"
    lda_models["standard"] = lda
    lda.save(str(path_to_str))

    for alpha in alpha_values:
        logger.info(f"Tuning LDA with alpha={alpha}...")
        lda = LDAPrototype(K=K, prototype=NUM_LDAs, alpha=alpha)
        lda.fit(df["preprocessed_text"].tolist(), workers=workers)
        path_to_str = extended_path / f"lda_finetuned_K{K}_alpha{alpha:.3f}.pkl"
        lda.save(str(path_to_str))
        lda_models[f"alpha_{alpha:.3f}"] = lda

    logger.info("Writing LDA Topic summaries to Excel...")
    ## save lda models to
    excel_writer = pd.ExcelWriter(
        extended_path / "lda_model_summary.xlsx", engine="xlsxwriter"
    )
    for name, model in tqdm(
        lda_models.items(), desc="Writing LDA model Topic summaries to Excel"
    ):
        logger.info(f"Writing summary for model: {name}...")
        df = add_topic_descriptions_to_excel(model, excel_writer, sheet_name=name)
        logger.info(f"Finished writing summary for model: {name}...")
        logger.info("Creating Topic Specific Wordclouds...")
        descriptions = df["clear_description"].tolist()
        fig, ax = create_topic_wordcloud_reports(
            model,
            n_cols=2,
            topic_names=descriptions,
            title=f"LDA Topic WordClouds K={K} - Model: {name}",
        )
        fig.savefig(
            extended_path / f"lda_wordclouds_{name}.png",
            dpi=300,
            bbox_inches="tight",
        )

    excel_writer.close()

    # logger.info("Generating Topic Specific Wordclouds for all models...")
    # ## Create word cloud reports
    # for name, model in tqdm(lda_models.items(), desc="Generating wordcloud reports"):
    #     fig, ax = create_topic_wordcloud_reports(model, n_cols=2, num_topwords=20)
    #     fig.savefig(
    #         extended_path / f"lda_wordclouds_{name}.png",
    #         dpi=300,
    #         bbox_inches="tight",
    #     )
