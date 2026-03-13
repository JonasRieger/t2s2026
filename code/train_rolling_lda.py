import pandas as pd
from ttta.methods.rolling_lda import RollingLDA
from matplotlib import pyplot as plt
from typing import Optional
import datetime

df = pd.read_feather("temp/database_preprocessed.feather")
df["preprocessed_text"] = df["preprocessed_text"].apply(lambda x: x.tolist())

roll_lda = RollingLDA(K=12, alpha=0.05, how="YE", warmup=10, memory=3, prototype=100,
                      initial_epochs=500, subsequent_epochs=500, min_docs_per_chunk=1)
roll_lda.fit(df, text_column="preprocessed_text", date_column="date")
roll_lda.save("data/results/roll_lda.pickle")