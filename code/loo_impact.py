import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cosine
from ttta.methods.rolling_lda import RollingLDA
from operator import itemgetter


def word_impact(roll: RollingLDA, number: int = 5, previous_chunks: int = 1,
                date_format: str = "%Y-%m-%d", fast: int = -1) -> pd.DataFrame:
    """Calculate the leave-one-out word impact for each topic and time chunk.

    Compares each time chunk's word-topic distribution against the aggregated
    distribution from the previous `previous_chunks` time chunks to identify
    which words contributed most to changes in topic composition over time.

    Args:
        roll: A fitted RollingLDA model.
        number: The number of top impact words to return per topic per time chunk.
        previous_chunks: The number of previous time chunks to aggregate as the
            comparison baseline. For example, if previous_chunks=3, each chunk
            is compared against the sum of the 3 preceding chunks.
        date_format: The date format for the output DataFrame.
        fast: If True, skip leave-one-out calculation for words that have zero
            count in both the current and previous chunks. This speeds up
            computation without affecting results unless `number` is very large
            relative to vocabulary size.

    Returns:
        A pandas DataFrame with columns:
            - Topic: The topic index
            - Date: The end date of the time chunk being analyzed
            - Significant Words: Tuple of the top `number` words that most
              influenced the change in topic distribution
    """
    # Validate inputs
    if not isinstance(number, int):
        try:
            if number == int(number):
                number = int(number)
            else:
                raise ValueError
        except ValueError:
            raise TypeError("number must be an integer!")
    if number < 1:
        raise ValueError("number must be a natural number greater than 0")

    if not isinstance(previous_chunks, int):
        try:
            if previous_chunks == int(previous_chunks):
                previous_chunks = int(previous_chunks)
            else:
                raise ValueError
        except ValueError:
            raise TypeError("previous_chunks must be an integer!")
    if previous_chunks < 1:
        raise ValueError("previous_chunks must be a natural number greater than 0")

    if not isinstance(date_format, str):
        raise TypeError("date_format must be a string!")
    if not isinstance(fast, int):
        raise TypeError("fast must be an integer!")

    # Get word-topic matrices for all chunks
    # Shape: (n_chunks, n_words, n_topics)
    assignments = np.array(
        [roll.get_word_topic_matrix(chunk=chunk).transpose() for
         chunk, row in roll.chunk_indices.iterrows()])

    # Transpose to (n_topics, n_words, n_chunks) for easier topic-wise iteration
    topics = assignments.transpose((1, 2, 0))

    n_topics = topics.shape[0]
    n_chunks = topics.shape[2]

    if n_chunks <= previous_chunks:
        raise ValueError(
            f"Not enough time chunks ({n_chunks}) for previous_chunks={previous_chunks}. "
            f"Need at least {previous_chunks + 1} chunks."
        )

    # Get chunk end dates
    date_column = roll._date_column
    chunk_dates = roll.chunk_indices[date_column].tolist()

    # Get vocabulary
    vocab = roll.lda.get_vocab()

    leave_one_out_word_impact = {"Topic": [], "Date": [], "Significant Words": [], "Impacts": []}

    # Iterate over each time chunk starting from index `previous_chunks`
    for chunk_idx in range(1, n_chunks):
        # Get the range of previous chunks to compare against
        start_idx = chunk_idx - min(chunk_idx, previous_chunks)

        for topic_idx in range(n_topics):
            # Current chunk's word distribution for this topic
            current_dist = topics[topic_idx, :, chunk_idx]

            # Aggregate previous chunks' word distributions for this topic
            previous_dist = topics[topic_idx, :, start_idx:chunk_idx].sum(axis=1)

            # Skip if both distributions are all zeros
            if current_dist.sum() == 0 and previous_dist.sum() == 0:
                continue

            # Calculate baseline cosine distance
            baseline_distance = cosine(current_dist, previous_dist)

            # Calculate leave-one-out impact for each word
            n_words = topics.shape[1]
            loo_impact = np.zeros(n_words)

            for word_idx in range(n_words):
                # Fast mode: skip words with zero count in both distributions
                if current_dist[word_idx] < fast and previous_dist[word_idx] < fast:
                    loo_impact[word_idx] = 0
                else:
                    # Remove this word and recalculate distance
                    current_without_word = np.delete(current_dist, word_idx)
                    previous_without_word = np.delete(previous_dist, word_idx)

                    distance_without_word = cosine(current_without_word, previous_without_word)

                    # Impact = how much the distance decreases when word is removed
                    # Negative values mean removing the word decreases distance
                    # (i.e., the word was contributing to the difference)
                    loo_impact[word_idx] = distance_without_word - baseline_distance

            # Get indices of words with most negative impact (most influential)
            # These are words whose removal most reduces the distance
            top_word_indices = loo_impact.argsort()[:number].astype(np.uint64).tolist()

            # Handle case where number=1 (itemgetter returns single item, not tuple)
            if number == 1:
                significant_words = (vocab[top_word_indices[0]],)
            else:
                significant_words = itemgetter(*top_word_indices)(vocab)

            relevant_impacts = loo_impact[top_word_indices]

            leave_one_out_word_impact["Topic"].append(topic_idx)
            leave_one_out_word_impact["Date"].append(
                chunk_dates[chunk_idx].strftime(date_format))
            leave_one_out_word_impact["Significant Words"].append(significant_words)
            leave_one_out_word_impact["Impacts"].append(relevant_impacts)

    return pd.DataFrame(leave_one_out_word_impact)

if __name__ == '__main__':
    # Check if required files exist
    if not os.path.exists('roll_lda.pickle'):
        print("Error: roll_lda.pickle not found.")
        exit(1)

    roll = RollingLDA(5)
    roll.load("roll_lda.pickle")
    
    wi = word_impact(roll)

    wi.to_pickle("word_impacts.pickle")
