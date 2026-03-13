"""
Analyze topic prevalence over time by location.

This script loads the RollingLDA model and computes average topic prevalence
for each location at each time period, saving the results to a JSON file
that can be used by Postprocessing.py for visualization.

Supports topic clustering: topics can be grouped into clusters, and cluster
prevalence is computed by summing the topic prevalences within each cluster.
"""

import json
import pandas as pd
import numpy as np
import os


def analyze_topic_prevalence(model_path='roll_lda.pickle',
                              data_path='english_database.xlsx',
                              sheet_name='Filtered_Conflicts',
                              output_path='topic_prevalence_by_location.json',
                              cluster_mapping=None):
    """
    Analyze topic prevalence over time for each location.

    Parameters:
    -----------
    model_path : str
        Path to the RollingLDA model pickle file
    data_path : str
        Path to the Excel file with document data
    sheet_name : str
        Sheet name in the Excel file
    output_path : str
        Path to save the output JSON file
    cluster_mapping : dict, optional
        Mapping from cluster names to lists of topic indices.
        Example: {"Environment": [0, 2], "Social": [1, 3, 4]}
        If None, no cluster analysis is performed.

    Returns:
    --------
    dict : The computed prevalence data
    """
    from ttta.methods.rolling_lda import RollingLDA

    print("Loading RollingLDA model...")
    roll = RollingLDA(5)
    roll.load(model_path)

    print("Loading document data...")
    docs = pd.read_excel(data_path, sheet_name=sheet_name)
    docs = docs[docs["date"].isna() == False]

    # Apply the same sorting that RollingLDA used internally
    docs = docs.loc[roll.sorting].reset_index(drop=True)

    # Get the full document-topic matrix
    print("Getting document-topic matrix...")
    full_doc_topic_matrix = roll.get_document_topic_matrix()
    n_topics = full_doc_topic_matrix.shape[1]
    print(f"  Shape: {full_doc_topic_matrix.shape}")
    print(f"  Number of topics: {n_topics}")

    # Normalize to get probabilities (each row sums to 1)
    # The raw matrix contains word counts per topic, not probabilities
    row_sums = full_doc_topic_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    full_doc_topic_matrix = full_doc_topic_matrix / row_sums
    print(f"  Normalized: row sums now equal 1.0")

    # Process cluster mapping
    if cluster_mapping:
        print(f"\nCluster mapping provided:")
        for cluster_name, topic_indices in cluster_mapping.items():
            print(f"  {cluster_name}: Topics {topic_indices}")
        n_clusters = len(cluster_mapping)
        cluster_names = list(cluster_mapping.keys())
    else:
        n_clusters = 0
        cluster_names = []

    # Get chunk indices
    chunk_indices = roll.chunk_indices
    n_chunks = len(chunk_indices)
    print(f"  Number of time chunks: {n_chunks}")

    # Get time labels
    time_labels = []
    for i in range(n_chunks):
        date = chunk_indices["date"].iloc[i]
        if hasattr(date, 'strftime'):
            time_labels.append(date.strftime('%Y-%m'))
        else:
            time_labels.append(str(date)[:7])
    print(f"  Time labels: {time_labels}")

    # Get vocabulary and top words per topic per time chunk
    vocabulary = roll.lda.get_vocab()

    # Get top words for each time chunk
    top_words_by_chunk = {}  # {time_label: {topic_idx: [words]}}
    for chunk_idx in range(n_chunks):
        time_label = time_labels[chunk_idx]
        word_topic_matrix = roll.get_word_topic_matrix(chunk_idx)
        top_words_by_chunk[time_label] = {}
        for topic_idx in range(n_topics):
            top_indices = np.argsort(word_topic_matrix[:, topic_idx])[::-1][:10]
            top_words_by_chunk[time_label][topic_idx] = [vocabulary[i] for i in top_indices]

    # Also keep top_words for backward compatibility (last chunk)
    top_words = top_words_by_chunk[time_labels[-1]]

    # Location column mapping
    granularity_to_column = {
        'region': 'region',
        'province': 'province',
        'comuna': 'municipality'
    }

    # Initialize result structure
    result = {
        'time_labels': time_labels,
        'n_topics': n_topics,
        'top_words': top_words,
        'top_words_by_chunk': top_words_by_chunk,  # Top words for each time chunk
        'granularity_levels': list(granularity_to_column.keys()),
        'prevalence_data': {},
        'overall_prevalence': {},  # Overall prevalence per time chunk (all locations combined)
        'cluster_mapping': cluster_mapping if cluster_mapping else {},
        'cluster_names': cluster_names,
        'n_clusters': n_clusters
    }

    # Compute overall prevalence per time chunk (all locations combined)
    print("\nComputing overall prevalence per time chunk...")
    for chunk_idx in range(n_chunks):
        start_idx = chunk_indices["chunk_start"].iloc[chunk_idx]
        if chunk_idx < n_chunks - 1:
            end_idx = chunk_indices["chunk_start"].iloc[chunk_idx + 1]
        else:
            end_idx = len(docs)

        chunk_topics = full_doc_topic_matrix[start_idx:end_idx]
        time_label = time_labels[chunk_idx]

        # Average topic prevalence across all documents in this chunk
        avg_prevalence = chunk_topics.mean(axis=0).tolist()
        doc_count = len(chunk_topics)

        result['overall_prevalence'][time_label] = {
            'topic_prevalence': avg_prevalence,
            'doc_count': doc_count
        }

        # Compute cluster prevalence if cluster mapping is provided
        if cluster_mapping:
            cluster_prevalence = {}
            for cluster_name, topic_indices in cluster_mapping.items():
                # Sum the prevalences of topics in this cluster
                cluster_prev = sum(avg_prevalence[i] for i in topic_indices if i < n_topics)
                cluster_prevalence[cluster_name] = cluster_prev
            result['overall_prevalence'][time_label]['cluster_prevalence'] = cluster_prevalence

    # Process each granularity level
    for granularity, column in granularity_to_column.items():
        print(f"\nProcessing granularity: {granularity} (column: {column})")

        if column not in docs.columns:
            print(f"  Warning: Column '{column}' not found in data, skipping...")
            continue

        result['prevalence_data'][granularity] = {}

        # Get all unique locations for this granularity
        all_locations = docs[column].dropna().unique().tolist()
        print(f"  Found {len(all_locations)} unique locations")

        # Process each time chunk
        for chunk_idx in range(n_chunks):
            # Get chunk boundaries
            start_idx = chunk_indices["chunk_start"].iloc[chunk_idx]
            if chunk_idx < n_chunks - 1:
                end_idx = chunk_indices["chunk_start"].iloc[chunk_idx + 1]
            else:
                end_idx = len(docs)

            # Get documents and topic distributions for this chunk
            chunk_docs = docs.iloc[start_idx:end_idx]
            chunk_topics = full_doc_topic_matrix[start_idx:end_idx]

            time_label = time_labels[chunk_idx]

            # Process each location
            for location in all_locations:
                # Filter documents for this location
                location_mask = chunk_docs[column] == location
                location_topics = chunk_topics[location_mask.values]

                if len(location_topics) == 0:
                    continue

                # Calculate average topic prevalence for this location in this time chunk
                avg_prevalence = location_topics.mean(axis=0).tolist()
                doc_count = len(location_topics)

                # Initialize location entry if needed
                if location not in result['prevalence_data'][granularity]:
                    result['prevalence_data'][granularity][location] = {
                        'time_series': {},
                        'total_docs': 0
                    }

                # Store the topic prevalence data
                time_entry = {
                    'prevalence': avg_prevalence,
                    'doc_count': doc_count
                }

                # Compute cluster prevalence if cluster mapping is provided
                if cluster_mapping:
                    cluster_prevalence = {}
                    for cluster_name, topic_indices in cluster_mapping.items():
                        # Sum the prevalences of topics in this cluster
                        cluster_prev = sum(avg_prevalence[i] for i in topic_indices if i < n_topics)
                        cluster_prevalence[cluster_name] = cluster_prev
                    time_entry['cluster_prevalence'] = cluster_prevalence

                result['prevalence_data'][granularity][location]['time_series'][time_label] = time_entry
                result['prevalence_data'][granularity][location]['total_docs'] += doc_count

        # Count locations with data
        locations_with_data = len(result['prevalence_data'][granularity])
        print(f"  Locations with data: {locations_with_data}")

    # Save to JSON file
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Done!")

    # Print summary
    print("\n=== Summary ===")
    print(f"Time periods: {n_chunks}")
    print(f"Topics: {n_topics}")
    if cluster_mapping:
        print(f"Clusters: {n_clusters} ({', '.join(cluster_names)})")
    for granularity in result['prevalence_data']:
        n_locations = len(result['prevalence_data'][granularity])
        print(f"{granularity}: {n_locations} locations")

    return result


def load_prevalence_data(path='topic_prevalence_by_location.json'):
    """
    Load the pre-computed prevalence data from JSON file.

    Parameters:
    -----------
    path : str
        Path to the JSON file

    Returns:
    --------
    dict : The prevalence data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    # Check if required files exist
    if not os.path.exists('roll_lda.pickle'):
        print("Error: roll_lda.pickle not found. Please run toy_example_with_lda.py first.")
        exit(1)

    if not os.path.exists('english_database.xlsx'):
        print("Error: english_database.xlsx not found.")
        exit(1)

    # Example cluster mapping - customize this based on your topics!
    # Set to None to disable clustering, or define your own mapping
    example_cluster_mapping = {
        "Cluster A": [0, 1],      # Topics 0 and 1
        "Cluster B": [2, 3, 4]    # Topics 2, 3, and 4
    }

    # Run the analysis with cluster mapping
    # Set cluster_mapping=None to disable clustering
    result = analyze_topic_prevalence(cluster_mapping=example_cluster_mapping)

    # Print example of the data structure
    print("\n=== Data Structure Example ===")
    print("Top-level keys:", list(result.keys()))

    # Show overall prevalence example
    if result['overall_prevalence']:
        first_time = list(result['overall_prevalence'].keys())[0]
        print(f"\nOverall prevalence for {first_time}:")
        print(json.dumps(result['overall_prevalence'][first_time], indent=2))

    # Show location example
    if result['prevalence_data']:
        first_granularity = list(result['prevalence_data'].keys())[0]
        first_location = list(result['prevalence_data'][first_granularity].keys())[0]
        print(f"\nExample for {first_granularity} -> {first_location}:")
        location_data = result['prevalence_data'][first_granularity][first_location]
        first_time = list(location_data['time_series'].keys())[0]
        print(f"Time series entry for {first_time}:")
        print(json.dumps(location_data['time_series'][first_time], indent=2))
