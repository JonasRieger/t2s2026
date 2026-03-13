import numpy as np
import pandas as pd
import pickle
import os
import json


def load_topic_descriptions(csv_path):
    """
    Load topic descriptions from a CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing topic descriptions.
        Expected columns: topic_id, title, explanation

    Returns:
    --------
    dict : Dictionary mapping topic_id (int) to dict with 'title' and 'explanation'
    """
    if csv_path is None or not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)
        descriptions = {}
        for _, row in df.iterrows():
            topic_id = int(row['topic_id'])
            descriptions[topic_id] = {
                'title': row['title'],
                'explanation': row.get('explanation', '')
            }
        return descriptions
    except Exception as e:
        print(f"Warning: Could not load topic descriptions from {csv_path}: {e}")
        return {}


def load_word_impacts(pickle_path):
    """
    Load word impacts from a pickle file.

    Parameters:
    -----------
    pickle_path : str
        Path to the pickle file containing word impacts.
        Expected DataFrame columns: Topic, Date, Significant Words, Impacts

    Returns:
    --------
    dict : Dictionary mapping topic_id -> year -> {words: [...], impacts: [...]}
    """
    if pickle_path is None or not os.path.exists(pickle_path):
        return {}

    try:
        df = pd.read_pickle(pickle_path)
        word_impacts = {}

        for _, row in df.iterrows():
            topic_id = int(row['Topic'])
            # Extract year from date (e.g., "2012-12-31" -> "2012")
            date_str = str(row['Date'])
            year = date_str[:4]

            words = list(row['Significant Words']) if hasattr(row['Significant Words'], '__iter__') else []
            impacts = list(row['Impacts']) if hasattr(row['Impacts'], '__iter__') else []

            if topic_id not in word_impacts:
                word_impacts[topic_id] = {}

            word_impacts[topic_id][year] = {
                'words': words,
                'impacts': [float(v) for v in impacts]
            }

        return word_impacts
    except Exception as e:
        print(f"Warning: Could not load word impacts from {pickle_path}: {e}")
        return {}


def load_word_translations(csv_path):
    """
    Load word translations from a CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing word translations.
        Expected columns: spanish, english

    Returns:
    --------
    dict : Dictionary mapping Spanish words to English translations
    """
    if csv_path is None or not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)
        translations = {}
        for _, row in df.iterrows():
            spanish = row['spanish'].strip().lower()
            english = row['english'].strip()
            translations[spanish] = english
        return translations
    except Exception as e:
        print(f"Warning: Could not load word translations from {csv_path}: {e}")
        return {}


def load_cluster_definitions(csv_path, topic_descriptions=None):
    """
    Load cluster definitions from a CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing cluster definitions.
        Expected columns: cluster_id, topic_ids
        Optional columns: title, description (will be auto-generated if not present)
    topic_descriptions : dict, optional
        Dictionary mapping topic_id to dict with 'title' and 'explanation'
        Used for generating cluster titles/descriptions if not in CSV

    Returns:
    --------
    dict : Dictionary with:
        - 'cluster_mapping': dict mapping cluster names to list of topic indices
        - 'cluster_names': list of cluster names in order
        - 'cluster_titles': dict mapping cluster names to display titles
        - 'cluster_descriptions': dict mapping cluster names to descriptions
        - 'n_clusters': number of clusters
    """
    if csv_path is None or not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)

        cluster_mapping = {}
        cluster_names = []
        cluster_titles = {}
        cluster_descriptions = {}

        # Check if title and description columns exist in CSV
        has_title_col = 'title' in df.columns
        has_desc_col = 'description' in df.columns

        for _, row in df.iterrows():
            cluster_id = int(row['cluster_id'])
            # Parse topic_ids - handle both string "0,1,2" and list formats
            topic_ids_raw = row['topic_ids']
            if isinstance(topic_ids_raw, str):
                topic_ids = [int(t.strip()) for t in topic_ids_raw.split(',')]
            else:
                topic_ids = [int(topic_ids_raw)]

            # Create cluster name (C1, C2, etc.)
            cluster_num = cluster_id + 1
            cluster_name = f"C{cluster_num}"

            # Use title from CSV if available, otherwise generate it
            if has_title_col and pd.notna(row['title']) and str(row['title']).strip():
                title = str(row['title']).strip()
            else:
                title = generate_cluster_title(topic_ids, topic_descriptions)

            # Use description from CSV if available, otherwise generate it
            if has_desc_col and pd.notna(row['description']) and str(row['description']).strip():
                description = str(row['description']).strip()
            else:
                description = generate_cluster_description(topic_ids, topic_descriptions)

            cluster_mapping[cluster_name] = topic_ids
            cluster_names.append(cluster_name)
            # Store the short title (without "C1:" prefix) for panel display
            cluster_titles[cluster_name] = f"{cluster_name}: {title}"
            cluster_descriptions[cluster_name] = description

        return {
            'cluster_mapping': cluster_mapping,
            'cluster_names': cluster_names,
            'cluster_titles': cluster_titles,  # Short titles for panel display
            'cluster_descriptions': cluster_descriptions,
            'n_clusters': len(cluster_names)
        }
    except Exception as e:
        print(f"Warning: Could not load cluster definitions from {csv_path}: {e}")
        return None


def generate_cluster_title(topic_ids, topic_descriptions):
    """
    Generate a short title for a cluster based on its topics.

    Parameters:
    -----------
    topic_ids : list
        List of topic indices in this cluster
    topic_descriptions : dict
        Dictionary mapping topic_id to dict with 'title' and 'explanation'

    Returns:
    --------
    str : Generated cluster title
    """
    if not topic_descriptions:
        return f"Topics {', '.join(str(t) for t in topic_ids)}"

    # Collect topic titles
    topic_titles = []
    for tid in topic_ids:
        if tid in topic_descriptions:
            topic_titles.append(topic_descriptions[tid].get('title', ''))

    if not topic_titles:
        return f"Topics {', '.join(str(t) for t in topic_ids)}"

    # Generate a synthesized title based on common themes
    # Extract key terms from topic titles
    title_words = ' '.join(topic_titles).lower()

    # Count thematic keywords to determine dominant theme
    # Be more specific with keyword detection

    # Mining theme - prioritize if copper/lithium mining present
    mining_keywords = ['lithium mining', 'copper mining', 'mining']
    has_mining = any(kw in title_words for kw in mining_keywords)

    # Industrial pollution theme (exclude "light pollution")
    pollution_keywords = ['industrial pollution', 'contamination', 'sacrifice zone', 'health crisis']
    has_industrial_pollution = any(kw in title_words for kw in pollution_keywords)

    # Environmental/Energy theme
    env_energy_keywords = ['environmental impact', 'hydroelectric', 'wind', 'energy transition',
                           'environmental court', 'environmental assessment']
    has_env_energy = any(kw in title_words for kw in env_energy_keywords)

    # Government/Policy theme
    gov_keywords = ['government', 'legislation', 'hydrogen development', 'assessment (sma)']
    has_gov = any(kw in title_words for kw in gov_keywords)

    # Determine title based on dominant themes
    if has_mining:
        if 'water' in title_words:
            return "Mining & Water Resources"
        return "Resource Extraction"
    elif has_industrial_pollution and has_gov:
        return "Industrial Development & Policy"
    elif has_industrial_pollution:
        return "Industrial Pollution & Health"
    elif has_env_energy:
        if 'court' in title_words:
            return "Environmental Regulation"
        return "Environmental & Energy"
    elif has_gov:
        return "Governance & Policy"
    elif 'maritime' in title_words:
        return "Maritime & Services"
    else:
        # Use the first topic's main keyword
        first_title = topic_titles[0] if topic_titles else ""
        main_word = first_title.split('&')[0].strip() if '&' in first_title else first_title.split()[0] if first_title else "Cluster"
        return f"{main_word} Topics"


def generate_cluster_description(topic_ids, topic_descriptions):
    """
    Generate a natural language description for a cluster based on its topics.

    Parameters:
    -----------
    topic_ids : list
        List of topic indices in this cluster
    topic_descriptions : dict
        Dictionary mapping topic_id to dict with 'title' and 'explanation'

    Returns:
    --------
    str : Generated cluster description
    """
    if not topic_descriptions:
        return f"This cluster contains topics {', '.join(str(t) for t in topic_ids)}."

    # Collect topic information
    topic_info = []
    for tid in topic_ids:
        if tid in topic_descriptions:
            title = topic_descriptions[tid].get('title', f'Topic {tid}')
            topic_info.append(title)

    if not topic_info:
        return f"This cluster contains topics {', '.join(str(t) for t in topic_ids)}."

    # Generate description based on topic count and content
    n_topics = len(topic_info)

    if n_topics == 1:
        return f"This cluster focuses on {topic_info[0].lower()}."
    elif n_topics == 2:
        return f"This cluster encompasses {topic_info[0]} and {topic_info[1]}."
    else:
        # List all topics with proper grammar
        topic_list = ', '.join(topic_info[:-1]) + f', and {topic_info[-1]}'

        # Generate thematic summary based on specific content patterns
        title_words = ' '.join(topic_info).lower()

        # Mining cluster
        mining_keywords = ['lithium mining', 'copper mining']
        has_mining = any(kw in title_words for kw in mining_keywords)

        # Industrial pollution cluster
        pollution_keywords = ['industrial pollution', 'contamination']
        has_pollution = any(kw in title_words for kw in pollution_keywords)

        # Environmental/Energy cluster
        env_keywords = ['environmental impact', 'hydroelectric', 'wind', 'energy transition', 'environmental court']
        has_env = any(kw in title_words for kw in env_keywords)

        # Government/policy cluster
        gov_keywords = ['government', 'hydrogen', 'assessment']
        has_gov = any(kw in title_words for kw in gov_keywords)

        if has_mining:
            theme = "Chile's extractive industries and their environmental implications, including conflicts over water resources and community relations"
        elif has_pollution and has_gov:
            theme = "industrial development, pollution control, governmental oversight, and emerging clean energy industries"
        elif has_pollution:
            theme = "industrial activities and their impacts on public health and the environment in affected communities"
        elif has_env:
            theme = "environmental impacts of energy projects, regulatory processes, court decisions, and Chile's energy transition policies"
        elif has_gov:
            theme = "governmental oversight, emerging industries, and environmental assessment procedures"
        else:
            theme = "multiple interconnected environmental and energy topics in Chile"

        return f"This cluster addresses {theme}. It covers {topic_list}."


def update_cluster_csv_with_metadata(csv_path, cluster_data, topic_descriptions):
    """
    Update the cluster CSV file with generated titles and descriptions.
    Only fills in missing titles/descriptions, preserves existing ones.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    cluster_data : dict
        Cluster data returned by load_cluster_definitions
    topic_descriptions : dict
        Dictionary mapping topic_id to dict with 'title' and 'explanation'
    """
    if not csv_path or not os.path.exists(csv_path):
        return

    try:
        df = pd.read_csv(csv_path)

        # Check if columns exist, create them if not
        if 'title' not in df.columns:
            df['title'] = ''
        if 'description' not in df.columns:
            df['description'] = ''

        # Track if any updates were made
        updates_made = False

        # Only fill in missing titles and descriptions
        for idx, row in df.iterrows():
            cluster_id = int(row['cluster_id'])
            topic_ids_raw = row['topic_ids']
            if isinstance(topic_ids_raw, str):
                topic_ids = [int(t.strip()) for t in topic_ids_raw.split(',')]
            else:
                topic_ids = [int(topic_ids_raw)]

            # Only generate title if it's missing or empty
            if pd.isna(row['title']) or str(row['title']).strip() == '':
                df.at[idx, 'title'] = generate_cluster_title(topic_ids, topic_descriptions)
                updates_made = True

            # Only generate description if it's missing or empty
            if pd.isna(row['description']) or str(row['description']).strip() == '':
                df.at[idx, 'description'] = generate_cluster_description(topic_ids, topic_descriptions)
                updates_made = True

        if updates_made:
            df.to_csv(csv_path, index=False)
            print(f"Updated {csv_path} with generated titles and descriptions for missing entries")
        else:
            print(f"No updates needed for {csv_path} - all titles and descriptions already present")

    except Exception as e:
        print(f"Warning: Could not update cluster CSV: {e}")


# Assuming you have:
# - document_topic_matrix: numpy array (n_docs, n_topics)
# - word_topic_matrix: numpy array (n_words, n_topics)
# - locations: list of location names for each document
# - vocabulary: list of words


def load_rolling_lda_data(model_path, data_path, sheet_name="Filtered_Conflicts"):
    """
    Load RollingLDA model and extract data for visualization.

    Parameters:
    -----------
    model_path : str
        Path to the saved RollingLDA model pickle file
    data_path : str
        Path to the Excel file with the original data
    sheet_name : str
        Name of the sheet to read from the Excel file

    Returns:
    --------
    document_topic_matrices : list of numpy arrays
        List of document-topic matrices, one per time chunk
    word_topic_matrices : list of numpy arrays
        List of word-topic matrices, one per time chunk
    locations_dict : dict
        Dictionary with keys 'region', 'province', 'municipality', 'locality'
        each containing a list of location lists per time chunk
    vocabulary : list
        List of words in the vocabulary
    time_labels : list
        List of time period labels (formatted dates)
    """
    from ttta.methods.rolling_lda import RollingLDA

    # Load the model
    roll = RollingLDA(5)  # n_topics doesn't matter when loading
    roll.load(model_path)

    # Load the original data to get location columns
    docs = pd.read_excel(data_path, sheet_name=sheet_name)
    docs = docs[docs["date"].isna() == False]

    # Recreate the sorting done by RollingLDA internally
    docs = docs.loc[roll.sorting].reset_index(drop=True)

    # Get vocabulary
    vocabulary = roll.lda.get_vocab()

    # Get word-topic matrices per time chunk
    n_chunks = len(roll.chunk_indices)
    word_topic_matrices = [roll.get_word_topic_matrix(x) for x in range(n_chunks)]

    # Get full document-topic matrix
    full_doc_topic_matrix = roll.get_document_topic_matrix()
    full_doc_topic_matrix = full_doc_topic_matrix / full_doc_topic_matrix.sum(axis=1,
                                                                        keepdims=True)

    # Split document-topic matrix by chunk boundaries
    document_topic_matrices = []
    chunk_indices = roll.chunk_indices

    for i in range(n_chunks):
        start_idx = chunk_indices["chunk_start"].iloc[i]
        if i < n_chunks - 1:
            end_idx = chunk_indices["chunk_start"].iloc[i + 1]
        else:
            end_idx = len(full_doc_topic_matrix)
        document_topic_matrices.append(full_doc_topic_matrix[start_idx:end_idx])

    # Build locations_dict with granularity levels
    # Map granularity names to actual column names in the data
    granularity_to_column = {
        'region': 'region',
        'province': 'province',
        'comuna': 'municipality'  # Data uses 'municipality' for comunas
    }
    location_columns = ['region', 'province', 'comuna']
    locations_dict = {col: [] for col in location_columns}

    for i in range(n_chunks):
        start_idx = chunk_indices["chunk_start"].iloc[i]
        if i < n_chunks - 1:
            end_idx = chunk_indices["chunk_start"].iloc[i + 1]
        else:
            end_idx = len(docs)

        for col in location_columns:
            data_col = granularity_to_column.get(col, col)
            if data_col in docs.columns:
                locs = docs[data_col].iloc[start_idx:end_idx].fillna('Unknown').tolist()
            else:
                locs = ['Unknown'] * (end_idx - start_idx)
            locations_dict[col].append(locs)

    # Get time labels from chunk dates (year only)
    time_labels = []
    for i in range(n_chunks):
        date = chunk_indices["date"].iloc[i]
        if hasattr(date, 'strftime'):
            time_labels.append(date.strftime('%Y'))
        else:
            time_labels.append(str(date)[:4])

    return document_topic_matrices, word_topic_matrices, locations_dict, vocabulary, time_labels


# 1. Calculate topic prevalence by location
def calculate_location_topic_prevalence(document_topic_matrix, locations):
    """
    Calculate average topic prevalence for each location.

    Returns:
        DataFrame with locations and their average topic distributions
    """
    df = pd.DataFrame(document_topic_matrix)
    df['location'] = locations

    # Group by location and calculate mean topic prevalence
    location_topics = df.groupby('location').agg({
        col: 'mean' for col in df.columns if col != 'location'
        })

    # Add document count per location
    location_topics['doc_count'] = df.groupby('location').size()

    return location_topics


# 2. Get top words for each topic
def get_top_words_per_topic(word_topic_matrix, vocabulary, n_words=10):
    """
    Extract top N words for each topic.

    Returns:
        Dictionary mapping topic index to list of top words
    """
    top_words = {}
    n_topics = word_topic_matrix.shape[1]

    for topic_idx in range(n_topics):
        top_indices = np.argsort(word_topic_matrix[:, topic_idx])[::-1][
            :n_words]
        top_words[topic_idx] = [vocabulary[i] for i in top_indices]

    return top_words


def create_spatiotemporal_interactive_map(document_topic_matrices,
                                          word_topic_matrices,
                                          locations_list, vocabulary,
                                          time_labels,
                                          output_file='spatiotemporal_topics.html',
                                          coordinates=None,
                                          title='Spatiotemporal Topic Map',
                                          map_bounds=None,
                                          geojson=None,
                                          location_name_property='name',
                                          chart_data=None,
                                          topic_descriptions_csv=None,
                                          word_translations_csv=None,
                                          cluster_sets_csv=None,
                                          word_impacts_path=None):
    """
    Create an interactive map with both time period and topic selection.

    Parameters:
    -----------
    document_topic_matrices : list of numpy arrays
        List of document-topic matrices, one per time period
    word_topic_matrices : list of numpy arrays
        List of word-topic matrices, one per time period
    locations_list : list of lists OR dict
        Either a list of location lists (one per time period) for backwards compatibility,
        OR a dict with keys like 'region', 'province', 'municipality', 'locality'
        each containing a list of location lists per time chunk for granularity switching.
    vocabulary : list
        Vocabulary (can be same across time periods or different)
    time_labels : list of str
        Labels for each time period (e.g., ['2020-Q1', '2020-Q2', ...])
    output_file : str
        Output HTML filename
    coordinates : dict, optional
        Dictionary mapping location names to (lat, lon) tuples.
        If None, uses the built-in Chilean geocoder.
        Only used when geojson is None (for circle markers).
    title : str, optional
        Title for the map (default: 'Spatiotemporal Topic Map')
    map_bounds : list, optional
        Custom map bounds as [[sw_lat, sw_lon], [ne_lat, ne_lon]].
        If None, bounds are calculated automatically from the coordinates.
    geojson : dict or str, optional
        GeoJSON data for choropleth visualization. Can be:
        - A dict (parsed GeoJSON FeatureCollection)
        - A str (path to a GeoJSON file)
        - A dict mapping granularity level names to GeoJSON data/paths
        If provided, regions will be filled polygons instead of circles.
    location_name_property : str or dict, optional
        The property name in GeoJSON features that contains the location name.
        Can be a single string (used for all granularity levels) or a dict
        mapping granularity levels to property names.
        Default: 'name'
    topic_descriptions_csv : str, optional
        Path to a CSV file containing topic descriptions.
        Expected columns: topic_id, title, explanation
        If not provided or file not found, topics will be labeled as "Topic 0", "Topic 1", etc.
    word_translations_csv : str, optional
        Path to a CSV file containing word translations (Spanish to English).
        Expected columns: spanish, english
        If not provided, translation hover functionality will be disabled.
    cluster_sets_csv : str, optional
        Path to a CSV file containing cluster definitions.
        Expected columns: cluster_id, topic_ids
        Optional columns: title, description (will be auto-generated if not present)
        If provided, clusters will be read from this file and the CSV will be updated
        with generated titles and descriptions.
    word_impacts_path : str, optional
        Path to a pickle file containing word impact data.
        Expected DataFrame columns: Topic, Date, Significant Words, Impacts
        If provided, hovering over year headers in the top words table will show
        a bar plot of word impacts (only for topic mode, not corpus/clusters).
    """

    n_time_periods = len(document_topic_matrices)
    n_topics = document_topic_matrices[0].shape[1]

    # Load topic descriptions from CSV if provided, otherwise use generic names
    topic_descriptions = load_topic_descriptions(topic_descriptions_csv)

    # Load cluster definitions from CSV if provided
    cluster_data = None
    if cluster_sets_csv:
        cluster_data = load_cluster_definitions(cluster_sets_csv, topic_descriptions)
        if cluster_data:
            # Update the CSV with generated titles and descriptions
            update_cluster_csv_with_metadata(cluster_sets_csv, cluster_data, topic_descriptions)
            print(f"Loaded {cluster_data['n_clusters']} clusters from {cluster_sets_csv}")

            # Merge cluster data into chart_data if chart_data exists
            if chart_data:
                chart_data['cluster_mapping'] = cluster_data['cluster_mapping']
                chart_data['cluster_names'] = cluster_data['cluster_names']
                chart_data['cluster_titles'] = cluster_data['cluster_titles']
                chart_data['cluster_descriptions'] = cluster_data['cluster_descriptions']
                chart_data['n_clusters'] = cluster_data['n_clusters']
            else:
                # Create minimal chart_data with cluster info
                chart_data = {
                    'cluster_mapping': cluster_data['cluster_mapping'],
                    'cluster_names': cluster_data['cluster_names'],
                    'cluster_titles': cluster_data['cluster_titles'],
                    'cluster_descriptions': cluster_data['cluster_descriptions'],
                    'n_clusters': cluster_data['n_clusters']
                }

    # Load word translations if provided
    word_translations = load_word_translations(word_translations_csv)

    # Load word impacts if provided
    word_impacts = load_word_impacts(word_impacts_path)

    # Build topic_titles dict: use CSV titles if available, fall back to "Topic X"
    # Prefix all titles with "TX: " where X is 1-indexed topic number
    topic_titles = {}
    topic_explanations = {}
    for i in range(n_topics):
        topic_num = i + 1  # 1-indexed topic number
        if i in topic_descriptions:
            topic_titles[i] = f"T{topic_num}: {topic_descriptions[i]['title']}"
            topic_explanations[i] = topic_descriptions[i].get('explanation', '')
        else:
            topic_titles[i] = f"T{topic_num}: Topic {i}"
            topic_explanations[i] = ""

    # Determine if we have granularity support (dict) or legacy format (list)
    if isinstance(locations_list, dict):
        granularity_levels = list(locations_list.keys())
        has_granularity = True
    else:
        # Convert legacy list format to dict with single 'locality' key
        granularity_levels = ['locality']
        locations_list = {'locality': locations_list}
        has_granularity = False

    # Process GeoJSON data if provided
    use_geojson = geojson is not None
    geojson_data = {}

    if use_geojson:
        import json as json_module

        def load_geojson(data):
            """Load GeoJSON from dict or file path."""
            if isinstance(data, dict):
                return data
            elif isinstance(data, str):
                with open(data, 'r', encoding='utf-8') as f:
                    return json_module.load(f)
            else:
                raise ValueError(f"Invalid GeoJSON data type: {type(data)}")

        # Handle GeoJSON as dict mapping granularity->data or single GeoJSON for all
        if isinstance(geojson, dict) and 'type' not in geojson:
            # Dict mapping granularity levels to GeoJSON
            for granularity in granularity_levels:
                if granularity in geojson:
                    geojson_data[granularity] = load_geojson(geojson[granularity])
                else:
                    print(f"Warning: No GeoJSON provided for granularity '{granularity}'")
                    geojson_data[granularity] = None
        else:
            # Single GeoJSON for all granularity levels
            loaded = load_geojson(geojson)
            for granularity in granularity_levels:
                geojson_data[granularity] = loaded

        # Get location name property for each granularity
        if isinstance(location_name_property, dict):
            name_props = location_name_property
        else:
            name_props = {g: location_name_property for g in granularity_levels}

        # Extract valid location names from each GeoJSON for filtering
        # This ensures only locations that can be displayed on the map are included in statistics
        valid_geojson_names = {}
        for granularity in granularity_levels:
            gj = geojson_data.get(granularity)
            if gj is not None:
                name_prop = name_props.get(granularity, 'name')
                valid_names = set()
                for feature in gj.get('features', []):
                    name = feature.get('properties', {}).get(name_prop)
                    if name:
                        valid_names.add(name)
                valid_geojson_names[granularity] = valid_names
                print(f"  GeoJSON {granularity}: {len(valid_names)} valid location names")
            else:
                valid_geojson_names[granularity] = None

    # Define geocoding function based on whether custom coordinates are provided
    # Only used when not using GeoJSON
    def geocode_locations(locations):
        if coordinates is not None:
            # Use custom coordinates
            coords = []
            for loc in locations:
                if loc in coordinates:
                    coords.append(coordinates[loc])
                else:
                    # Try case-insensitive match
                    loc_lower = str(loc).lower() if loc else ''
                    found = False
                    for key, coord in coordinates.items():
                        if str(key).lower() == loc_lower:
                            coords.append(coord)
                            found = True
                            break
                    if not found:
                        coords.append(None)
            return coords
        else:
            # Use built-in Chilean geocoder
            return geocode_chilean_locations(locations)

    # Process data for each time period AND each granularity level
    all_granularity_data = {}
    all_coords = []  # Collect all coordinates for bounds calculation (circle mode only)

    # Track excluded locations and their document counts for reporting
    excluded_locations_report = {}

    for granularity in granularity_levels:
        all_time_data = []
        excluded_locations_report[granularity] = {}

        # Check if this granularity has GeoJSON
        has_geojson_for_granularity = use_geojson and geojson_data.get(granularity) is not None

        for t_idx in range(n_time_periods):
            doc_topic = document_topic_matrices[t_idx]
            word_topic = word_topic_matrices[t_idx]
            locs = locations_list[granularity][t_idx]

            # Calculate location-topic prevalence for this time period
            location_topics = calculate_location_topic_prevalence(doc_topic, locs)

            # Get top words for each topic
            top_words = get_top_words_per_topic(word_topic, vocabulary)

            # Calculate total word counts per topic from word-topic matrix
            total_word_counts_per_topic = word_topic.sum(axis=0)

            # Calculate word counts per location per topic
            # Distribute total word counts proportionally based on document-topic prevalence
            # For each location, we compute: (sum of doc_topic for docs in location) / (total topic prevalence) * total_word_counts
            topic_prevalence_totals = np.zeros(n_topics)
            location_topic_sums = {}

            for doc_idx, loc in enumerate(locs):
                if loc not in location_topic_sums:
                    location_topic_sums[loc] = np.zeros(n_topics)
                location_topic_sums[loc] += doc_topic[doc_idx]
                topic_prevalence_totals += doc_topic[doc_idx]

            # Get unique locations
            unique_locations = location_topics.index.tolist()

            # Prepare data for this time period
            location_data = {}

            if has_geojson_for_granularity:
                # GeoJSON mode for this granularity: no coordinates needed
                # Get valid names for filtering (only include locations that exist in GeoJSON)
                valid_names = valid_geojson_names.get(granularity)

                for location in unique_locations:
                    # Skip locations that don't exist in the GeoJSON
                    # These cannot be displayed on the map and should not be counted in statistics
                    if valid_names is not None and location not in valid_names:
                        # Track excluded location and its document count
                        doc_count = int(location_topics.loc[location, 'doc_count'])
                        if location not in excluded_locations_report[granularity]:
                            excluded_locations_report[granularity][location] = 0
                        excluded_locations_report[granularity][location] += doc_count
                        continue

                    # Calculate word counts for this location
                    if location in location_topic_sums:
                        loc_word_counts = []
                        for t in range(n_topics):
                            if topic_prevalence_totals[t] > 0:
                                loc_word_counts.append(int(location_topic_sums[location][t] / topic_prevalence_totals[t] * total_word_counts_per_topic[t]))
                            else:
                                loc_word_counts.append(0)
                    else:
                        loc_word_counts = [0] * n_topics

                    location_data[location] = {
                        'doc_count': int(location_topics.loc[location, 'doc_count']),
                        'topics': [float(location_topics.loc[location, i]) for i in
                                   range(n_topics)],
                        'word_counts': loc_word_counts
                        }
            else:
                # Circle marker mode: need coordinates
                coords_map = dict(
                    zip(unique_locations, geocode_locations(unique_locations)))

                for location in unique_locations:
                    if coords_map[location] is None:
                        continue
                    # Calculate word counts for this location
                    if location in location_topic_sums:
                        loc_word_counts = []
                        for t in range(n_topics):
                            if topic_prevalence_totals[t] > 0:
                                loc_word_counts.append(int(location_topic_sums[location][t] / topic_prevalence_totals[t] * total_word_counts_per_topic[t]))
                            else:
                                loc_word_counts.append(0)
                    else:
                        loc_word_counts = [0] * n_topics

                    location_data[location] = {
                        'coords': coords_map[location],
                        'doc_count': int(location_topics.loc[location, 'doc_count']),
                        'topics': [float(location_topics.loc[location, i]) for i in
                                   range(n_topics)],
                        'word_counts': loc_word_counts
                        }
                    # Collect coordinates for bounds calculation
                    all_coords.append(coords_map[location])

            all_time_data.append({
                'location_data': location_data,
                'top_words': top_words,
                'time_label': time_labels[t_idx]
                })

        all_granularity_data[granularity] = all_time_data

    # Print report of excluded locations (locations not in GeoJSON)
    if use_geojson:
        for granularity in granularity_levels:
            excluded = excluded_locations_report.get(granularity, {})
            if excluded:
                total_excluded_docs = sum(excluded.values())
                print(f"  {granularity.upper()}: Excluded {len(excluded)} locations ({total_excluded_docs} documents) - not in GeoJSON:")
                for loc, doc_count in sorted(excluded.items()):
                    print(f"    - '{loc}': {doc_count} documents")
            else:
                print(f"  {granularity.upper()}: All locations matched GeoJSON features")

    # For backwards compatibility, use first granularity level for top_words reference
    first_granularity = granularity_levels[0]
    all_time_data = all_granularity_data[first_granularity]

    # Calculate map bounds and center from coordinates if not provided
    if map_bounds is not None:
        bounds = map_bounds
    elif use_geojson:
        # Extract bounds from GeoJSON geometry
        all_lats = []
        all_lons = []

        def extract_coords_from_geometry(geometry):
            """Recursively extract all coordinates from a GeoJSON geometry."""
            geom_type = geometry.get('type', '')
            coords = geometry.get('coordinates', [])

            if geom_type == 'Point':
                return [(coords[1], coords[0])]  # GeoJSON is [lon, lat]
            elif geom_type in ('LineString', 'MultiPoint'):
                return [(c[1], c[0]) for c in coords]
            elif geom_type in ('Polygon', 'MultiLineString'):
                result = []
                for ring in coords:
                    result.extend([(c[1], c[0]) for c in ring])
                return result
            elif geom_type == 'MultiPolygon':
                result = []
                for polygon in coords:
                    for ring in polygon:
                        result.extend([(c[1], c[0]) for c in ring])
                return result
            elif geom_type == 'GeometryCollection':
                result = []
                for geom in geometry.get('geometries', []):
                    result.extend(extract_coords_from_geometry(geom))
                return result
            return []

        for granularity, gj in geojson_data.items():
            if gj is None:
                continue
            for feature in gj.get('features', []):
                geometry = feature.get('geometry')
                if geometry:
                    for lat, lon in extract_coords_from_geometry(geometry):
                        all_lats.append(lat)
                        all_lons.append(lon)

        if all_lats and all_lons:
            padding = 0.5  # Smaller padding for GeoJSON
            bounds = [
                [min(all_lats) - padding, min(all_lons) - padding],
                [max(all_lats) + padding, max(all_lons) + padding]
            ]
        else:
            bounds = [[-60, -180], [60, 180]]
    elif all_coords:
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        padding = 2.0  # Add padding around the data points
        print("latitude bounds: ", min(lats), max(lats))
        print("longitude bounds: ", min(lons), max(lons))
        bounds = [
            [min(lats) - padding    , min(lons) - padding],
            [max(lats) + padding, max(lons) + padding]
        ]
    else:
        # Default to world view if no coordinates
        bounds = [[-60, -180], [60, 180]]

    # Calculate center from bounds
    center_lat = (bounds[0][0] + bounds[1][0]) / 2
    center_lon = (bounds[0][1] + bounds[1][1]) / 2

    # Build HTML with embedded JavaScript
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif;
            }}
            #map {{
                position: absolute;
                top: 0;
                bottom: 0;
                width: 100%;
            }}
            .control-panel {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                z-index: 1000;
                max-width: 320px;
                max-height: calc(100vh - 40px);
                overflow-y: auto;
            }}
            .control-panel h2 {{
                margin: 0 0 10px 0;
                font-size: 20px;
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }}
            .control-section {{
                margin-bottom: 12px;
            }}
            .control-section h3 {{
                margin: 0 0 5px 0;
                font-size: 13px;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .control-panel select {{
                width: 100%;
                padding: 10px;
                font-size: 14px;
                border: 2px solid #ddd;
                border-radius: 5px;
                background: white;
                cursor: pointer;
                transition: border-color 0.3s;
            }}
            .control-panel select:hover {{
                border-color: #4CAF50;
            }}
            .control-panel select:focus {{
                outline: none;
                border-color: #4CAF50;
            }}
            .top-words {{
                font-size: 13px;
                color: #666;
                line-height: 1.8;
                background: #f9f9f9;
                padding: 12px;
                border-radius: 5px;
                margin-top: 10px;
            }}
            .top-words strong {{
                color: #333;
                display: block;
                margin-bottom: 5px;
            }}
            .stats {{
                font-size: 11px;
                color: #888;
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid #eee;
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }}
            .stats-item {{
                margin: 0;
            }}

            /* Global Figures panel */
            .global-figures-panel {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                z-index: 1000;
                max-width: 350px;
                max-height: calc(100vh - 40px);
                overflow-y: auto;
            }}
            .global-figures-panel h2 {{
                margin: 0 0 10px 0;
                font-size: 18px;
                color: #333;
                border-bottom: 2px solid #2196F3;
                padding-bottom: 10px;
            }}
            .topic-description {{
                font-size: 12px;
                color: #555;
                line-height: 1.5;
                margin-bottom: 12px;
                padding: 10px;
                background: #f0f7ff;
                border-radius: 6px;
                border-left: 3px solid #2196F3;
            }}
            .top-words-display {{
                margin-bottom: 12px;
            }}
            .top-words-display h4 {{
                margin: 0 0 8px 0;
                font-size: 13px;
                color: #333;
            }}
            .top-words-list {{
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
            }}
            .word-tag {{
                background: #e8f4e8;
                border: 1px solid #c8e6c9;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
                color: #2e7d32;
                cursor: default;
                transition: background 0.2s;
            }}
            .word-tag:hover {{
                background: #c8e6c9;
            }}
            .word-tag[data-translation]:hover {{
                background: #a5d6a7;
            }}

            /* Mini chart containers */
            .mini-chart-section {{
                margin-top: 0;
            }}

            .mini-chart-container {{
                background: #f9f9f9;
                border-radius: 6px;
                padding: 8px;
                margin-bottom: 10px;
            }}

            .mini-chart-container:last-child {{
                margin-bottom: 0;
            }}

            .mini-chart-container h4 {{
                margin: 0 0 6px 0;
                font-size: 13px;
                color: #333;
            }}

            .mini-chart-wrapper {{
                height: 100px;
                position: relative;
            }}

            .mini-chart-wrapper.tall {{
                height: 200px;
            }}

            /* Chart header with title and export button */
            .chart-header-row {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 6px;
            }}

            .chart-header-row h4 {{
                margin: 0;
                font-size: 13px;
                color: #333;
            }}

            .export-btn {{
                background: none;
                border: none;
                cursor: pointer;
                padding: 2px 4px;
                font-size: 14px;
                color: #666;
                border-radius: 3px;
                transition: all 0.2s;
                opacity: 0.6;
            }}

            .export-btn:hover {{
                background: #e0e0e0;
                color: #333;
                opacity: 1;
            }}

            .export-btn:active {{
                background: #d0d0d0;
            }}

            /* Stats info bar - centered at bottom */
            .stats-info-bar {{
                position: absolute;
                bottom: 30px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 12px;
                color: #333;
                background: rgba(200, 200, 200, 0.9);
                padding: 10px 20px;
                border-radius: 6px;
                z-index: 1000;
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }}

            .stats-info-bar .info-item {{
                display: flex;
                align-items: center;
                gap: 4px;
            }}

            .stats-info-bar .info-label {{
                color: #555;
            }}

            .stats-info-bar .info-value {{
                font-weight: bold;
                color: #000;
            }}

            /* Top words clickable section */
            .top-words-clickable {{
                background: #f0f7ff;
                border: 1px solid #cce0ff;
                border-radius: 8px;
                padding: 12px;
                margin-top: 15px;
                cursor: pointer;
                transition: all 0.2s;
            }}

            .top-words-clickable:hover {{
                background: #e0efff;
                border-color: #99c2ff;
            }}

            .top-words-clickable h4 {{
                margin: 0 0 8px 0;
                font-size: 13px;
                color: #333;
                display: flex;
                align-items: center;
                gap: 8px;
            }}

            .top-words-preview {{
                font-size: 12px;
                color: #666;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}

            .click-hint {{
                font-size: 10px;
                color: #999;
                margin-top: 5px;
            }}

            /* Top words table modal */
            .top-words-modal {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 2000;
                justify-content: center;
                align-items: center;
            }}

            .top-words-modal.visible {{
                display: flex;
            }}

            .top-words-table-container {{
                background: white;
                border-radius: 12px;
                padding: 20px;
                max-width: 90%;
                max-height: 80%;
                overflow: auto;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }}

            .top-words-table-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eee;
                gap: 15px;
            }}

            .top-words-table-header h3 {{
                margin: 0;
                color: #333;
                flex: 1;
            }}

            .top-words-table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 12px;
            }}

            .top-words-table th, .top-words-table td {{
                padding: 8px 12px;
                text-align: left;
                border: 1px solid #e0e0e0;
            }}

            .top-words-table th {{
                background: #f5f5f5;
                font-weight: bold;
                color: #333;
                position: sticky;
                top: 0;
            }}

            .top-words-table th.rank-header {{
                background: #e8f5e9;
                width: 60px;
            }}

            .top-words-table td.rank-cell {{
                background: #f9f9f9;
                font-weight: bold;
                color: #666;
            }}

            .top-words-table tr:hover td {{
                background: #f0f7ff;
            }}

            .top-words-table tr:hover td.rank-cell {{
                background: #e3f2fd;
            }}

            /* Language toggle for top words table */
            .language-toggle {{
                display: flex;
                align-items: center;
                gap: 8px;
                margin-right: 20px;
            }}

            .language-toggle-label {{
                font-size: 12px;
                color: #666;
            }}

            .language-toggle-buttons {{
                display: flex;
                border: 1px solid #ddd;
                border-radius: 4px;
                overflow: hidden;
            }}

            .language-toggle-btn {{
                padding: 4px 12px;
                border: none;
                background: #f5f5f5;
                cursor: pointer;
                font-size: 12px;
                font-weight: bold;
                transition: all 0.2s;
            }}

            .language-toggle-btn:first-child {{
                border-right: 1px solid #ddd;
            }}

            .language-toggle-btn.active {{
                background: #4CAF50;
                color: white;
            }}

            .language-toggle-btn:hover:not(.active) {{
                background: #e0e0e0;
            }}

            /* Word cell with translation hover */
            .word-cell {{
                position: relative;
                cursor: default;
            }}

            .word-cell[data-alt]:hover {{
                z-index: 200;
            }}

            .word-cell[data-alt]:hover::after {{
                content: attr(data-alt);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(50, 50, 50, 0.95);
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                white-space: nowrap;
                z-index: 1000;
                pointer-events: none;
            }}

            .word-cell[data-alt]:hover::before {{
                content: '';
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%) translateY(100%);
                border: 5px solid transparent;
                border-top-color: rgba(50, 50, 50, 0.95);
                z-index: 1000;
                pointer-events: none;
            }}

            /* Word impacts bar plot tooltip */
            .top-words-table {{
                position: relative;
                z-index: 1;
            }}

            .top-words-table thead {{
                position: relative;
                z-index: 100;
            }}

            .top-words-table tbody {{
                position: relative;
            }}

            .year-header {{
                position: relative;
                cursor: pointer;
            }}

            .year-header.has-impacts:hover {{
                background: #e3f2fd !important;
            }}

            .word-impacts-tooltip {{
                display: none;
                position: fixed;
                background: #ffffff;
                border: 2px solid #666;
                border-radius: 8px;
                padding: 12px 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.5);
                z-index: 99999;
                min-width: 320px;
                white-space: nowrap;
                pointer-events: none;
            }}

            .word-impacts-tooltip.visible {{
                display: block;
            }}

            .word-impacts-tooltip h4 {{
                margin: 0 0 10px 0;
                font-size: 13px;
                font-weight: bold;
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 8px;
            }}

            .impact-bar-container {{
                margin: 5px 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }}

            .impact-word {{
                width: 180px;
                font-size: 11px;
                text-align: left;
                overflow: hidden;
                text-overflow: ellipsis;
                color: #333;
            }}

            .impact-bar-wrapper {{
                flex: 1;
                height: 14px;
                background: #e8e8e8;
                border-radius: 3px;
                position: relative;
                min-width: 80px;
                overflow: hidden;
            }}

            .impact-bar {{
                height: 100%;
                border-radius: 3px;
                position: absolute;
                top: 0;
                right: 0;
                background: #e57373;
            }}

            .impact-value {{
                width: 55px;
                font-size: 10px;
                color: #666;
                text-align: right;
            }}

            .legend {{
                position: absolute;
                bottom: 30px;
                left: 50%;
                transform: translateX(-50%);
                background: white;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                z-index: 1000;
            }}
            .legend h4 {{
                margin: 0 0 12px 0;
                font-size: 14px;
                color: #333;
            }}
            .legend-item {{
                font-size: 12px;
                margin: 8px 0;
                display: flex;
                align-items: center;
            }}
            .legend-circle {{
                display: inline-block;
                border-radius: 50%;
                background: #ff4444;
                opacity: 0.6;
                margin-right: 10px;
                flex-shrink: 0;
            }}
            .time-indicator {{
                display: none;
            }}
            .bottom-slider-panel {{
                position: absolute;
                bottom: 30px;
                left: 10px;
                background: white;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                z-index: 1000;
                display: flex;
                gap: 30px;
                align-items: flex-start;
            }}

            .slider-section {{
                min-width: 180px;
            }}

            .slider-section h4 {{
                margin: 0 0 10px 0;
                font-size: 14px;
                color: #333;
            }}

            .zoom-slider-container {{
                display: flex;
                flex-direction: column;
            }}
            
            .bottom-time-slider-section {{
                min-width: 250px;
            }}

            .bottom-time-slider-section .time-slider-row {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}

            .bottom-time-slider-section .time-label-display {{
                font-size: 14px;
                font-weight: bold;
                color: #4CAF50;
                margin-top: 8px;
            }}

            .bottom-time-slider-section .speed-control {{
                margin-top: 8px;
                font-size: 12px;
            }}

            .bottom-time-slider-section .speed-control select {{
                padding: 2px 5px;
                font-size: 11px;
            }}
            
            .zoom-slider {{
                width: 100%;
                height: 6px;
                border-radius: 3px;
                background: linear-gradient(to right, #ddd, #4CAF50);
                outline: none;
                -webkit-appearance: none;
            }}
            
            .zoom-slider::-webkit-slider-thumb {{
                -webkit-appearance: none;
                appearance: none;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #4CAF50;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            
            .zoom-slider::-moz-range-thumb {{
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #4CAF50;
                cursor: pointer;
                border: none;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            
            .zoom-slider::-webkit-slider-thumb:hover {{
                background: #45a049;
                transform: scale(1.1);
            }}
            
            .zoom-slider::-moz-range-thumb:hover {{
                background: #45a049;
                transform: scale(1.1);
            }}
            
            .zoom-level-display {{
                text-align: center;
                margin-top: 8px;
                font-size: 12px;
                color: #666;
            }}

            /* Time slider styles */

            .time-slider {{
                flex: 1;
                min-width: 150px;
                height: 6px;
                border-radius: 3px;
                background: linear-gradient(to right, #ddd, #4CAF50);
                outline: none;
                -webkit-appearance: none;
                cursor: pointer;
            }}

            .time-slider::-webkit-slider-thumb {{
                -webkit-appearance: none;
                appearance: none;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #4CAF50;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: transform 0.1s;
            }}

            .time-slider::-moz-range-thumb {{
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #4CAF50;
                cursor: pointer;
                border: none;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}

            .time-slider::-webkit-slider-thumb:hover {{
                background: #45a049;
                transform: scale(1.1);
            }}

            .play-btn {{
                width: 36px;
                height: 36px;
                border-radius: 50%;
                border: none;
                background: #4CAF50;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: background 0.2s, transform 0.1s;
                flex-shrink: 0;
            }}

            .play-btn:hover {{
                background: #45a049;
                transform: scale(1.05);
            }}

            .play-btn.playing {{
                background: #f44336;
            }}

            .play-btn.playing:hover {{
                background: #d32f2f;
            }}

            .time-label-display {{
                text-align: center;
                margin-top: 8px;
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }}

            .speed-control {{
                display: flex;
                align-items: center;
                gap: 8px;
                margin-top: 10px;
                font-size: 12px;
                color: #666;
            }}

            .speed-control select {{
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid #ddd;
                font-size: 12px;
            }}

            /* Checkbox option styling */
            .checkbox-option input[type="checkbox"] {{
                width: 14px;
                height: 14px;
                cursor: pointer;
                accent-color: #4CAF50;
            }}

            /* Location tooltip styling */
            .location-tooltip {{
                background: rgba(50, 50, 50, 0.9);
                border: none;
                border-radius: 4px;
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 6px 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            }}

            /* Cluster bar plot styles for regions */
            .cluster-barplot-marker {{
                background: transparent !important;
                border: none !important;
            }}
            .cluster-barplot {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 4px;
                padding: 4px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                pointer-events: auto;
                cursor: pointer;
            }}
            .cluster-barplot:hover {{
                box-shadow: 0 3px 10px rgba(0,0,0,0.4);
            }}
            .cluster-barplot-container {{
                display: flex;
                align-items: flex-end;
                gap: 2px;
                height: 40px;
            }}
            .cluster-bar {{
                width: 16px;
                min-height: 2px;
                border-radius: 2px 2px 0 0;
                transition: height 0.3s ease;
            }}
            .cluster-barplot-labels {{
                display: flex;
                gap: 2px;
                margin-top: 2px;
            }}
            .cluster-bar-label {{
                width: 16px;
                font-size: 7px;
                text-align: center;
                color: #666;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            /* Cluster tooltip styles */
            .cluster-tooltip {{
                background: rgba(255, 255, 255, 0.95) !important;
                border: 1px solid #ccc !important;
                border-radius: 6px !important;
                padding: 8px 12px !important;
                font-size: 12px !important;
                line-height: 1.5 !important;
                box-shadow: 0 3px 10px rgba(0,0,0,0.2) !important;
            }}
            .cluster-tooltip::before {{
                border-top-color: rgba(255, 255, 255, 0.95) !important;
            }}

            /* Chart modal styles */
            .chart-modal {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 2000;
                justify-content: center;
                align-items: center;
            }}

            .chart-modal.visible {{
                display: flex;
            }}

            .chart-container {{
                background: white;
                border-radius: 12px;
                padding: 20px;
                width: 80%;
                max-width: 900px;
                max-height: 80vh;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                position: relative;
            }}

            .chart-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eee;
            }}

            .chart-title {{
                margin: 0;
                color: #333;
                font-size: 20px;
            }}

            .chart-close-btn {{
                background: #f44336;
                color: white;
                border: none;
                border-radius: 50%;
                width: 32px;
                height: 32px;
                cursor: pointer;
                font-size: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.2s;
            }}

            .chart-close-btn:hover {{
                background: #d32f2f;
            }}

            .chart-wrapper {{
                position: relative;
                height: 400px;
            }}

            .chart-info {{
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #eee;
                font-size: 13px;
                color: #666;
            }}

            .chart-section {{
                margin-bottom: 20px;
            }}

            .chart-section-title {{
                margin: 0;
                font-size: 16px;
                color: #333;
                font-weight: bold;
            }}

            .chart-section-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }}

            .modal-export {{
                font-size: 16px;
                padding: 4px 8px;
            }}

            .legend-hint {{
                font-size: 11px;
                color: #888;
                font-style: italic;
                margin: 0 0 10px 0;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>

        <!-- Chart Modal -->
        <div id="chart-modal" class="chart-modal" onclick="closeChartModal(event)">
            <div class="chart-container" onclick="event.stopPropagation()" style="max-height: 90vh; overflow-y: auto;">
                <div class="chart-header">
                    <h3 id="chart-title" class="chart-title">Location Name</h3>
                    <button class="chart-close-btn" onclick="closeChartModal()">&times;</button>
                </div>
                <!-- Number of Documents Over Time Chart -->
                <div class="chart-section">
                    <div class="chart-section-header">
                        <h4 class="chart-section-title">Number of Documents</h4>
                        <button class="export-btn modal-export" onclick="exportChart('doc-count-chart', 'doc-count')" title="Export chart">⬇</button>
                    </div>
                    <div class="chart-wrapper" style="height: 200px;">
                        <canvas id="doc-count-chart"></canvas>
                    </div>
                </div>
                <!-- Topic/Cluster Prevalence Chart -->
                <div class="chart-section">
                    <div class="chart-section-header">
                        <h4 id="prevalence-section-title" class="chart-section-title">Topic Prevalence</h4>
                        <button class="export-btn modal-export" onclick="exportChart('location-chart', 'prevalence')" title="Export chart">⬇</button>
                    </div>
                    <p class="legend-hint">Click on legend items to show/hide curves</p>
                    <div class="chart-wrapper">
                        <canvas id="location-chart"></canvas>
                    </div>
                </div>
                <div class="chart-info" id="chart-info">
                    Click on a location to see topic prevalence over time.
                </div>
            </div>
        </div>

        <!-- Top Words Table Modal -->
        <div id="top-words-modal" class="top-words-modal" onclick="closeTopWordsModal(event)">
            <div class="top-words-table-container" onclick="event.stopPropagation()">
                <div class="top-words-table-header">
                    <h3 id="top-words-table-title">Top Words Over Time</h3>
                    <div class="language-toggle">
                        <span class="language-toggle-label">Language:</span>
                        <div class="language-toggle-buttons">
                            <button class="language-toggle-btn active" id="lang-es-btn" onclick="setTableLanguage('es')">ES</button>
                            <button class="language-toggle-btn" id="lang-en-btn" onclick="setTableLanguage('en')">EN</button>
                        </div>
                    </div>
                    <button class="export-btn" onclick="exportTopWordsCSV()" title="Download as CSV">⬇</button>
                    <button class="chart-close-btn" onclick="closeTopWordsModal()">&times;</button>
                </div>
                <div id="top-words-table-wrapper">
                    <!-- Table will be inserted here by JavaScript -->
                </div>
            </div>
        </div>

        <div class="time-indicator" id="time-indicator">
            Loading...
        </div>

        <div class="control-panel">
            <h2>🗺️ Controls</h2>

    """

    # Check if cluster data is available
    has_clusters = chart_data and chart_data.get('n_clusters', 0) > 0

    html_content += """

            <div class="control-section">
                <h3>📊 View Mode</h3>
                <select id="view-mode-selector" onchange="onViewModeChange()">
                    <option value="topics">Topics</option>
    """

    if has_clusters:
        html_content += '                    <option value="clusters">Clusters</option>\n'

    html_content += """
                </select>
                <div class="checkbox-option" style="margin-top: 12px;">
                    <label style="display: flex; align-items: center; gap: 10px; cursor: pointer; font-size: 14px; color: #333; padding: 8px 0;">
                        <input type="checkbox" id="show-barplots-checkbox" onchange="updateMap()" style="width: 18px; height: 18px; cursor: pointer;">
                        Show Cluster Prevalences in Map
                    </label>
                </div>
            </div>

            <div class="control-section" id="topic-section">
                <h3>🏷️ Topic</h3>
                <select id="topic-selector" onchange="updateMap(); updateMiniCharts();">
                    <option value="corpus" selected>Corpus</option>
    """

    # Add topic options using topic titles with explanations as tooltips
    for topic_idx in range(n_topics):
        title = topic_titles.get(topic_idx, f"Topic {topic_idx}")
        explanation = topic_explanations.get(topic_idx, "")
        # Escape quotes in explanation for HTML attribute
        explanation_escaped = explanation.replace('"', '&quot;')
        html_content += f'                    <option value="{topic_idx}" title="{explanation_escaped}">{title}</option>\n'

    html_content += """
                </select>
            </div>
    """

    # Add cluster selector (hidden by default, shown when cluster mode is selected)
    if has_clusters:
        cluster_names = chart_data.get('cluster_names', [])
        cluster_mapping = chart_data.get('cluster_mapping', {})
        cluster_titles = chart_data.get('cluster_titles', {})
        cluster_descriptions = chart_data.get('cluster_descriptions', {})
        html_content += """
            <div class="control-section" id="cluster-section" style="display: none;">
                <h3>🗂️ Cluster</h3>
                <select id="cluster-selector" onchange="updateMap(); updateMiniCharts(); updateGlobalFigures();">
        """
        for cluster_name in cluster_names:
            topic_indices = cluster_mapping.get(cluster_name, [])
            topics_str = ', '.join(str(i+1) for i in topic_indices)
            # Get short title and construct display title with "C1: <title>" format
            short_title = cluster_titles.get(cluster_name, '')
            display_title = short_title if short_title else cluster_name
            # Add description as tooltip
            description = cluster_descriptions.get(cluster_name, '')
            description_escaped = description.replace('"', '&quot;')
            html_content += f'                    <option value="{cluster_name}" title="{description_escaped}">{display_title} (Topics: {topics_str})</option>\n'
        html_content += """
                </select>
            </div>
        """

    # Add granularity selector if multiple granularity levels are available
    if has_granularity and len(granularity_levels) > 1:
        granularity_labels = {
            'region': 'Region',
            'province': 'Province',
            'comuna': 'Comuna'
        }
        html_content += """
            <div class="control-section">
                <h3>📍 Location Granularity</h3>
                <select id="granularity-selector" onchange="updateMap(); updateMiniCharts();">
        """
        for granularity in granularity_levels:
            label = granularity_labels.get(granularity, granularity.upper())
            html_content += f'                    <option value="{granularity}">{label}</option>\n'
        html_content += """
                </select>
            </div>
        """

    # Add zoom and time sliders
    html_content += """
            <div class="control-section">
                <h3>🔍 Zoom Level</h3>
                <input type="range" min="4" max="12" value="4" class="zoom-slider" id="zoom-slider">
                <div class="zoom-level-display">
                    Level: <span id="zoom-level">4</span>
                </div>
            </div>

            <div class="control-section bottom-time-slider-section">
                <h3>⏰ Time Period</h3>
                <div class="time-slider-row">
                    <button id="play-btn" class="play-btn" onclick="togglePlay()" title="Play/Pause">
                        ▶
                    </button>
                    <input type="range" id="time-slider" class="time-slider"
                           min="-1" max=\"""" + str(len(time_labels) - 1) + """\" value="-1"
                           oninput="onTimeSliderChange()">
                </div>
                <div id="time-label" class="time-label-display">All Periods</div>
                <div class="speed-control">
                    <label>Speed:</label>
                    <select id="speed-selector" onchange="updatePlaySpeed()">
                        <option value="2000">Slow (2s)</option>
                        <option value="1000" selected>Normal (1s)</option>
                        <option value="500">Fast (0.5s)</option>
                        <option value="250">Very Fast (0.25s)</option>
                    </select>
                </div>
            </div>

            <div class="control-section" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                <h3 id="legend-title">📊 Document Count</h3>
                <div style="display: flex; flex-direction: column; margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 5px;">
                        <span>Low</span>
                        <span style="color: #999;">Medium</span>
                        <span>High</span>
                    </div>
                    <div style="height: 20px; background: linear-gradient(to right, #ffeb3b, #ff9800 50%, #f44336); border-radius: 4px;"></div>
                </div>
                <div style="font-size: 10px; color: #666; margin-top: 8px;" id="legend-subtitle">
                </div>
            </div>
        </div>

        <!-- Stats Info Bar - Centered at Bottom -->
        <div class="stats-info-bar">
            <div class="info-item">
                <span class="info-label">Time Period:</span>
                <span class="info-value" id="stats-time-period">All Periods</span>
            </div>
            <div class="info-item">
                <span class="info-label">Documents:</span>
                <span class="info-value" id="document-count">-</span>
            </div>
            <div class="info-item">
                <span class="info-label">Locations:</span>
                <span class="info-value" id="location-count">-</span>
            </div>
        </div>

        <!-- Global Figures Panel -->
        <div class="global-figures-panel">
            <h2 id="global-panel-title">📊 Corpus</h2>

            <!-- Cluster Topic Composition (shown only in cluster mode) -->
            <div id="cluster-topics-container" style="display: none; font-size: 12px; color: #666; margin-bottom: 8px; font-style: italic;"></div>

            <!-- Topic Description -->
            <div id="topic-description-container" class="topic-description" style="display: none;"></div>

            <!-- Top Words Section -->
            <div class="top-words-display">
                <h4 onclick="showTopWordsTable()" style="cursor: pointer;">🔤 Top Words <span style="font-size: 11px; color: #666;">(click for full table)</span></h4>
                <div id="top-words-tags" class="top-words-list"></div>
            </div>

            <!-- Charts Section -->
            <div class="mini-chart-section">
                <div class="mini-chart-container">
                    <div class="chart-header-row">
                        <h4 id="prevalence-chart-title">📈 Number of Documents</h4>
                        <button class="export-btn" onclick="exportChart('prevalence-mini-chart', 'prevalence')" title="Export chart">⬇</button>
                    </div>
                    <div class="mini-chart-wrapper tall">
                        <canvas id="prevalence-mini-chart"></canvas>
                    </div>
                </div>

                <div class="mini-chart-container">
                    <div class="chart-header-row">
                        <h4 id="location-counts-chart-title">📍 Number of Documents in Locations</h4>
                        <button class="export-btn" onclick="exportChart('location-counts-mini-chart', 'location-counts')" title="Export chart">⬇</button>
                    </div>
                    <div class="mini-chart-wrapper tall">
                        <canvas id="location-counts-mini-chart"></canvas>
                    </div>
                </div>

                <div class="mini-chart-container">
                    <div class="chart-header-row">
                        <h4 id="gini-chart-title">📊 Gini Coefficient (Documents in Locations)</h4>
                        <button class="export-btn" onclick="exportChart('gini-mini-chart', 'gini')" title="Export chart">⬇</button>
                    </div>
                    <div class="mini-chart-wrapper">
                        <canvas id="gini-mini-chart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        <script>
    """

    # Embed data as JavaScript
    import json

    # Prepare all granularity data for embedding
    all_granularity_data_json = {}
    for granularity, time_data in all_granularity_data.items():
        all_granularity_data_json[granularity] = [
            {k: v for k, v in td.items() if k != 'time_label'}
            for td in time_data
        ]

    html_content += f"        const allGranularityData = {json.dumps(all_granularity_data_json)};\n"
    html_content += f"        const granularityLevels = {json.dumps(granularity_levels)};\n"
    html_content += f"        const hasGranularity = {'true' if has_granularity and len(granularity_levels) > 1 else 'false'};\n"
    html_content += f"        const timeLabels = {json.dumps(time_labels)};\n"
    html_content += f"        const nTopics = {n_topics};\n"
    html_content += f"        const topicTitles = {json.dumps(topic_titles)};\n"
    html_content += f"        const topicExplanations = {json.dumps(topic_explanations)};\n"
    html_content += f"        const wordTranslations = {json.dumps(word_translations)};\n"
    html_content += f"        const hasTranslations = {'true' if word_translations else 'false'};\n"
    html_content += f"        const wordImpacts = {json.dumps(word_impacts)};\n"
    html_content += f"        const hasWordImpacts = {'true' if word_impacts else 'false'};\n"
    html_content += f"        const useGeoJSON = {'true' if use_geojson else 'false'};\n"

    # Embed chart data if provided
    if chart_data:
        html_content += f"        const chartData = {json.dumps(chart_data)};\n"
    else:
        html_content += "        const chartData = null;\n"

    # Embed GeoJSON data if provided
    if use_geojson:
        html_content += f"        const geoJSONData = {json.dumps(geojson_data)};\n"
        html_content += f"        const locationNameProps = {json.dumps(name_props)};\n"
    else:
        html_content += "        const geoJSONData = null;\n"
        html_content += "        const locationNameProps = null;\n"

    # Add JavaScript logic - embed dynamic bounds and center
    html_content += f"""
        // Initialize map with dynamic bounds
        const mapBounds = [
            [{bounds[0][0]}, {bounds[0][1]}],  // Southwest corner (bottom-left)
            [{bounds[1][0]}, {bounds[1][1]}]   // Northeast corner (top-right)
        ];

        const map = L.map('map', {{
            center: [{center_lat}, {center_lon}],
            zoom: 4,
            maxBounds: mapBounds,
            maxBoundsViscosity: 1.0,  // Makes the bounds "hard" - can't drag outside
            zoomControl: false
        }});"""

    html_content += """
        L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            attribution: '© OpenStreetMap contributors © CARTO',
            minZoom: 4,
            maxZoom: 12
        }).addTo(map);


        let mapLayers = [];

        // Play/animation state
        let isPlaying = false;
        let playInterval = null;
        let playSpeed = 1000; // milliseconds between frames

        // Time slider functions
        function onTimeSliderChange() {
            const slider = document.getElementById('time-slider');
            const timeIdx = parseInt(slider.value);
            if (timeIdx === -1) {
                document.getElementById('time-label').textContent = 'All Periods';
            } else {
                document.getElementById('time-label').textContent = timeLabels[timeIdx];
            }
            updateMap();
            updateTopWordsTags();
        }

        function togglePlay() {
            const playBtn = document.getElementById('play-btn');
            if (isPlaying) {
                // Stop playing
                isPlaying = false;
                clearInterval(playInterval);
                playInterval = null;
                playBtn.textContent = '▶';
                playBtn.classList.remove('playing');
            } else {
                // Start playing
                isPlaying = true;
                playBtn.textContent = '⏸';
                playBtn.classList.add('playing');
                playInterval = setInterval(advanceTime, playSpeed);
            }
        }

        function advanceTime() {
            const slider = document.getElementById('time-slider');
            let currentIdx = parseInt(slider.value);
            const maxIdx = parseInt(slider.max);

            currentIdx++;
            if (currentIdx > maxIdx) {
                currentIdx = 0; // Loop back to start (skip -1 "All Periods" during playback)
            }

            slider.value = currentIdx;
            document.getElementById('time-label').textContent = timeLabels[currentIdx];
            updateMap();
        }

        function updatePlaySpeed() {
            const speedSelector = document.getElementById('speed-selector');
            playSpeed = parseInt(speedSelector.value);

            // If currently playing, restart with new speed
            if (isPlaying) {
                clearInterval(playInterval);
                playInterval = setInterval(advanceTime, playSpeed);
            }
        }

        // View mode (topics vs clusters)
        function onViewModeChange() {
            const viewMode = document.getElementById('view-mode-selector').value;
            const topicSection = document.getElementById('topic-section');
            const clusterSection = document.getElementById('cluster-section');
            const topicSelector = document.getElementById('topic-selector');

            if (viewMode === 'clusters') {
                if (topicSection) topicSection.style.display = 'none';
                if (clusterSection) clusterSection.style.display = 'block';

                // Switch away from Corpus mode when entering Clusters mode
                // This prevents conflicts between Corpus and Clusters views
                if (topicSelector && topicSelector.value === 'corpus') {
                    topicSelector.value = '0';  // Select first topic
                }
            } else {
                if (topicSection) topicSection.style.display = 'block';
                if (clusterSection) clusterSection.style.display = 'none';
            }

            updateMap();
            updateMiniCharts();
            updateGlobalFigures();
        }

        function getCurrentViewMode() {
            const selector = document.getElementById('view-mode-selector');
            return selector ? selector.value : 'topics';
        }

        function getSelectedCluster() {
            const selector = document.getElementById('cluster-selector');
            return selector ? selector.value : null;
        }

        // Chart functionality
        let locationChart = null;
        let docCountChart = null;
        const topicColors = [
            '#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336',
            '#00BCD4', '#FFEB3B', '#795548', '#607D8B', '#E91E63'
        ];

        function showLocationChart(locationName) {
            // Get current granularity
            let currentGranularity = granularityLevels[0];
            const granularitySelector = document.getElementById('granularity-selector');
            if (granularitySelector) {
                currentGranularity = granularitySelector.value;
            }

            const viewMode = getCurrentViewMode();
            const clusterNames = (chartData && chartData.cluster_names) ? chartData.cluster_names : [];
            const hasClusterData = clusterNames.length > 0;
            const chartTimeLabels = timeLabels;
            const numTopics = nTopics;

            // Get embedded data for this granularity
            const allTimeData = allGranularityData[currentGranularity];
            if (!allTimeData) {
                alert('No data available for this granularity level.');
                return;
            }

            // Check if we have pre-computed chart data OR use embedded data
            let usePrecomputedData = chartData && chartData.prevalence_data &&
                                      chartData.prevalence_data[currentGranularity] &&
                                      chartData.prevalence_data[currentGranularity][locationName];

            let locationPrevalence = null;
            if (usePrecomputedData) {
                locationPrevalence = chartData.prevalence_data[currentGranularity][locationName];
            }

            // Build datasets from embedded data or pre-computed data
            const datasets = [];
            const docCountsPerPeriod = [];
            let totalDocs = 0;

            if (viewMode === 'clusters' && hasClusterData && chartData.cluster_mapping) {
                // Show cluster prevalence - compute from embedded data
                for (let clusterIdx = 0; clusterIdx < clusterNames.length; clusterIdx++) {
                    const clusterName = clusterNames[clusterIdx];
                    const clusterTopics = chartData.cluster_mapping[clusterName] || [];
                    const clusterData = [];
                    let hasData = false;

                    for (let timeIdx = 0; timeIdx < chartTimeLabels.length; timeIdx++) {
                        const periodData = allTimeData[timeIdx];
                        if (periodData && periodData.location_data && periodData.location_data[locationName]) {
                            const locData = periodData.location_data[locationName];
                            // Compute cluster prevalence by summing topic prevalences
                            let clusterPrevalence = 0;
                            for (const topicIdx of clusterTopics) {
                                if (locData.topics && locData.topics[topicIdx] !== undefined) {
                                    clusterPrevalence += locData.topics[topicIdx];
                                }
                            }
                            clusterData.push(clusterPrevalence);
                            hasData = true;

                            // Collect doc counts for first cluster only
                            if (clusterIdx === 0) {
                                const docCount = locData.doc_count || 0;
                                docCountsPerPeriod.push(docCount);
                                totalDocs += docCount;
                            }
                        } else {
                            clusterData.push(null);
                            if (clusterIdx === 0) docCountsPerPeriod.push(0);
                        }
                    }

                    if (hasData) {
                        datasets.push({
                            label: `${clusterName} (Topics: ${clusterTopics.join(', ')})`,
                            data: clusterData,
                            borderColor: topicColors[clusterIdx % topicColors.length],
                            backgroundColor: topicColors[clusterIdx % topicColors.length] + '33',
                            fill: false,
                            tension: 0.3,
                            pointRadius: 5,
                            pointHoverRadius: 8,
                            borderWidth: 3
                        });
                    }
                }
            } else {
                // Show topic prevalence from embedded data
                for (let topicIdx = 0; topicIdx < numTopics; topicIdx++) {
                    const topicData = [];
                    let hasData = false;

                    for (let timeIdx = 0; timeIdx < chartTimeLabels.length; timeIdx++) {
                        const periodData = allTimeData[timeIdx];
                        if (periodData && periodData.location_data && periodData.location_data[locationName]) {
                            const locData = periodData.location_data[locationName];
                            if (locData.topics && locData.topics[topicIdx] !== undefined) {
                                topicData.push(locData.topics[topicIdx]);
                                hasData = true;
                            } else {
                                topicData.push(null);
                            }

                            // Collect doc counts for first topic only
                            if (topicIdx === 0) {
                                const docCount = locData.doc_count || 0;
                                docCountsPerPeriod.push(docCount);
                                totalDocs += docCount;
                            }
                        } else {
                            topicData.push(null);
                            if (topicIdx === 0) docCountsPerPeriod.push(0);
                        }
                    }

                    if (hasData) {
                        // Use topic title from embedded data
                        const topicTitle = topicTitles[topicIdx] || topicTitles[topicIdx.toString()] || `Topic ${topicIdx}`;
                        datasets.push({
                            label: topicTitle,
                            data: topicData,
                            borderColor: topicColors[topicIdx % topicColors.length],
                            backgroundColor: topicColors[topicIdx % topicColors.length] + '33',
                            fill: false,
                            tension: 0.3,
                            pointRadius: 5,
                            pointHoverRadius: 8
                        });
                    }
                }
            }

            // Check if we have any data
            if (datasets.length === 0) {
                alert('No data available for this location: ' + locationName);
                return;
            }

            // Update modal title to just the location name
            const chartType = viewMode === 'clusters' ? 'Cluster' : 'Topic';
            document.getElementById('chart-title').textContent = locationName;

            // Update prevalence section title
            const prevalenceSectionTitle = document.getElementById('prevalence-section-title');
            if (prevalenceSectionTitle) {
                prevalenceSectionTitle.textContent = chartType + ' Prevalence';
            }

            // Format granularity for display (uppercase)
            const granularityDisplay = currentGranularity.charAt(0).toUpperCase() + currentGranularity.slice(1);
            document.getElementById('chart-info').textContent =
                `Total Number of Documents Across All Time Periods: ${totalDocs} | Granularity Level: ${granularityDisplay} | View: ${chartType}s`;

            // Destroy existing charts if any
            if (docCountChart) {
                docCountChart.destroy();
            }
            if (locationChart) {
                locationChart.destroy();
            }

            // Show modal FIRST so Chart.js can properly calculate dimensions
            // (Charts created while container is display:none will have sizing issues)
            document.getElementById('chart-modal').classList.add('visible');

            // Helper function to get top words for a location at a specific time period
            // Returns array of objects with {word, translation} for each top word
            function getLocationTopWords(timeIdx, locName, topN = 10) {
                const periodData = allTimeData[timeIdx];
                if (!periodData || !periodData.top_words || !periodData.location_data) {
                    return [];
                }
                const locData = periodData.location_data[locName];
                if (!locData || !locData.topics) {
                    return [];
                }

                // Collect words from topics weighted by topic prevalence
                const wordScores = {};
                const topicPrevalences = locData.topics;

                for (let topicIdx = 0; topicIdx < topicPrevalences.length; topicIdx++) {
                    const prevalence = topicPrevalences[topicIdx] || 0;
                    if (prevalence > 0 && periodData.top_words[topicIdx]) {
                        const topicWords = periodData.top_words[topicIdx];
                        // Weight words by their position (higher position = higher weight) and topic prevalence
                        for (let wordIdx = 0; wordIdx < topicWords.length; wordIdx++) {
                            const word = topicWords[wordIdx];
                            const positionWeight = (topicWords.length - wordIdx) / topicWords.length;
                            const score = prevalence * positionWeight;
                            wordScores[word] = (wordScores[word] || 0) + score;
                        }
                    }
                }

                // Sort by score and return top N with translations
                const sortedWords = Object.entries(wordScores)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, topN)
                    .map(entry => {
                        const word = entry[0];
                        const lowerWord = word.toLowerCase();
                        const translation = hasTranslations ? (wordTranslations[lowerWord] || '') : '';
                        return { word, translation };
                    });

                return sortedWords;
            }

            // Create document count chart
            const docCtx = document.getElementById('doc-count-chart').getContext('2d');
            docCountChart = new Chart(docCtx, {
                type: 'bar',
                data: {
                    labels: chartTimeLabels,
                    datasets: [{
                        label: 'Number of Documents',
                        data: docCountsPerPeriod,
                        backgroundColor: '#2196F3',
                        borderColor: '#1976D2',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                afterBody: function(context) {
                                    const timeIdx = context[0].dataIndex;
                                    const topWords = getLocationTopWords(timeIdx, locationName, 5);
                                    if (topWords.length > 0) {
                                        const lines = ['', 'Top words:'];
                                        topWords.forEach(item => {
                                            if (item.translation) {
                                                lines.push(`  ${item.word} (${item.translation})`);
                                            } else {
                                                lines.push(`  ${item.word}`);
                                            }
                                        });
                                        return lines;
                                    }
                                    return [];
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Document Count'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time Period'
                            }
                        }
                    }
                }
            });

            // Create prevalence chart
            const ctx = document.getElementById('location-chart').getContext('2d');
            locationChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartTimeLabels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                usePointStyle: true,
                                padding: 15
                            },
                            onClick: function(e, legendItem, legend) {
                                // Default click behavior - toggle visibility
                                const index = legendItem.datasetIndex;
                                const ci = legend.chart;
                                if (ci.isDatasetVisible(index)) {
                                    ci.hide(index);
                                    legendItem.hidden = true;
                                } else {
                                    ci.show(index);
                                    legendItem.hidden = false;
                                }
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let value = context.parsed.y;
                                    if (value !== null) {
                                        return context.dataset.label + ': ' + value.toFixed(4);
                                    }
                                    return context.dataset.label + ': No data';
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: chartType + ' Prevalence'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time Period'
                            }
                        }
                    }
                }
            });
        }

        function closeChartModal(event) {
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('chart-modal').classList.remove('visible');
        }

        // Export chart as PNG image
        function exportChart(canvasId, baseFilename) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) {
                alert('Chart not found');
                return;
            }

            // Get current context for filename
            const viewMode = getCurrentViewMode();
            const topicSelectorValue = document.getElementById('topic-selector').value;
            const isCorpusMode = topicSelectorValue === 'corpus';
            const timeIdx = parseInt(document.getElementById('time-slider').value);
            const timeLabel = timeIdx === -1 ? 'all-periods' : timeLabels[timeIdx];

            let contextName = '';
            if (isCorpusMode) {
                contextName = 'corpus';
            } else if (viewMode === 'clusters') {
                const selectedCluster = getSelectedCluster();
                contextName = selectedCluster || 'clusters';
            } else {
                const topicIdx = parseInt(topicSelectorValue);
                contextName = 'topic-' + topicIdx;
            }

            // Create filename
            const filename = baseFilename + '_' + contextName + '_' + timeLabel + '.png';

            // Create a temporary link and trigger download
            const link = document.createElement('a');
            link.download = filename;
            link.href = canvas.toDataURL('image/png');
            link.click();
        }

        // Export top words table as CSV
        function exportTopWordsCSV() {
            const table = document.querySelector('.top-words-table');
            if (!table) {
                alert('Table not found');
                return;
            }

            // Get current language setting
            const currentLang = document.getElementById('lang-en-btn').classList.contains('active') ? 'en' : 'es';

            // Build CSV content
            let csvContent = '';
            const rows = table.querySelectorAll('tr');

            rows.forEach((row, rowIdx) => {
                const cells = row.querySelectorAll('th, td');
                const rowData = [];

                cells.forEach(cell => {
                    // Get the appropriate text based on language
                    let cellText = '';
                    if (cell.classList.contains('word-cell')) {
                        // Word cell - get based on current language
                        const esWord = cell.getAttribute('data-es') || '';
                        const enWord = cell.getAttribute('data-en') || '';
                        cellText = currentLang === 'en' ? enWord : esWord;
                    } else {
                        cellText = cell.textContent.trim();
                    }

                    // Escape quotes and wrap in quotes if contains comma, quote, or newline
                    if (cellText.includes(',') || cellText.includes('"') || cellText.includes('\\n')) {
                        cellText = '"' + cellText.replace(/"/g, '""') + '"';
                    }
                    rowData.push(cellText);
                });

                csvContent += rowData.join(',') + '\\n';
            });

            // Get context for filename
            const title = document.getElementById('top-words-table-title').textContent;
            let filename = 'top_words';

            const viewMode = getCurrentViewMode();
            const topicSelectorValue = document.getElementById('topic-selector').value;
            const isCorpusMode = topicSelectorValue === 'corpus';

            if (isCorpusMode) {
                filename = 'top_words_corpus';
            } else if (viewMode === 'clusters') {
                const selectedCluster = getSelectedCluster();
                filename = 'top_words_' + (selectedCluster || 'cluster').replace(/[^a-z0-9]/gi, '_');
            } else {
                const topicIdx = parseInt(topicSelectorValue);
                filename = 'top_words_topic_' + topicIdx;
            }

            filename += '_' + currentLang + '.csv';

            // Create download link
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.click();
            URL.revokeObjectURL(link.href);
        }

        // Top words table modal functions
        function showTopWordsTable() {
            if (!chartData || !chartData.top_words_by_chunk) {
                alert('Top words data not available');
                return;
            }

            const viewMode = getCurrentViewMode();
            const topicSelectorValue = document.getElementById('topic-selector').value;
            const isCorpusMode = topicSelectorValue === 'corpus';
            const topicIdx = isCorpusMode ? -1 : parseInt(topicSelectorValue);
            const selectedCluster = getSelectedCluster();

            let title, topicsToShow;
            if (viewMode === 'clusters' && chartData.cluster_mapping && selectedCluster) {
                const clusterTitle = chartData.cluster_titles[selectedCluster] || selectedCluster;
                title = `Top Words Over Time - ${clusterTitle}`;
                topicsToShow = chartData.cluster_mapping[selectedCluster] || [];
            } else if (isCorpusMode) {
                title = `Top Words Over Time - Corpus`;
                topicsToShow = Array.from({length: nTopics}, (_, i) => i);
            } else {
                const topicTitle = topicTitles[topicIdx] || topicTitles[topicIdx.toString()] || `Topic ${topicIdx}`;
                title = `Top Words Over Time - ${topicTitle}`;
                topicsToShow = [topicIdx];
            }

            document.getElementById('top-words-table-title').textContent = title;

            // Check if we're in single topic mode (not corpus, not clusters)
            const isSingleTopicMode = !isCorpusMode && topicsToShow.length === 1;
            const currentTopicIdx = isSingleTopicMode ? topicsToShow[0] : null;

            // Helper function to create word impacts tooltip HTML content (without wrapper)
            function createWordImpactsContent(topicIdx, year) {
                if (!hasWordImpacts || !wordImpacts[topicIdx] || !wordImpacts[topicIdx][year]) {
                    return '';
                }
                const impactData = wordImpacts[topicIdx][year];
                const words = impactData.words || [];
                const impacts = impactData.impacts || [];

                if (words.length === 0) return '';

                // Find max absolute impact for scaling (all values are negative)
                const maxAbsImpact = Math.max(...impacts.map(v => Math.abs(v)));
                if (maxAbsImpact === 0) return '';

                let tooltipHtml = `<h4>Word Impacts (${year})</h4>`;

                for (let i = 0; i < words.length && i < 5; i++) {
                    const word = words[i];
                    const impact = impacts[i];
                    // Bar width as percentage (bars grow from right to left for negative values)
                    const barWidth = Math.abs(impact) / maxAbsImpact * 100;

                    // Get translation if available and add in parentheses
                    const lowerWord = word.toLowerCase();
                    const translation = hasTranslations ? (wordTranslations[lowerWord] || '') : '';
                    const displayWord = translation ? `${word} (${translation})` : word;

                    tooltipHtml += `<div class="impact-bar-container">`;
                    tooltipHtml += `<div class="impact-word">${displayWord}</div>`;
                    tooltipHtml += `<div class="impact-bar-wrapper">`;
                    tooltipHtml += `<div class="impact-bar" style="width: ${barWidth}%;"></div>`;
                    tooltipHtml += `</div>`;
                    tooltipHtml += `<div class="impact-value">${impact.toFixed(4)}</div>`;
                    tooltipHtml += `</div>`;
                }

                return tooltipHtml;
            }

            // Store tooltip data for each year header
            const yearTooltipData = {};

            // Build the table
            let tableHtml = '<table class="top-words-table"><thead><tr>';
            tableHtml += '<th class="rank-header">Rank</th>';

            // Add time labels as column headers with word impacts data attribute for single topic mode
            for (let i = 0; i < timeLabels.length; i++) {
                const timeLabel = timeLabels[i];
                if (isSingleTopicMode && hasWordImpacts && i > 0) {
                    // Check if we have impacts for this topic and year (skip first year)
                    const tooltipContent = createWordImpactsContent(currentTopicIdx, timeLabel);
                    if (tooltipContent) {
                        yearTooltipData[timeLabel] = tooltipContent;
                        tableHtml += `<th class="year-header has-impacts" data-year="${timeLabel}">${timeLabel}</th>`;
                    } else {
                        tableHtml += `<th>${timeLabel}</th>`;
                    }
                } else {
                    tableHtml += `<th>${timeLabel}</th>`;
                }
            }
            tableHtml += '</tr></thead><tbody>';

            // For corpus mode, pre-compute deduplicated top words per time period
            const corpusTopWordsByTime = {};
            if (isCorpusMode) {
                for (const timeLabel of timeLabels) {
                    const chunkWords = chartData.top_words_by_chunk[timeLabel];
                    if (chunkWords) {
                        const wordFreq = {};
                        for (let tIdx = 0; tIdx < nTopics; tIdx++) {
                            if (chunkWords[tIdx]) {
                                chunkWords[tIdx].forEach((w, idx) => {
                                    // Weight by position (higher rank = higher weight)
                                    wordFreq[w] = (wordFreq[w] || 0) + (10 - Math.min(idx, 9));
                                });
                            }
                        }
                        corpusTopWordsByTime[timeLabel] = Object.entries(wordFreq)
                            .sort((a, b) => b[1] - a[1])
                            .map(([w, _]) => w);
                    }
                }
            }

            // For clusters, pre-compute aggregated top words per time period
            const clusterTopWordsByTime = {};
            if (viewMode === 'clusters' && topicsToShow.length > 1) {
                for (const timeLabel of timeLabels) {
                    const chunkWords = chartData.top_words_by_chunk[timeLabel];
                    if (chunkWords) {
                        const wordFreq = {};
                        for (const tIdx of topicsToShow) {
                            if (chunkWords[tIdx]) {
                                chunkWords[tIdx].forEach((w, idx) => {
                                    // Weight by position (higher rank = higher weight)
                                    wordFreq[w] = (wordFreq[w] || 0) + (10 - Math.min(idx, 9));
                                });
                            }
                        }
                        clusterTopWordsByTime[timeLabel] = Object.entries(wordFreq)
                            .sort((a, b) => b[1] - a[1])
                            .map(([w, _]) => w);
                    }
                }
            }

            // Helper function to create a word cell with translation
            function createWordCell(word) {
                if (!word || word === '-') {
                    return '<td>-</td>';
                }
                const lowerWord = word.toLowerCase();
                const translation = hasTranslations ? (wordTranslations[lowerWord] || '') : '';
                if (translation) {
                    // data-es = Spanish word, data-en = English translation
                    // Initially show Spanish (ES mode), hover shows English
                    return `<td class="word-cell" data-es="${word}" data-en="${translation}" data-alt="${translation}">${word}</td>`;
                } else {
                    return `<td class="word-cell" data-es="${word}" data-en="${word}">${word}</td>`;
                }
            }

            // For clusters, we'll show aggregated top words for the whole cluster
            // For single topics, show the top 10 words
            const numWords = 10;

            for (let rank = 0; rank < numWords; rank++) {
                tableHtml += `<tr><td class="rank-cell">${rank + 1}</td>`;

                for (const timeLabel of timeLabels) {
                    const chunkWords = chartData.top_words_by_chunk[timeLabel];
                    if (chunkWords) {
                        if (isCorpusMode) {
                            // For corpus mode, show deduplicated top words
                            const corpusWords = corpusTopWordsByTime[timeLabel] || [];
                            tableHtml += createWordCell(corpusWords[rank]);
                        } else if (topicsToShow.length > 1) {
                            // For clusters, show aggregated top words for the entire cluster
                            const clusterWords = clusterTopWordsByTime[timeLabel] || [];
                            tableHtml += createWordCell(clusterWords[rank]);
                        } else {
                            // Single topic
                            const tIdx = topicsToShow[0];
                            const word = chunkWords[tIdx] ? chunkWords[tIdx][rank] : '-';
                            tableHtml += createWordCell(word);
                        }
                    } else {
                        tableHtml += '<td>-</td>';
                    }
                }
                tableHtml += '</tr>';
            }

            tableHtml += '</tbody></table>';
            document.getElementById('top-words-table-wrapper').innerHTML = tableHtml;
            document.getElementById('top-words-modal').classList.add('visible');

            // Reset language toggle to ES when opening
            setTableLanguage('es');

            // Set up word impacts tooltip for year headers
            if (isSingleTopicMode && hasWordImpacts && Object.keys(yearTooltipData).length > 0) {
                // Create or get the floating tooltip container
                let impactTooltip = document.getElementById('word-impacts-floating-tooltip');
                if (!impactTooltip) {
                    impactTooltip = document.createElement('div');
                    impactTooltip.id = 'word-impacts-floating-tooltip';
                    impactTooltip.className = 'word-impacts-tooltip';
                    document.body.appendChild(impactTooltip);
                }

                // Add event listeners to year headers
                const yearHeaders = document.querySelectorAll('.year-header.has-impacts');
                yearHeaders.forEach(header => {
                    const year = header.getAttribute('data-year');
                    if (year && yearTooltipData[year]) {
                        header.addEventListener('mouseenter', function(e) {
                            impactTooltip.innerHTML = yearTooltipData[year];
                            impactTooltip.classList.add('visible');
                            // Position the tooltip below the header
                            const rect = header.getBoundingClientRect();
                            const tooltipRect = impactTooltip.getBoundingClientRect();
                            let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
                            // Keep tooltip within viewport
                            if (left < 10) left = 10;
                            if (left + tooltipRect.width > window.innerWidth - 10) {
                                left = window.innerWidth - tooltipRect.width - 10;
                            }
                            impactTooltip.style.left = left + 'px';
                            impactTooltip.style.top = (rect.bottom + 5) + 'px';
                        });
                        header.addEventListener('mouseleave', function(e) {
                            impactTooltip.classList.remove('visible');
                        });
                    }
                });
            }
        }

        // Function to toggle table language between ES and EN
        function setTableLanguage(lang) {
            const esBtn = document.getElementById('lang-es-btn');
            const enBtn = document.getElementById('lang-en-btn');
            const wordCells = document.querySelectorAll('.word-cell');

            if (lang === 'en') {
                esBtn.classList.remove('active');
                enBtn.classList.add('active');
                // Show English in cells, Spanish on hover
                wordCells.forEach(cell => {
                    const enWord = cell.getAttribute('data-en');
                    const esWord = cell.getAttribute('data-es');
                    cell.textContent = enWord || cell.textContent;
                    cell.setAttribute('data-alt', esWord);
                });
            } else {
                esBtn.classList.add('active');
                enBtn.classList.remove('active');
                // Show Spanish in cells, English on hover
                wordCells.forEach(cell => {
                    const esWord = cell.getAttribute('data-es');
                    const enWord = cell.getAttribute('data-en');
                    cell.textContent = esWord || cell.textContent;
                    // Only show hover if there's a different translation
                    if (enWord && enWord !== esWord) {
                        cell.setAttribute('data-alt', enWord);
                    } else {
                        cell.removeAttribute('data-alt');
                    }
                });
            }
        }

        function closeTopWordsModal(event) {
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('top-words-modal').classList.remove('visible');
        }

        // Mini chart instances
        let prevalenceMiniChart = null;
        let giniMiniChart = null;
        let locationCountsMiniChart = null;
        let hoveredTopicIdx = null;  // Track which topic is being hovered
        let hoveredClusterName = null;  // Track which cluster is being hovered

        // Initialize mini charts
        function initMiniCharts() {
            const prevalenceCtx = document.getElementById('prevalence-mini-chart').getContext('2d');
            const giniCtx = document.getElementById('gini-mini-chart').getContext('2d');
            const locationCountsCtx = document.getElementById('location-counts-mini-chart').getContext('2d');

            const baseChartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        ticks: { font: { size: 9 }, maxRotation: 45 }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { font: { size: 9 } }
                    }
                },
                elements: {
                    point: { radius: 2 },
                    line: { tension: 0.3 }
                }
            };

            // Prevalence chart with interactive hover/click for topic selection
            const prevalenceOptions = {
                ...baseChartOptions,
                interaction: {
                    intersect: false,
                    mode: 'nearest',
                    axis: 'xy'
                },
                plugins: {
                    ...baseChartOptions.plugins,
                    tooltip: {
                        enabled: true,
                        callbacks: {
                            label: function(context) {
                                const topicSelectorValue = document.getElementById('topic-selector').value;
                                const isCorpusMode = topicSelectorValue === 'corpus';
                                const viewMode = getCurrentViewMode();

                                if (isCorpusMode) {
                                    // For corpus mode (bar chart), show document count
                                    return 'Documents: ' + context.parsed.y.toLocaleString();
                                }

                                if (viewMode === 'clusters' && context.dataset.clusterName !== undefined) {
                                    // Cluster mode: show cluster title
                                    const clusterName = context.dataset.clusterName;
                                    const clusterTitle = chartData && chartData.cluster_titles ? chartData.cluster_titles[clusterName] : clusterName;
                                    return clusterTitle + ': ' + context.parsed.y.toFixed(4);
                                }

                                const topicIdx = context.dataset.topicIdx;
                                if (topicIdx !== undefined && topicIdx !== null) {
                                    const title = topicTitles[topicIdx] || `Topic ${topicIdx}`;
                                    return title + ': ' + context.parsed.y.toFixed(4);
                                }
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(4);
                            }
                        }
                    }
                },
                onHover: function(event, elements, chart) {
                    const topicSelectorValue = document.getElementById('topic-selector').value;
                    const isCorpusMode = topicSelectorValue === 'corpus';
                    const viewMode = getCurrentViewMode();

                    if (isCorpusMode || elements.length === 0) {
                        // Reset hover state
                        if (hoveredTopicIdx !== null || hoveredClusterName !== null) {
                            hoveredTopicIdx = null;
                            hoveredClusterName = null;
                            updatePrevalenceChartStyles();
                        }
                        chart.canvas.style.cursor = isCorpusMode && elements.length > 0 ? 'pointer' : 'default';
                        return;
                    }

                    const element = elements[0];
                    const datasetIndex = element.datasetIndex;
                    const dataset = chart.data.datasets[datasetIndex];

                    if (viewMode === 'clusters' && dataset.clusterName !== undefined) {
                        // Cluster mode hover
                        chart.canvas.style.cursor = 'pointer';
                        if (hoveredClusterName !== dataset.clusterName) {
                            hoveredClusterName = dataset.clusterName;
                            hoveredTopicIdx = null;
                            updatePrevalenceChartStyles();
                        }
                    } else if (dataset.topicIdx !== undefined && dataset.topicIdx !== null) {
                        // Topic mode hover
                        chart.canvas.style.cursor = 'pointer';
                        if (hoveredTopicIdx !== dataset.topicIdx) {
                            hoveredTopicIdx = dataset.topicIdx;
                            hoveredClusterName = null;
                            updatePrevalenceChartStyles();
                        }
                    }
                },
                onClick: function(event, elements, chart) {
                    const topicSelectorValue = document.getElementById('topic-selector').value;
                    const isCorpusMode = topicSelectorValue === 'corpus';
                    const viewMode = getCurrentViewMode();

                    if (elements.length === 0) return;

                    const element = elements[0];
                    const datasetIndex = element.datasetIndex;
                    const dataset = chart.data.datasets[datasetIndex];
                    const timeIndex = element.index;  // Get the time period index from the clicked point

                    // Update the time slider to the clicked period
                    const timeSlider = document.getElementById('time-slider');
                    if (timeSlider && timeIndex !== undefined) {
                        timeSlider.value = timeIndex;
                        onTimeSliderChange();
                    }

                    if (viewMode === 'clusters' && dataset.clusterName !== undefined) {
                        // Cluster mode click - select the clicked cluster
                        const selector = document.getElementById('cluster-selector');
                        if (selector) {
                            selector.value = dataset.clusterName;
                            hoveredClusterName = null;
                            updateMap();
                            updateMiniCharts();
                            updateGlobalFigures();
                        }
                    } else if (!isCorpusMode && dataset.topicIdx !== undefined && dataset.topicIdx !== null) {
                        // Topic mode click - select the clicked topic
                        const selector = document.getElementById('topic-selector');
                        selector.value = dataset.topicIdx.toString();
                        hoveredTopicIdx = null;
                        updateMap();
                        updateMiniCharts();
                    }
                }
            };

            prevalenceMiniChart = new Chart(prevalenceCtx, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: prevalenceOptions
            });

            giniMiniChart = new Chart(giniCtx, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    ...baseChartOptions,
                    plugins: {
                        ...baseChartOptions.plugins,
                        tooltip: {
                            enabled: true,
                            callbacks: {
                                label: function(context) {
                                    return 'Gini: ' + context.parsed.y.toFixed(3);
                                }
                            }
                        }
                    },
                    scales: {
                        ...baseChartOptions.scales,
                        y: {
                            ...baseChartOptions.scales.y,
                            max: 1,
                            ticks: {
                                font: { size: 9 },
                                callback: function(value) {
                                    return value.toFixed(1);
                                }
                            }
                        }
                    }
                }
            });

            // Location counts chart - shows counts per location over time
            locationCountsMiniChart = new Chart(locationCountsCtx, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    ...baseChartOptions,
                    plugins: {
                        ...baseChartOptions.plugins,
                        legend: {
                            display: true,
                            position: 'bottom',
                            labels: {
                                boxWidth: 8,
                                font: { size: 9 },
                                padding: 4
                            }
                        },
                        tooltip: {
                            enabled: true,
                            callbacks: {
                                label: function(context) {
                                    return context.dataset.label + ': ' + context.parsed.y.toFixed(0);
                                }
                            }
                        }
                    },
                    elements: {
                        point: { radius: 2 },
                        line: { tension: 0.3, borderWidth: 1.5 }
                    },
                    onClick: function(event, elements, chart) {
                        if (elements.length === 0) return;

                        const element = elements[0];
                        const datasetIndex = element.datasetIndex;
                        const dataset = chart.data.datasets[datasetIndex];
                        const locationName = dataset.locationName || dataset.label;

                        if (locationName) {
                            showLocationChart(locationName);
                        }
                    }
                }
            });
        }

        // Update prevalence chart styles based on selected/hovered topic or cluster
        function updatePrevalenceChartStyles() {
            if (!prevalenceMiniChart) return;

            const viewMode = getCurrentViewMode();
            const topicSelectorValue = document.getElementById('topic-selector').value;
            const isCorpusMode = topicSelectorValue === 'corpus';
            const selectedTopicIdx = isCorpusMode ? -1 : parseInt(topicSelectorValue);
            const selectedCluster = getSelectedCluster();

            const datasets = prevalenceMiniChart.data.datasets;
            const clusterColors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#607D8B'];

            for (let i = 0; i < datasets.length; i++) {
                const dataset = datasets[i];

                if (viewMode === 'clusters' && dataset.clusterName !== undefined) {
                    // Cluster mode styling
                    const clusterName = dataset.clusterName;
                    const clusterIdx = dataset.clusterIdx;
                    const isSelected = clusterName === selectedCluster;
                    const isHovered = clusterName === hoveredClusterName;
                    const clusterColor = clusterColors[clusterIdx % clusterColors.length];

                    if (isSelected) {
                        dataset.borderColor = clusterColor;
                        dataset.backgroundColor = clusterColor + '33';
                        dataset.borderWidth = 2;
                        dataset.pointRadius = 3;
                        dataset.order = 0;
                    } else if (isHovered) {
                        dataset.borderColor = clusterColor;
                        dataset.backgroundColor = clusterColor + '22';
                        dataset.borderWidth = 2;
                        dataset.pointRadius = 2;
                        dataset.order = 1;
                    } else {
                        dataset.borderColor = '#e0e0e0';
                        dataset.backgroundColor = 'transparent';
                        dataset.borderWidth = 1;
                        dataset.pointRadius = 0;
                        dataset.order = 10;
                    }
                } else if (dataset.topicIdx !== undefined && dataset.topicIdx !== null) {
                    // Topic mode styling
                    const topicIdx = dataset.topicIdx;
                    const isSelected = topicIdx === selectedTopicIdx;
                    const isHovered = topicIdx === hoveredTopicIdx;
                    const topicColor = topicColors[topicIdx % topicColors.length];

                    if (isSelected) {
                        dataset.borderColor = topicColor;
                        dataset.backgroundColor = topicColor + '33';
                        dataset.borderWidth = 2;
                        dataset.pointRadius = 3;
                        dataset.order = 0;
                    } else if (isHovered) {
                        dataset.borderColor = topicColor;
                        dataset.backgroundColor = topicColor + '22';
                        dataset.borderWidth = 2;
                        dataset.pointRadius = 2;
                        dataset.order = 1;
                    } else {
                        dataset.borderColor = '#e0e0e0';
                        dataset.backgroundColor = 'transparent';
                        dataset.borderWidth = 1;
                        dataset.pointRadius = 0;
                        dataset.order = 10;
                    }
                }
            }

            prevalenceMiniChart.update('none');  // Update without animation
        }

        // Update mini charts based on selected topic/cluster
        function updateMiniCharts() {
            const viewMode = getCurrentViewMode();
            const topicSelectorValue = document.getElementById('topic-selector').value;
            const isCorpusMode = topicSelectorValue === 'corpus';
            const selectedTopicIdx = isCorpusMode ? -1 : parseInt(topicSelectorValue);
            const selectedCluster = getSelectedCluster();
            const currentGranularity = document.getElementById('granularity-selector')?.value || granularityLevels[0];

            // Update chart title based on mode
            updateLegend();

            // Reset hovered topic/cluster when updating charts
            hoveredTopicIdx = null;
            hoveredClusterName = null;

            // Helper function to get prevalence for a specific topic at a location
            function getTopicPrevalence(data, tIdx) {
                return data.topics[tIdx] || 0;
            }

            // Helper function to get cluster prevalence
            function getClusterPrevalence(data) {
                if (chartData && chartData.cluster_mapping && selectedCluster) {
                    const clusterTopics = chartData.cluster_mapping[selectedCluster] || [];
                    return clusterTopics.reduce((sum, tIdx) => sum + (data.topics[tIdx] || 0), 0);
                }
                return 0;
            }

            const allTimeData = allGranularityData[currentGranularity];

            // For Gini calculation: use selected topic or cluster
            const giniData = [];

            if (isCorpusMode) {
                // Corpus mode: show bar chart with article count
                const prevalenceData = [];

                for (let timeIdx = 0; timeIdx < timeLabels.length; timeIdx++) {
                    if (allTimeData && allTimeData[timeIdx]) {
                        const locationData = allTimeData[timeIdx].location_data;
                        let totalDocs = 0;
                        const docCounts = [];

                        for (const location in locationData) {
                            const data = locationData[location];
                            const docCount = data.doc_count || 0;
                            totalDocs += docCount;
                            if (docCount > 0) docCounts.push(docCount);
                        }

                        prevalenceData.push(totalDocs);
                        giniData.push(calculateGini(docCounts));
                    } else {
                        prevalenceData.push(0);
                        giniData.push(0);
                    }
                }

                // Bar chart dataset for corpus mode - use blue colors matching popup chart
                prevalenceMiniChart.config.type = 'bar';
                prevalenceMiniChart.data.labels = timeLabels;
                prevalenceMiniChart.data.datasets = [{
                    data: prevalenceData,
                    backgroundColor: '#2196F3',
                    borderColor: '#1976D2',
                    borderWidth: 1,
                    topicIdx: null
                }];
                // Update x-axis to use offset so bars appear between gridlines
                prevalenceMiniChart.options.scales.x.offset = true;
                prevalenceMiniChart.update();

            } else if (viewMode === 'clusters') {
                // Cluster mode: show ALL clusters, highlight selected one
                const clusterNames = chartData.cluster_names || [];
                const clusterPrevalenceData = {};

                // Initialize data arrays for each cluster
                for (const clusterName of clusterNames) {
                    clusterPrevalenceData[clusterName] = [];
                }

                for (let timeIdx = 0; timeIdx < timeLabels.length; timeIdx++) {
                    if (allTimeData && allTimeData[timeIdx]) {
                        const locationData = allTimeData[timeIdx].location_data;

                        // Calculate weighted average prevalence for each cluster
                        const clusterSums = {};
                        for (const clusterName of clusterNames) {
                            clusterSums[clusterName] = 0;
                        }
                        let totalDocs = 0;
                        const selectedClusterWordCounts = [];

                        for (const location in locationData) {
                            const data = locationData[location];
                            const docCount = data.doc_count || 1;
                            totalDocs += docCount;

                            // Calculate prevalence for each cluster
                            for (const clusterName of clusterNames) {
                                const clusterTopics = chartData.cluster_mapping[clusterName] || [];
                                let clusterPrevalence = 0;
                                for (const topicIdx of clusterTopics) {
                                    clusterPrevalence += data.topics[topicIdx] || 0;
                                }
                                clusterSums[clusterName] += clusterPrevalence * docCount;
                            }

                            // For Gini: use word-level counts for selected cluster
                            if (data.word_counts && chartData.cluster_mapping && selectedCluster) {
                                const clusterTopics = chartData.cluster_mapping[selectedCluster] || [];
                                let locWordCount = 0;
                                for (const topicIdx of clusterTopics) {
                                    locWordCount += data.word_counts[topicIdx] || 0;
                                }
                                if (locWordCount > 0) selectedClusterWordCounts.push(locWordCount);
                            }
                        }

                        // Store weighted average for each cluster
                        for (const clusterName of clusterNames) {
                            clusterPrevalenceData[clusterName].push(totalDocs > 0 ? clusterSums[clusterName] / totalDocs : 0);
                        }

                        giniData.push(calculateGini(selectedClusterWordCounts));
                    } else {
                        for (const clusterName of clusterNames) {
                            clusterPrevalenceData[clusterName].push(0);
                        }
                        giniData.push(0);
                    }
                }

                // Create datasets for ALL clusters
                const datasets = [];
                const clusterColors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#607D8B'];

                clusterNames.forEach((clusterName, idx) => {
                    const isSelected = clusterName === selectedCluster;
                    const clusterColor = clusterColors[idx % clusterColors.length];
                    const clusterTitle = chartData.cluster_titles ? chartData.cluster_titles[clusterName] : clusterName;

                    datasets.push({
                        data: clusterPrevalenceData[clusterName],
                        label: clusterTitle || clusterName,
                        borderColor: isSelected ? clusterColor : '#e0e0e0',
                        backgroundColor: isSelected ? clusterColor + '33' : 'transparent',
                        borderWidth: isSelected ? 2 : 1,
                        pointRadius: isSelected ? 3 : 0,
                        pointHoverRadius: 4,
                        fill: isSelected,
                        order: isSelected ? 0 : 10,
                        clusterName: clusterName,  // Store cluster name for hover/click handling
                        clusterIdx: idx  // Store cluster index for color lookup
                    });
                });

                prevalenceMiniChart.config.type = 'line';
                prevalenceMiniChart.data.labels = timeLabels;
                prevalenceMiniChart.data.datasets = datasets;
                // Reset x-axis offset for line chart
                prevalenceMiniChart.options.scales.x.offset = false;
                prevalenceMiniChart.update();

            } else {
                // Topic mode: show ALL topics, highlight selected one
                // Calculate prevalence for each topic across all time periods
                const topicPrevalenceData = {};
                for (let tIdx = 0; tIdx < nTopics; tIdx++) {
                    topicPrevalenceData[tIdx] = [];
                }

                for (let timeIdx = 0; timeIdx < timeLabels.length; timeIdx++) {
                    if (allTimeData && allTimeData[timeIdx]) {
                        const locationData = allTimeData[timeIdx].location_data;

                        // Calculate weighted average prevalence for each topic
                        const topicSums = {};
                        for (let tIdx = 0; tIdx < nTopics; tIdx++) {
                            topicSums[tIdx] = 0;
                        }
                        let totalDocs = 0;
                        const selectedTopicWordCounts = [];

                        for (const location in locationData) {
                            const data = locationData[location];
                            const docCount = data.doc_count || 1;
                            totalDocs += docCount;

                            for (let tIdx = 0; tIdx < nTopics; tIdx++) {
                                const locPrevalence = getTopicPrevalence(data, tIdx);
                                topicSums[tIdx] += locPrevalence * docCount;
                            }

                            // For Gini: use word-level counts for selected topic
                            if (data.word_counts) {
                                const wordCount = data.word_counts[selectedTopicIdx] || 0;
                                if (wordCount > 0) selectedTopicWordCounts.push(wordCount);
                            }
                        }

                        // Store weighted average for each topic
                        for (let tIdx = 0; tIdx < nTopics; tIdx++) {
                            topicPrevalenceData[tIdx].push(totalDocs > 0 ? topicSums[tIdx] / totalDocs : 0);
                        }

                        giniData.push(calculateGini(selectedTopicWordCounts));
                    } else {
                        for (let tIdx = 0; tIdx < nTopics; tIdx++) {
                            topicPrevalenceData[tIdx].push(0);
                        }
                        giniData.push(0);
                    }
                }

                // Create datasets for ALL topics
                const datasets = [];
                for (let tIdx = 0; tIdx < nTopics; tIdx++) {
                    const isSelected = tIdx === selectedTopicIdx;
                    const topicColor = topicColors[tIdx % topicColors.length];

                    datasets.push({
                        data: topicPrevalenceData[tIdx],
                        label: topicTitles[tIdx] || `Topic ${tIdx}`,
                        borderColor: isSelected ? topicColor : '#e0e0e0',
                        backgroundColor: isSelected ? topicColor + '33' : 'transparent',
                        borderWidth: isSelected ? 2 : 1,
                        pointRadius: isSelected ? 3 : 0,
                        pointHoverRadius: 4,
                        fill: isSelected,
                        order: isSelected ? 0 : 10,
                        topicIdx: tIdx  // Store topic index for hover/click handling
                    });
                }

                prevalenceMiniChart.config.type = 'line';
                prevalenceMiniChart.data.labels = timeLabels;
                prevalenceMiniChart.data.datasets = datasets;
                // Reset x-axis offset for line chart
                prevalenceMiniChart.options.scales.x.offset = false;
                prevalenceMiniChart.update();
            }

            // Update Gini chart (same for all modes)
            giniMiniChart.data.labels = timeLabels;
            giniMiniChart.data.datasets = [{
                data: giniData,
                borderColor: '#FF9800',
                backgroundColor: '#FF980033',
                fill: true
            }];
            giniMiniChart.update();

            // Update location counts chart
            updateLocationCountsChart(isCorpusMode, selectedTopicIdx, currentGranularity, allTimeData, viewMode, selectedCluster);

            // Update location counts chart title
            const locationCountsTitle = document.getElementById('location-counts-chart-title');
            if (locationCountsTitle) {
                if (isCorpusMode) {
                    locationCountsTitle.textContent = '📍 Number of Documents in Top 10 Locations';
                } else if (viewMode === 'clusters') {
                    locationCountsTitle.textContent = '📍 Word-level Cluster Counts in Top 10 Locations';
                } else {
                    locationCountsTitle.textContent = '📍 Word-level Topic Counts in Top 10 Locations';
                }
            }

            // Update top words tags
            updateTopWordsTags();
        }

        // Update location counts chart - shows counts per location over time
        function updateLocationCountsChart(isCorpusMode, selectedTopicIdx, currentGranularity, allTimeData, viewMode, selectedCluster) {
            if (!locationCountsMiniChart || !allTimeData) return;

            // Helper function to calculate word count for a location based on mode
            function getWordCount(data) {
                if (isCorpusMode) {
                    return data.doc_count || 0;
                } else if (viewMode === 'clusters' && chartData && chartData.cluster_mapping && selectedCluster) {
                    // Cluster mode: sum word counts for all topics in the cluster
                    const clusterTopics = chartData.cluster_mapping[selectedCluster] || [];
                    let total = 0;
                    if (data.word_counts) {
                        for (const topicIdx of clusterTopics) {
                            total += data.word_counts[topicIdx] || 0;
                        }
                    }
                    return total;
                } else {
                    // Topic mode: use word-level counts for selected topic
                    return data.word_counts ? (data.word_counts[selectedTopicIdx] || 0) : 0;
                }
            }

            // Get all unique locations across all time periods
            const allLocations = new Set();
            for (let timeIdx = 0; timeIdx < timeLabels.length; timeIdx++) {
                if (allTimeData[timeIdx]) {
                    const locationData = allTimeData[timeIdx].location_data;
                    for (const location in locationData) {
                        allLocations.add(location);
                    }
                }
            }

            // Sort locations by total count and limit to top 10
            const locationTotals = {};
            allLocations.forEach(loc => {
                let total = 0;
                for (let timeIdx = 0; timeIdx < timeLabels.length; timeIdx++) {
                    if (allTimeData[timeIdx] && allTimeData[timeIdx].location_data[loc]) {
                        const data = allTimeData[timeIdx].location_data[loc];
                        total += getWordCount(data);
                    }
                }
                locationTotals[loc] = total;
            });

            // Sort by total and take top 10
            const sortedLocations = Object.entries(locationTotals)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10)
                .map(([loc, _]) => loc);

            // Generate colors for each location
            const locationColors = [
                '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336',
                '#00BCD4', '#FFEB3B', '#795548', '#607D8B', '#E91E63'
            ];

            // Calculate data for each location
            const datasets = [];
            sortedLocations.forEach((location, idx) => {
                const data = [];
                for (let timeIdx = 0; timeIdx < timeLabels.length; timeIdx++) {
                    if (allTimeData[timeIdx] && allTimeData[timeIdx].location_data[location]) {
                        const locData = allTimeData[timeIdx].location_data[location];
                        data.push(getWordCount(locData));
                    } else {
                        data.push(0);
                    }
                }

                datasets.push({
                    label: location,
                    data: data,
                    borderColor: locationColors[idx % locationColors.length],
                    backgroundColor: locationColors[idx % locationColors.length] + '22',
                    fill: false,
                    locationName: location  // Store location name for click handler
                });
            });

            locationCountsMiniChart.data.labels = timeLabels;
            locationCountsMiniChart.data.datasets = datasets;
            locationCountsMiniChart.update();
        }

        // Legacy function - kept for compatibility but functionality moved to updateTopWordsTags
        function updateTopWordsPreview() {
            const timeIdx = parseInt(document.getElementById('time-slider').value);
            const viewMode = getCurrentViewMode();
            const topicSelectorValue = document.getElementById('topic-selector').value;
            const isCorpusMode = topicSelectorValue === 'corpus';
            const topicIdx = isCorpusMode ? -1 : parseInt(topicSelectorValue);
            const selectedCluster = getSelectedCluster();

            let previewText = '';

            if (timeIdx === -1) {
                // "All Periods" mode: combine top words from all time periods
                if (chartData && chartData.top_words_by_chunk) {
                    const wordFreq = {};
                    for (const timeLabel in chartData.top_words_by_chunk) {
                        const chunkWords = chartData.top_words_by_chunk[timeLabel];
                        if (viewMode === 'clusters' && chartData.cluster_mapping && selectedCluster) {
                            const clusterTopics = chartData.cluster_mapping[selectedCluster] || [];
                            for (const tIdx of clusterTopics) {
                                if (chunkWords[tIdx]) {
                                    chunkWords[tIdx].slice(0, 5).forEach(w => {
                                        wordFreq[w] = (wordFreq[w] || 0) + 1;
                                    });
                                }
                            }
                        } else if (isCorpusMode) {
                            for (let tIdx = 0; tIdx < nTopics; tIdx++) {
                                if (chunkWords[tIdx]) {
                                    chunkWords[tIdx].slice(0, 3).forEach(w => {
                                        wordFreq[w] = (wordFreq[w] || 0) + 1;
                                    });
                                }
                            }
                        } else {
                            if (chunkWords[topicIdx]) {
                                chunkWords[topicIdx].slice(0, 5).forEach(w => {
                                    wordFreq[w] = (wordFreq[w] || 0) + 1;
                                });
                            }
                        }
                    }
                    // Sort by frequency and take top words
                    const sortedWords = Object.entries(wordFreq)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 10)
                        .map(([w, _]) => w);
                    previewText = sortedWords.join(', ');
                }
            } else {
                const timeLabel = timeLabels[timeIdx];
                if (chartData && chartData.top_words_by_chunk && chartData.top_words_by_chunk[timeLabel]) {
                    const chunkWords = chartData.top_words_by_chunk[timeLabel];
                    if (viewMode === 'clusters' && chartData.cluster_mapping && selectedCluster) {
                        const clusterTopics = chartData.cluster_mapping[selectedCluster] || [];
                        const words = [];
                        for (const tIdx of clusterTopics) {
                            if (chunkWords[tIdx]) {
                                words.push(...chunkWords[tIdx].slice(0, 3));
                            }
                        }
                        previewText = words.slice(0, 8).join(', ');
                    } else if (isCorpusMode) {
                        // Show deduplicated top words from all topics, sorted by frequency
                        const wordFreq = {};
                        for (let tIdx = 0; tIdx < nTopics; tIdx++) {
                            if (chunkWords[tIdx]) {
                                chunkWords[tIdx].slice(0, 5).forEach(w => {
                                    wordFreq[w] = (wordFreq[w] || 0) + 1;
                                });
                            }
                        }
                        const sortedWords = Object.entries(wordFreq)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 10)
                            .map(([w, _]) => w);
                        previewText = sortedWords.join(', ');
                    } else {
                        if (chunkWords[topicIdx]) {
                            previewText = chunkWords[topicIdx].slice(0, 8).join(', ');
                        }
                    }
                }
            }
            document.getElementById('top-words-preview').textContent = previewText || 'No data available';
        }

        // Close modal on Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeChartModal();
                closeTopWordsModal();
            }
        });

        // Helper function to interpolate between two colors
        function interpolateColor(color1, color2, factor) {
            const c1 = parseInt(color1.slice(1), 16);
            const c2 = parseInt(color2.slice(1), 16);

            const r1 = (c1 >> 16) & 0xff;
            const g1 = (c1 >> 8) & 0xff;
            const b1 = c1 & 0xff;

            const r2 = (c2 >> 16) & 0xff;
            const g2 = (c2 >> 8) & 0xff;
            const b2 = c2 & 0xff;

            const r = Math.round(r1 + factor * (r2 - r1));
            const g = Math.round(g1 + factor * (g2 - g1));
            const b = Math.round(b1 + factor * (b2 - b1));

            return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
        }

        // Get color based on intensity (0-1)
        function getColor(intensity) {
            if (intensity < 0.5) {
                // Yellow to Orange
                return interpolateColor('#ffeb3b', '#ff9800', intensity * 2);
            } else {
                // Orange to Red
                return interpolateColor('#ff9800', '#f44336', (intensity - 0.5) * 2);
            }
        }

        // Calculate Gini coefficient for an array of values
        // Gini = 0 means perfect equality, Gini = 1 means maximum inequality
        function calculateGini(values) {
            // Filter out zeros and non-positive values
            const filtered = values.filter(v => v > 0);

            if (filtered.length <= 1) {
                return 0; // No inequality with 0 or 1 location
            }

            // Sort values in ascending order
            const sorted = [...filtered].sort((a, b) => a - b);
            const n = sorted.length;
            const sum = sorted.reduce((acc, val) => acc + val, 0);

            if (sum === 0) {
                return 0;
            }

            // Calculate Gini using the formula: G = (2 * sum(i * yi)) / (n * sum(yi)) - (n + 1) / n
            let weightedSum = 0;
            for (let i = 0; i < n; i++) {
                weightedSum += (i + 1) * sorted[i];
            }

            const gini = (2 * weightedSum) / (n * sum) - (n + 1) / n;
            return Math.max(0, Math.min(1, gini)); // Clamp between 0 and 1
        }

        // Update legend, chart titles, panel title, description, and top words based on current mode
        function updateLegend() {
            const topicSelectorValue = document.getElementById('topic-selector').value;
            const isCorpusMode = topicSelectorValue === 'corpus';
            const topicIdx = isCorpusMode ? -1 : parseInt(topicSelectorValue);

            // Update map legend title
            const legendTitle = document.getElementById('legend-title');
            const viewMode = getCurrentViewMode();
            const selectedCluster = getSelectedCluster();
            if (legendTitle) {
                if (isCorpusMode) {
                    legendTitle.textContent = '📊 Document Count';
                } else if (viewMode === 'clusters') {
                    legendTitle.textContent = '📊 Cluster Prevalence';
                } else {
                    legendTitle.textContent = '📊 Topic Prevalence';
                }
            }

            // Update chart title
            const chartTitle = document.getElementById('prevalence-chart-title');

            if (chartTitle) {
                if (isCorpusMode) {
                    chartTitle.textContent = '📈 Number of Documents';
                } else if (viewMode === 'clusters') {
                    chartTitle.textContent = '📈 Overall Cluster Prevalence';
                } else {
                    chartTitle.textContent = '📈 Overall Topic Prevalence';
                }
            }

            // Update global panel title
            const panelTitle = document.getElementById('global-panel-title');

            if (panelTitle) {
                if (isCorpusMode) {
                    panelTitle.textContent = '📊 Corpus';
                } else if (viewMode === 'clusters' && selectedCluster && chartData && chartData.cluster_titles) {
                    // Cluster mode: show cluster title
                    const clusterTitle = chartData.cluster_titles[selectedCluster] || selectedCluster;
                    panelTitle.textContent = '📊 ' + clusterTitle;
                } else {
                    const topicTitle = topicTitles[topicIdx] || topicTitles[topicIdx.toString()] || `Topic ${topicIdx}`;
                    panelTitle.textContent = '📊 ' + topicTitle;
                }
            }

            // Update cluster topic composition display
            const clusterTopicsContainer = document.getElementById('cluster-topics-container');
            if (clusterTopicsContainer) {
                if (viewMode === 'clusters' && selectedCluster && chartData && chartData.cluster_mapping) {
                    const clusterTopics = chartData.cluster_mapping[selectedCluster] || [];
                    // Get topic titles for each topic in the cluster, one per line
                    let topicsHtml = '<strong>Topics:</strong><br>';
                    clusterTopics.forEach(tIdx => {
                        const title = topicTitles[tIdx] || topicTitles[tIdx.toString()] || `Topic ${tIdx}`;
                        topicsHtml += '• ' + title + '<br>';
                    });
                    clusterTopicsContainer.innerHTML = topicsHtml;
                    clusterTopicsContainer.style.display = 'block';
                } else {
                    clusterTopicsContainer.style.display = 'none';
                    clusterTopicsContainer.innerHTML = '';
                }
            }

            // Update Gini chart title based on mode
            const giniChartTitle = document.getElementById('gini-chart-title');
            if (giniChartTitle) {
                if (isCorpusMode) {
                    giniChartTitle.textContent = '📊 Gini Coefficient (Documents in Locations)';
                } else if (viewMode === 'clusters') {
                    giniChartTitle.textContent = '📊 Gini Coefficient (Cluster in Locations)';
                } else {
                    giniChartTitle.textContent = '📊 Gini Coefficient (Topic in Locations)';
                }
            }

            // Update topic/cluster description
            const descContainer = document.getElementById('topic-description-container');
            if (descContainer) {
                if (isCorpusMode) {
                    descContainer.style.display = 'none';
                    descContainer.textContent = '';
                } else if (viewMode === 'clusters' && selectedCluster && chartData && chartData.cluster_descriptions) {
                    // Cluster mode: show cluster description
                    const clusterDescription = chartData.cluster_descriptions[selectedCluster] || '';
                    if (clusterDescription) {
                        descContainer.textContent = clusterDescription;
                        descContainer.style.display = 'block';
                    } else {
                        descContainer.style.display = 'none';
                    }
                } else {
                    const explanation = topicExplanations[topicIdx] || topicExplanations[topicIdx.toString()] || '';
                    if (explanation) {
                        descContainer.textContent = explanation;
                        descContainer.style.display = 'block';
                    } else {
                        descContainer.style.display = 'none';
                    }
                }
            }

            // Update top words tags
            updateTopWordsTags();
        }

        // Update top words as clickable tags with translation tooltips
        function updateTopWordsTags() {
            const container = document.getElementById('top-words-tags');
            if (!container) return;

            const topicSelectorValue = document.getElementById('topic-selector').value;
            const isCorpusMode = topicSelectorValue === 'corpus';
            const topicIdx = isCorpusMode ? -1 : parseInt(topicSelectorValue);
            const timeIdx = parseInt(document.getElementById('time-slider').value);
            const viewMode = getCurrentViewMode();
            const selectedCluster = getSelectedCluster();

            let words = [];

            if (chartData && chartData.top_words_by_chunk) {
                if (timeIdx === -1) {
                    // All periods mode - aggregate across all time periods
                    const wordFreq = {};
                    for (const timeLabel in chartData.top_words_by_chunk) {
                        const chunkWords = chartData.top_words_by_chunk[timeLabel];
                        if (viewMode === 'clusters' && chartData.cluster_mapping && selectedCluster) {
                            // Aggregate top words across all topics in the cluster
                            // Weight by rank position within each topic
                            const clusterTopics = chartData.cluster_mapping[selectedCluster] || [];
                            for (const tIdx of clusterTopics) {
                                if (chunkWords[tIdx]) {
                                    chunkWords[tIdx].forEach((w, rank) => {
                                        const score = 10 - rank;
                                        wordFreq[w] = (wordFreq[w] || 0) + score;
                                    });
                                }
                            }
                        } else if (isCorpusMode) {
                            for (let tIdx = 0; tIdx < nTopics; tIdx++) {
                                if (chunkWords[tIdx]) {
                                    chunkWords[tIdx].slice(0, 3).forEach(w => {
                                        wordFreq[w] = (wordFreq[w] || 0) + 1;
                                    });
                                }
                            }
                        } else {
                            if (chunkWords[topicIdx]) {
                                chunkWords[topicIdx].slice(0, 5).forEach(w => {
                                    wordFreq[w] = (wordFreq[w] || 0) + 1;
                                });
                            }
                        }
                    }
                    words = Object.entries(wordFreq)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 10)
                        .map(([w, _]) => w);
                } else {
                    // Specific time period
                    const timeLabel = timeLabels[timeIdx];
                    const chunkWords = chartData.top_words_by_chunk[timeLabel];
                    if (chunkWords) {
                        if (viewMode === 'clusters' && chartData.cluster_mapping && selectedCluster) {
                            // Aggregate top words across all topics in the cluster
                            // Weight words by their rank position (higher rank = higher weight)
                            const clusterTopics = chartData.cluster_mapping[selectedCluster] || [];
                            const wordScores = {};
                            for (const tIdx of clusterTopics) {
                                if (chunkWords[tIdx]) {
                                    chunkWords[tIdx].forEach((w, rank) => {
                                        // Weight by inverse rank (top word gets highest score)
                                        const score = 10 - rank;
                                        wordScores[w] = (wordScores[w] || 0) + score;
                                    });
                                }
                            }
                            // Sort by score and take top 10
                            words = Object.entries(wordScores)
                                .sort((a, b) => b[1] - a[1])
                                .slice(0, 10)
                                .map(([w, _]) => w);
                        } else if (isCorpusMode) {
                            const wordFreq = {};
                            for (let tIdx = 0; tIdx < nTopics; tIdx++) {
                                if (chunkWords[tIdx]) {
                                    chunkWords[tIdx].slice(0, 5).forEach(w => {
                                        wordFreq[w] = (wordFreq[w] || 0) + 1;
                                    });
                                }
                            }
                            words = Object.entries(wordFreq)
                                .sort((a, b) => b[1] - a[1])
                                .slice(0, 10)
                                .map(([w, _]) => w);
                        } else {
                            words = chunkWords[topicIdx] ? chunkWords[topicIdx].slice(0, 10) : [];
                        }
                    }
                }
            }

            // Build word tags HTML
            container.innerHTML = words.map(word => {
                const lowerWord = word.toLowerCase();
                const translation = hasTranslations ? (wordTranslations[lowerWord] || '') : '';
                if (translation) {
                    return `<span class="word-tag" data-translation="${translation}" title="${translation}">${word}</span>`;
                } else {
                    return `<span class="word-tag">${word}</span>`;
                }
            }).join('');
        }

        function updateMap() {
            // Remove existing layers
            mapLayers.forEach(layer => map.removeLayer(layer));
            mapLayers = [];

            const timeIdx = parseInt(document.getElementById('time-slider').value);
            const topicIdx = parseInt(document.getElementById('topic-selector').value);
            const viewMode = getCurrentViewMode();

            // Update legend based on mode
            updateLegend();
            const selectedCluster = getSelectedCluster();

            // Get selected granularity level (default to first if no selector exists)
            let currentGranularity = granularityLevels[0];
            const granularitySelector = document.getElementById('granularity-selector');
            if (granularitySelector) {
                currentGranularity = granularitySelector.value;
            }

            // Get data for the selected granularity level
            const allTimeData = allGranularityData[currentGranularity];

            let locationData, topWords;

            if (timeIdx === -1) {
                // "All Periods" mode: compute averaged data across all time periods
                locationData = {};
                topWords = allTimeData[0].top_words; // Use first period's top words as reference

                // Collect all unique locations and their data across all time periods
                const locationCounts = {}; // Track how many periods each location appears in

                for (let t = 0; t < allTimeData.length; t++) {
                    const periodData = allTimeData[t].location_data;
                    for (const loc in periodData) {
                        if (!locationData[loc]) {
                            locationData[loc] = {
                                doc_count: 0,
                                topics: new Array(nTopics).fill(0),
                                word_counts: new Array(nTopics).fill(0),
                                coords: periodData[loc].coords
                            };
                            locationCounts[loc] = 0;
                        }
                        locationData[loc].doc_count += periodData[loc].doc_count;
                        locationCounts[loc]++;

                        // Sum topic prevalences
                        if (periodData[loc].topics) {
                            for (let i = 0; i < periodData[loc].topics.length; i++) {
                                locationData[loc].topics[i] += periodData[loc].topics[i] || 0;
                            }
                        }

                        // Sum word counts (these should be summed, not averaged)
                        if (periodData[loc].word_counts) {
                            for (let i = 0; i < periodData[loc].word_counts.length; i++) {
                                locationData[loc].word_counts[i] += periodData[loc].word_counts[i] || 0;
                            }
                        }
                    }
                }

                // Average the topic prevalences by the number of periods each location appears
                // Note: word_counts are kept as sums (not averaged) since they represent raw counts
                for (const loc in locationData) {
                    const count = locationCounts[loc];
                    if (count > 0) {
                        for (let i = 0; i < locationData[loc].topics.length; i++) {
                            locationData[loc].topics[i] /= count;
                        }
                    }
                }

                // Update time indicator
                document.getElementById('time-indicator').textContent = 'All Periods';
                document.getElementById('stats-time-period').textContent = 'All Periods';
            } else {
                const currentData = allTimeData[timeIdx];
                locationData = currentData.location_data;
                topWords = currentData.top_words;

                // Update time indicator
                document.getElementById('time-indicator').textContent = timeLabels[timeIdx];
                document.getElementById('stats-time-period').textContent = timeLabels[timeIdx];
            }

            // Helper function to get raw word count value based on view mode (for normalization)
            function getWordCountValue(locData, locationName) {
                if (viewMode === 'clusters' && chartData && chartData.cluster_mapping && selectedCluster) {
                    // Compute cluster word count by summing topic word counts
                    const clusterTopics = chartData.cluster_mapping[selectedCluster] || [];
                    let clusterWordCount = 0;
                    for (const tIdx of clusterTopics) {
                        if (locData.word_counts && locData.word_counts[tIdx] !== undefined) {
                            clusterWordCount += locData.word_counts[tIdx];
                        }
                    }
                    return clusterWordCount;
                }
                // Handle "corpus" mode - use document count to show regional activity
                const topicSelectorValue = document.getElementById('topic-selector').value;
                if (topicSelectorValue === 'corpus') {
                    return locData.doc_count || 0;
                }
                // Topics mode - use word counts
                // Get the selected topic index directly from the selector
                const selectedTopicIdx = parseInt(topicSelectorValue);
                if (isNaN(selectedTopicIdx) || !locData.word_counts) {
                    return 0;
                }
                return locData.word_counts[selectedTopicIdx] || 0;
            }

            // Check if we're in corpus mode
            const isCorpusMode = document.getElementById('topic-selector').value === 'corpus';

            // Find total word counts for normalization and max for corpus mode
            let totalWordCount = 0;
            let maxDocCount = 0;
            let totalDocs = 0;
            for (const location in locationData) {
                const wordCount = getWordCountValue(locationData[location], location);
                totalWordCount += wordCount;
                if (isCorpusMode && wordCount > maxDocCount) maxDocCount = wordCount;
                totalDocs += locationData[location].doc_count;
            }

            // Update statistics
            document.getElementById('location-count').textContent = Object.keys(locationData).length;
            document.getElementById('document-count').textContent = totalDocs;

            if (useGeoJSON && geoJSONData && geoJSONData[currentGranularity]) {
                // GeoJSON choropleth mode
                const geojson = geoJSONData[currentGranularity];
                const nameProp = locationNameProps[currentGranularity];

                // Pre-calculate max normalized value for scaling (only needed for topics/clusters mode)
                let maxNormalized = 0;
                if (!isCorpusMode) {
                    for (const loc in locationData) {
                        const locWordCount = getWordCountValue(locationData[loc], loc);
                        const normalized = totalWordCount > 0 ? locWordCount / totalWordCount : 0;
                        if (normalized > maxNormalized) maxNormalized = normalized;
                    }
                }

                const geoLayer = L.geoJSON(geojson, {
                    style: function(feature) {
                        const locationName = feature.properties[nameProp];
                        const data = locationData[locationName];

                        if (!data) {
                            // Location not in data - show as gray
                            return {
                                fillColor: '#cccccc',
                                weight: 1,
                                opacity: 1,
                                color: '#666666',
                                fillOpacity: 0.3
                            };
                        }

                        const wordCount = getWordCountValue(data, locationName);
                        // For corpus mode, use max scaling; for topics/clusters, use normalized prevalence
                        let intensity;
                        if (isCorpusMode) {
                            intensity = maxDocCount > 0 ? wordCount / maxDocCount : 0;
                        } else {
                            const normalized = totalWordCount > 0 ? wordCount / totalWordCount : 0;
                            intensity = maxNormalized > 0 ? normalized / maxNormalized : 0;
                        }
                        const color = getColor(intensity);

                        return {
                            fillColor: color,
                            weight: 1,
                            opacity: 1,
                            color: '#333333',
                            fillOpacity: 0.7
                        };
                    },
                    onEachFeature: function(feature, layer) {
                        const locationName = feature.properties[nameProp];
                        const data = locationData[locationName];

                        // Add tooltip with location name and prevalence/documents
                        let tooltipText = locationName;
                        if (data) {
                            const wordCount = getWordCountValue(data, locationName);
                            if (isCorpusMode) {
                                tooltipText += `<br><span style="font-size: 11px; font-weight: normal;">Documents: ${wordCount}</span>`;
                            } else {
                                const normalizedPrevalence = totalWordCount > 0 ? wordCount / totalWordCount : 0;
                                tooltipText += `<br><span style="font-size: 11px; font-weight: normal;">Prevalence: ${normalizedPrevalence.toFixed(4)}</span>`;
                            }
                        }
                        layer.bindTooltip(tooltipText, {
                            permanent: false,
                            direction: 'top',
                            className: 'location-tooltip'
                        });

                        // Click handler to show chart
                        layer.on('click', function(e) {
                            showLocationChart(locationName);
                        });

                        // Hover effects
                        layer.on('mouseover', function(e) {
                            this.setStyle({
                                weight: 2,
                                fillOpacity: 0.9
                            });
                            this.bringToFront();
                        });

                        layer.on('mouseout', function(e) {
                            geoLayer.resetStyle(this);
                        });
                    }
                }).addTo(map);

                mapLayers.push(geoLayer);

                // Add cluster bar plots when checkbox is checked
                const showBarPlots = document.getElementById('show-barplots-checkbox')?.checked || false;
                const hasClusters = chartData && chartData.cluster_mapping && chartData.cluster_names && chartData.cluster_names.length > 0;

                if (showBarPlots && hasClusters) {
                    const clusterNames = chartData.cluster_names;
                    const clusterMapping = chartData.cluster_mapping;
                    const clusterColors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#607D8B'];

                    // Helper function to get centroid of a GeoJSON feature
                    function getFeatureCentroid(feature) {
                        const bounds = L.geoJSON(feature).getBounds();
                        return bounds.getCenter();
                    }

                    // For each feature in the GeoJSON, add a bar plot
                    geojson.features.forEach(function(feature) {
                        const locationName = feature.properties[nameProp];
                        const data = locationData[locationName];

                        if (!data || !data.topics) return;

                        // Calculate cluster shares for this location
                        const clusterShares = [];
                        let totalShare = 0;

                        for (let i = 0; i < clusterNames.length; i++) {
                            const clusterName = clusterNames[i];
                            const topicIndices = clusterMapping[clusterName] || [];
                            let clusterSum = 0;

                            for (const topicIdx of topicIndices) {
                                if (data.topics[topicIdx] !== undefined) {
                                    clusterSum += data.topics[topicIdx];
                                }
                            }

                            clusterShares.push(clusterSum);
                            totalShare += clusterSum;
                        }

                        // Normalize to get proportions
                        if (totalShare > 0) {
                            for (let i = 0; i < clusterShares.length; i++) {
                                clusterShares[i] = clusterShares[i] / totalShare;
                            }
                        }

                        // Create bar plot HTML
                        let barsHtml = '<div class="cluster-barplot-container">';
                        let labelsHtml = '<div class="cluster-barplot-labels">';

                        // Build tooltip content with cluster prevalence values
                        let tooltipContent = `<strong>${locationName}</strong><br>`;
                        for (let i = 0; i < clusterNames.length; i++) {
                            const height = Math.max(2, clusterShares[i] * 40); // Max 40px height
                            const color = clusterColors[i % clusterColors.length];
                            const shortLabel = clusterNames[i].replace('Cluster ', '').substring(0, 2);
                            const clusterTitle = chartData.cluster_titles ? chartData.cluster_titles[clusterNames[i]] : clusterNames[i];
                            const prevalencePercent = (clusterShares[i] * 100).toFixed(1);

                            barsHtml += `<div class="cluster-bar" style="height: ${height}px; background: ${color};"></div>`;
                            labelsHtml += `<div class="cluster-bar-label">${shortLabel}</div>`;
                            tooltipContent += `<span style="color: ${color};">■</span> ${clusterTitle}: ${prevalencePercent}%<br>`;
                        }

                        barsHtml += '</div>';
                        labelsHtml += '</div>';

                        const barPlotHtml = `<div class="cluster-barplot">${barsHtml}${labelsHtml}</div>`;

                        // Get centroid and create marker
                        const centroid = getFeatureCentroid(feature);
                        const barPlotIcon = L.divIcon({
                            className: 'cluster-barplot-marker',
                            html: barPlotHtml,
                            iconSize: [clusterNames.length * 18 + 8, 60],
                            iconAnchor: [(clusterNames.length * 18 + 8) / 2, 30]
                        });

                        const barPlotMarker = L.marker(centroid, {
                            icon: barPlotIcon,
                            interactive: true
                        }).addTo(map);

                        // Add tooltip with cluster prevalence values
                        barPlotMarker.bindTooltip(tooltipContent, {
                            permanent: false,
                            direction: 'top',
                            offset: [0, -30],
                            className: 'cluster-tooltip'
                        });

                        mapLayers.push(barPlotMarker);
                    });
                }

            } else {
                // Circle marker mode
                // First, find max normalized value for scaling
                let maxNormalized = 0;
                if (!isCorpusMode) {
                    for (const loc in locationData) {
                        const locWordCount = getWordCountValue(locationData[loc], loc);
                        const normalized = totalWordCount > 0 ? locWordCount / totalWordCount : 0;
                        if (normalized > maxNormalized) maxNormalized = normalized;
                    }
                }

                for (const location in locationData) {
                    const data = locationData[location];
                    const wordCount = getWordCountValue(data, location);
                    const coords = data.coords;

                    if (!coords) continue;

                    // Calculate intensity based on mode
                    let intensity;
                    if (isCorpusMode) {
                        intensity = maxDocCount > 0 ? wordCount / maxDocCount : 0;
                    } else {
                        const normalized = totalWordCount > 0 ? wordCount / totalWordCount : 0;
                        intensity = maxNormalized > 0 ? normalized / maxNormalized : 0;
                    }

                    // Scale radius by intensity (5 to 35 pixels)
                    const radius = 5 + intensity * 30;
                    const color = getColor(intensity);

                    const marker = L.circleMarker([coords[0], coords[1]], {
                        radius: radius,
                        color: color,
                        fillColor: color,
                        fillOpacity: 0.7,
                        weight: 1
                    }).addTo(map);

                    // Add tooltip with location name and prevalence/documents
                    let tooltipText;
                    if (isCorpusMode) {
                        tooltipText = `${location}<br><span style="font-size: 11px; font-weight: normal;">Documents: ${wordCount}</span>`;
                    } else {
                        const normalizedPrevalence = totalWordCount > 0 ? wordCount / totalWordCount : 0;
                        tooltipText = `${location}<br><span style="font-size: 11px; font-weight: normal;">Prevalence: ${normalizedPrevalence.toFixed(4)}</span>`;
                    }
                    marker.bindTooltip(tooltipText, {
                        permanent: false,
                        direction: 'top',
                        className: 'location-tooltip'
                    });

                    // Click handler to show chart
                    (function(loc) {
                        marker.on('click', function(e) {
                            showLocationChart(loc);
                        });
                    })(location);

                    // Add hover effects
                    marker.on('mouseover', function(e) {
                        this.setStyle({
                            fillOpacity: 0.9,
                            weight: 2
                        });
                    });

                    marker.on('mouseout', function(e) {
                        this.setStyle({
                            fillOpacity: 0.7,
                            weight: 1
                        });
                    });

                    mapLayers.push(marker);
                }
            }
        }

        // Initialize mini charts
        initMiniCharts();

        // Initialize with first time period and topic
        updateMap();
        updateMiniCharts();

        // Zoom slider functionality
        const zoomSlider = document.getElementById('zoom-slider');
        const zoomLevelDisplay = document.getElementById('zoom-level');
        
        zoomSlider.addEventListener('input', function() {
            const zoomLevel = parseInt(this.value);
            map.setZoom(zoomLevel);
            zoomLevelDisplay.textContent = zoomLevel;
        });
        
        // Update slider when map zoom changes (e.g., via mouse wheel)
        map.on('zoomend', function() {
            const currentZoom = map.getZoom();
            zoomSlider.value = currentZoom;
            zoomLevelDisplay.textContent = currentZoom;
        });
        </script>
    </body>
    </html>
    """

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Spatiotemporal interactive map saved to {output_file}")
    print(f"Time periods: {n_time_periods}")
    print(f"Topics: {n_topics}")
    print(f"Open the file in your browser to explore!")

    return all_time_data


def generate_toy_temporal_data(n_time_periods=4):
    """
    Generate toy data for multiple time periods.
    """
    np.random.seed(42)

    n_topics = 5
    n_words = 50

    # Same vocabulary across all time periods
    vocabulary = [
        'gobierno', 'presidente', 'congreso', 'ley', 'reforma',
        'economía', 'mercado', 'inversión', 'crecimiento', 'inflación',
        'educación', 'estudiantes', 'universidad', 'profesores', 'escuela',
        'salud', 'hospital', 'médicos', 'pacientes', 'tratamiento',
        'minería', 'cobre', 'exportación', 'producción', 'recursos',
        'región', 'ciudad', 'comuna', 'municipio', 'vecinos',
        'trabajo', 'empleo', 'sindicato', 'trabajadores', 'sueldo',
        'medio ambiente', 'agua', 'sequía', 'contaminación', 'energía',
        'seguridad', 'policía', 'delincuencia', 'justicia', 'tribunal',
        'cultura', 'arte', 'música', 'festival', 'patrimonio'
        ]

    chilean_cities = [
        'Santiago', 'Valparaíso', 'Concepción', 'La Serena',
        'Antofagasta', 'Temuco', 'Rancagua', 'Talca',
        'Arica', 'Iquique', 'Puerto Montt'
        ]

    document_topic_matrices = []
    word_topic_matrices = []
    locations_list = []
    time_labels = []

    for t in range(n_time_periods):
        # Number of documents can vary by time period
        n_docs = 80 + np.random.randint(-20, 30)

        # Create slightly different distributions for each time period
        doc_topic = np.random.dirichlet(np.ones(n_topics) * (2 + t * 0.5),
                                        size=n_docs)
        word_topic = np.random.dirichlet(np.ones(n_words) * (5 + t * 0.3),
                                         size=n_topics).T

        # Locations with some variation in distribution over time
        city_probs = np.random.dirichlet(np.ones(len(chilean_cities)) * 10)
        locs = np.random.choice(chilean_cities, size=n_docs, p=city_probs)

        document_topic_matrices.append(doc_topic)
        word_topic_matrices.append(word_topic)
        locations_list.append(locs.tolist())

        # Create time labels (years only)
        time_labels.append(f"{2022 + t}")

    return document_topic_matrices, word_topic_matrices, locations_list, vocabulary, time_labels


if __name__ == "__main__":
    # Check if RollingLDA model exists
    model_path = "roll_lda.pickle"
    data_path = "english_database.xlsx"
    chart_data_path = "topic_prevalence_by_location.json"

    # Load pre-computed chart data if available
    chart_data = None
    if os.path.exists(chart_data_path):
        print(f"Loading pre-computed chart data from {chart_data_path}...")
        with open(chart_data_path, 'r', encoding='utf-8') as f:
            chart_data = json.load(f)
        print(f"  Loaded data for {len(chart_data.get('granularity_levels', []))} granularity levels")
    else:
        print("No pre-computed chart data found. Charts will not be available.")
        print("Run analyze_topic_prevalence.py to generate chart data.")

    if os.path.exists(model_path) and os.path.exists(data_path):
        print("Found RollingLDA model and data files. Loading real model...")
        try:
            doc_topic_mats, word_topic_mats, locs_dict, vocab, time_labs = load_rolling_lda_data(
                model_path, data_path)

            print("Loaded RollingLDA data:")
            print(f"  Time periods: {len(time_labs)}")
            print(f"  Time labels: {time_labs}")
            print(f"  Topics: {doc_topic_mats[0].shape[1]}")
            print(f"  Granularity levels: {list(locs_dict.keys())}")
            print(f"  Documents per period: {[len(locs_dict['region'][i]) for i in range(len(time_labs))]}")

            # Check if chart data matches the current model
            if chart_data:
                chart_n_topics = chart_data.get('n_topics', 0)
                chart_time_labels = chart_data.get('time_labels', [])
                model_n_topics = doc_topic_mats[0].shape[1]

                if chart_n_topics != model_n_topics:
                    print(f"\n⚠️  WARNING: Chart data has {chart_n_topics} topics but model has {model_n_topics} topics!")
                    print("   Run 'python analyze_topic_prevalence.py' to regenerate chart data.")

                if chart_time_labels != time_labs:
                    print(f"\n⚠️  WARNING: Chart data time labels don't match model!")
                    print(f"   Chart data: {chart_time_labels}")
                    print(f"   Model: {time_labs}")
                    print("   Run 'python analyze_topic_prevalence.py' to regenerate chart data.")

            # Create the spatiotemporal map with granularity support
            # Use GeoJSON for choropleth if available
            geojson_files = {
                'region': 'chile_regions.geojson',
                'province': 'chile_provinces.geojson',
                'comuna': 'chile_comunas.geojson'
            }
            geojson_config = {}
            name_props = {}

            for level, path in geojson_files.items():
                if os.path.exists(path):
                    geojson_config[level] = path
                    print(f"Using GeoJSON for {level}: {path}")

            if 'region' in geojson_config:
                name_props['region'] = 'region_name'
            if 'province' in geojson_config:
                name_props['province'] = 'province_name'
            if 'comuna' in geojson_config:
                name_props['comuna'] = 'comuna_name'

            if geojson_config:
                # Explicit bounds for mainland Chile (excludes Easter Island which is at -109 lon)
                chile_bounds = [[-56.0, -76.0], [-17.0, -66.0]]
                all_data = create_spatiotemporal_interactive_map(
                    doc_topic_mats,
                    word_topic_mats,
                    locs_dict,
                    vocab,
                    time_labs,
                    output_file='chile_spatiotemporal_interactive.html',
                    title='Chilean Spatiotemporal Topic Map',
                    geojson=geojson_config,
                    location_name_property=name_props,
                    map_bounds=chile_bounds,
                    chart_data=chart_data,
                    topic_descriptions_csv="topic_descriptions.csv",
                    word_translations_csv="word_translations.csv",
                    cluster_sets_csv="cluster_sets.csv",
                    word_impacts_path="word_impacts.pickle"
                    )
            else:
                print("No GeoJSON found, using circle markers")
                all_data = create_spatiotemporal_interactive_map(
                    doc_topic_mats,
                    word_topic_mats,
                    locs_dict,
                    vocab,
                    time_labs,
                    output_file='chile_spatiotemporal_interactive.html',
                    title='Chilean Spatiotemporal Topic Map',
                    chart_data=chart_data,
                    cluster_sets_csv="cluster_sets.csv",
                    word_impacts_path="word_impacts.pickle"
                    )
        except Exception as e:
            print(f"Error loading RollingLDA model: {e}")
            print("Falling back to toy data...")
            # Fall through to toy data generation
            doc_topic_mats, word_topic_mats, locs_list, vocab, time_labs = generate_toy_temporal_data(
                n_time_periods=6)

            print("Generated toy temporal data:")
            print(f"  Time periods: {len(time_labs)}")
            print(f"  Time labels: {time_labs}")
            print(f"  Documents per period: {[len(locs) for locs in locs_list]}")

            all_data = create_spatiotemporal_interactive_map(
                doc_topic_mats,
                word_topic_mats,
                locs_list,
                vocab,
                time_labs,
                output_file='chile_spatiotemporal_interactive.html'
                )
    else:
        # Generate toy temporal data for demo/testing
        print("RollingLDA model not found. Generating toy data for demonstration...")
        doc_topic_mats, word_topic_mats, locs_list, vocab, time_labs = generate_toy_temporal_data(
            n_time_periods=6)

        print("Generated toy temporal data:")
        print(f"  Time periods: {len(time_labs)}")
        print(f"  Time labels: {time_labs}")
        print(f"  Documents per period: {[len(locs) for locs in locs_list]}")

        # Create the spatiotemporal map
        all_data = create_spatiotemporal_interactive_map(
            doc_topic_mats,
            word_topic_mats,
            locs_list,
            vocab,
            time_labs,
            output_file='chile_spatiotemporal_interactive.html'
            )

