import os
import requests
import json
from openai import OpenAI
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import time


# HELPER FUNCTIONS
def init_openai_client():
    """
    Initialize and return an OpenAI client.

    Returns:
        OpenAI: Initialized OpenAI client

    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment variables
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    client = OpenAI(api_key=api_key)
    return client


def init_genius_client():
    """
    Initialize and return Genius API configuration.

    Returns:
        tuple: (base_url, headers) for Genius API requests

    Raises:
        ValueError: If GENIUS_ACCESS_TOKEN is not found in environment variables
    """
    access_token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("GENIUS_ACCESS_TOKEN not found in environment variables")

    base_url = "https://api.genius.com"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "Trace/1.0",
    }

    return base_url, headers


def get_song_data(song_id, base_url, headers):
    """
    Fetch song data from Genius API.

    Args:
        song_id (int): ID of the song to fetch
        base_url (str): Genius API base URL
        headers (dict): Request headers including authorization

    Returns:
        dict: Song data from Genius API

    Raises:
        requests.exceptions.RequestException: If the API request fails
    """
    response = requests.get(f"{base_url}/songs/{song_id}", headers=headers)
    response.raise_for_status()
    return response.json()["response"]["song"]


def process_song_relationships(songs_dict, song_id, song_data):
    """
    Process song relationships and update the songs dictionary with sample information.

    Args:
        songs_dict (dict): Dictionary containing song information
        song_id (int): ID of the current song being processed
        song_data (dict): Song data from Genius API

    Returns:
        list: List of sample IDs for the current song
    """
    samples = []

    for relationship in song_data.get("song_relationships", []):
        if relationship["relationship_type"] == "samples":
            for sample in relationship["songs"]:
                if "(Instrumental)" in sample["title"]:
                    continue

                sample_id = sample["id"]
                if sample_id not in songs_dict:
                    songs_dict[sample_id] = {
                        "id": sample_id,
                        "title": sample["title"],
                        "artist": sample["artist_names"],
                        "year": sample["release_date_for_display"].split()[-1],
                        "sampled_by": [song_id],
                        "samples": [],
                    }
                else:
                    if "sampled_by" not in songs_dict[sample_id]:
                        songs_dict[sample_id]["sampled_by"] = []
                    songs_dict[sample_id]["sampled_by"].append(song_id)

                samples.append(sample_id)

    return samples


def update_songs_with_samples(songs_dict):
    """
    Updates the songs dictionary with sample information for all songs.

    Args:
        songs_dict (dict): Dictionary containing song information

    Returns:
        dict: Updated songs dictionary with sample information
    """
    # Create a copy of the keys to iterate over
    song_ids = list(songs_dict.keys())

    for song_id in song_ids:
        try:
            # Get song details from Genius API
            song = get_song_data(song_id, base_url, headers)

            # Process relationships using the helper function
            songs_dict[song_id]["samples"] = process_song_relationships(
                songs_dict, song_id, song
            )

            # Add a small delay to respect rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"Error processing song {song_id}: {str(e)}")
            continue

    return songs_dict


def update_visualization(songs_dict):
    """
    Updates the visualization with the latest song data.

    Args:
        songs_dict (dict): Updated songs dictionary

    Returns:
        plotly.graph_objects.Figure: Updated visualization
    """
    return visualize_sampling_network(songs_dict)


def calculate_y_positions(songs_dict, year_positions, k=1.0, iterations=50):
    """
    Calculate y positions using a force-directed layout that:
    - Keeps x positions fixed by year
    - Uses forces to prevent overlap
    - Keeps connected nodes closer together
    """
    # Initialize y positions randomly
    y_positions = {song_id: np.random.rand() for song_id in songs_dict}

    # Calculate year groups for better spacing
    year_groups = {}
    for song_id, song in songs_dict.items():
        year = int(song["year"])
        if year not in year_groups:
            year_groups[year] = []
        year_groups[year].append(song_id)

    # Force-directed layout iterations
    for _ in range(iterations):
        # Calculate repulsive forces between nodes in the same year
        for year, group in year_groups.items():
            for i, song_id1 in enumerate(group):
                for song_id2 in group[i + 1 :]:
                    # Repulsive force between nodes
                    dy = y_positions[song_id1] - y_positions[song_id2]
                    if abs(dy) < 0.1:  # If nodes are too close
                        # Push them apart
                        force = k * (0.1 - abs(dy))
                        y_positions[song_id1] += force
                        y_positions[song_id2] -= force

        # Attractive forces between connected nodes
        for song_id, song in songs_dict.items():
            for sampled_id in song["samples"]:
                if sampled_id in y_positions:
                    # Pull connected nodes closer
                    dy = y_positions[song_id] - y_positions[sampled_id]
                    force = -k * 0.1 * dy
                    y_positions[song_id] += force
                    y_positions[sampled_id] -= force

        # Keep y positions within bounds - tightened further to prevent text overflow
        for song_id in y_positions:
            y_positions[song_id] = max(0.45, min(0.75, y_positions[song_id]))

    return y_positions


def visualize_sampling_network(songs_dict, fig=None):
    """
    Create or update a visualization of the sampling network.
    Returns the figure object for further updates.
    """
    # Create or get the figure
    if fig is None:
        fig = go.Figure()

    # Create graph
    G = nx.DiGraph()

    # Add nodes to graph
    for song_id, song in songs_dict.items():
        G.add_node(song_id, **song)

    # Add edges to graph
    for song_id, song in songs_dict.items():
        for sampled_id in song["samples"]:
            G.add_edge(sampled_id, song_id)  # Edge from sampled to sampler

    # Calculate positions
    year_positions = {
        song_id: int(song["year"]) for song_id, song in songs_dict.items()
    }
    y_positions = calculate_y_positions(songs_dict, year_positions)

    # Combine x and y positions
    pos = {
        song_id: (year_positions[song_id], y_positions[song_id])
        for song_id in songs_dict
    }

    # Clear existing traces
    fig.data = []

    # Add edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )
    )

    # Add nodes
    node_x = [pos[song_id][0] for song_id in songs_dict]
    node_y = [pos[song_id][1] for song_id in songs_dict]

    # Calculate node sizes based on how many songs sample this song
    node_sizes = [
        8 + (len(songs_dict[song_id]["sampled_by"]) * 2) for song_id in songs_dict
    ]  # Reduced base size

    # Calculate node colors based on how many samples this song uses
    node_colors = [
        f"rgb({min(255, len(songs_dict[song_id]['samples']) * 50)}, 0, 255)"
        for song_id in songs_dict
    ]

    # Create hover text with just title and artist
    hover_texts = [
        f"{songs_dict[song_id]['title']}<br>{songs_dict[song_id]['artist']}"
        for song_id in songs_dict
    ]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=hover_texts,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line_width=1.5,
                line_color="black",  # Reduced line width
            ),
            textposition="top center",
        )
    )

    # Update layout
    fig.update_layout(
        title="Song Sampling Network",
        xaxis_title="Year",
        yaxis_title="",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=True, zeroline=True, showticklabels=True, title="Year"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[
            dict(
                text="Nodes",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=1.05,  # Moved closer to plot
                align="left",
            )
        ],
        height=600,  # Further reduced height
        width=900,  # Further reduced width
        margin=dict(t=80, b=30, l=30, r=30),  # Reduced margins
    )

    return fig


def initialize_main_song(song_id=109):
    """
    Initialize the songs dictionary with just the main song.

    Args:
        song_id (int): ID of the main song to fetch

    Returns:
        tuple: (songs_dict, base_url, headers, song_data, client)
    """
    # Initialize clients
    client = init_openai_client()
    base_url, headers = init_genius_client()

    # Fetch initial song data
    song_data = get_song_data(song_id, base_url, headers)

    # Initialize songs dictionary with just the main song
    songs_dict = {}
    main_song_id = song_data["id"]
    songs_dict[main_song_id] = {
        "id": main_song_id,
        "title": song_data["title"],
        "artist": song_data["artist_names"],
        "year": song_data["release_date_for_display"].split()[-1],
        "sampled_by": [],
        "samples": [],
    }

    return songs_dict, base_url, headers, song_data, client


def add_samples_layer(songs_dict, base_url, headers, song_data=None, client=None):
    """
    Add a layer of samples to the songs dictionary and update visualization.
    This will process samples for either:
    - The main song (if song_data is provided)
    - All songs that haven't had their samples processed yet (if song_data is None)

    Also analyzes and stores genres for any new songs added to the dictionary.

    Args:
        songs_dict (dict): Current songs dictionary
        base_url (str): Genius API base URL
        headers (dict): Request headers
        song_data (dict, optional): Data of the main song to process samples from
        client (OpenAI client, optional): OpenAI client for genre analysis

    Returns:
        tuple: (updated songs dictionary, updated figure)
    """
    # Keep track of new songs that need genre analysis
    new_songs = set()

    if song_data is not None:
        # Process samples for the main song
        main_song_id = song_data["id"]
        songs_dict[main_song_id]["samples"] = process_song_relationships(
            songs_dict, main_song_id, song_data
        )
        songs_dict[main_song_id]["samples_processed"] = True

        # Add any new songs to the set
        for sample_id in songs_dict[main_song_id]["samples"]:
            if "genres" not in songs_dict[sample_id]:
                new_songs.add(sample_id)
    else:
        # Find songs that haven't had their samples processed
        songs_to_process = [
            song_id
            for song_id, song in songs_dict.items()
            if not song.get("samples_processed", False)
        ]

        for song_id in songs_to_process:
            try:
                # Get song details from Genius API
                song = get_song_data(song_id, base_url, headers)

                # Process relationships
                songs_dict[song_id]["samples"] = process_song_relationships(
                    songs_dict, song_id, song
                )

                # Mark this song as processed
                songs_dict[song_id]["samples_processed"] = True

                # Add any new songs to the set
                for sample_id in songs_dict[song_id]["samples"]:
                    if "genres" not in songs_dict[sample_id]:
                        new_songs.add(sample_id)

                # Add a small delay to respect rate limits
                time.sleep(0.5)

            except Exception as e:
                print(f"Error processing song {song_id}: {str(e)}")
                continue

    # Analyze genres for new songs if client is provided
    if client and new_songs:
        print(f"\nAnalyzing genres for {len(new_songs)} new songs...")
        for song_id in new_songs:
            print(f"\nProcessing genres for: {songs_dict[song_id]['title']}")
            genres = analyze_song_genres(songs_dict[song_id], client)
            songs_dict[song_id]["genres"] = genres
            # Add a small delay to respect rate limits
            time.sleep(1)

    # Create updated visualization
    fig = visualize_sampling_network(songs_dict)

    return songs_dict, fig


def analyze_song_genres(song_data, client):
    """
    Use OpenAI to analyze a song's genres based on its metadata.

    Args:
        song_data (dict): Song data from our songs dictionary
        client: OpenAI client

    Returns:
        list: List of genres identified for the song
    """
    print(f"Starting genre analysis for: {song_data['title']}")

    # Create a prompt that includes relevant song information
    prompt = f"""Analyze this song and identify its primary genres. Consider the following information:
    Title: {song_data["title"]}
    Artist: {song_data["artist"]}
    Year: {song_data["year"]}
    
    Return ONLY a JSON array of genre strings, ordered by relevance. Example: ["Hip Hop", "R&B", "Soul"]
    Focus on musical genres, not themes or moods. Be specific but concise.
    If you're unsure about a genre, omit it rather than guessing."""

    try:
        print("Making API call to OpenAI...")
        start_time = time.time()

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": """You are a music genre expert. Your task is to analyze songs and return their genres.
                    - Return genres in a JSON array format
                    - Order genres by relevance/primary influence
                    - Be specific but concise
                    - Focus on musical genres, not themes or moods
                    - If unsure about a genre, omit it
                    - Common genres include: Hip Hop, R&B, Soul, Rock, Pop, Jazz, Funk, Electronic, etc.
                    - IMPORTANT: Return ONLY a JSON object with a 'genres' key containing an array of strings.""",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=150,  # Limit response length since we only need a short list
        )

        end_time = time.time()
        print(f"API call completed in {end_time - start_time:.2f} seconds")

        # Get the raw response content
        response_content = response.choices[0].message.content
        print(f"Raw API response: {response_content}")

        try:
            # Try to parse the response as JSON
            genres = json.loads(response_content)
            if not isinstance(genres, dict) or "genres" not in genres:
                print(f"Warning: Response is not in expected format. Got: {genres}")
                # Try to handle case where response might be a direct array
                if isinstance(genres, list):
                    return genres
                return []
            print(f"Genres found: {genres.get('genres', [])}")
            return genres.get("genres", [])

        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error: {str(json_err)}")
            print(f"Failed to parse response: {response_content}")
            return []

    except Exception as e:
        print(f"Error analyzing genres for {song_data['title']}: {str(e)}")
        if hasattr(e, "response"):
            print(f"API Response: {e.response}")
        return []


def update_songs_with_genres(songs_dict, client):
    """
    Update the songs dictionary with genre information for each song.

    Args:
        songs_dict (dict): Dictionary containing song information
        client: OpenAI client

    Returns:
        dict: Updated songs dictionary with genre information
    """
    for song_id, song in songs_dict.items():
        print(f"\nProcessing genres for: {song['title']}")
        genres = analyze_song_genres(song, client)
        songs_dict[song_id]["genres"] = genres
        # Add a small delay to respect rate limits
        time.sleep(1)
    return songs_dict


# Example usage:
# 1. Initialize with just the main song
songs_dict, base_url, headers, song_data, client = initialize_main_song()

# 2. Add genre information to the songs dictionary
songs_dict = update_songs_with_genres(songs_dict, client)

# 3. Print the songs dictionary to verify genre data
print("\nSongs dictionary with genres:")
for song_id, song in songs_dict.items():
    print(f"\n{song['title']} by {song['artist']}:")
    print(f"Genres: {song.get('genres', [])}")

# 4. Continue with visualization if desired
main_fig = visualize_sampling_network(songs_dict)
main_fig.show()

# 5. Add first layer of samples (main song) and analyze their genres
songs_dict, first_layer_fig = add_samples_layer(
    songs_dict, base_url, headers, song_data, client
)
first_layer_fig.show()

# 6. Add next layer of samples (all unprocessed songs) and analyze their genres
songs_dict, next_layer_fig = add_samples_layer(
    songs_dict, base_url, headers, client=client
)
next_layer_fig.show()

# 7. Print final songs dictionary with all genres
print("\nFinal songs dictionary with all genres:")
for song_id, song in songs_dict.items():
    print(f"\n{song['title']} by {song['artist']}:")
    print(f"Genres: {song.get('genres', [])}")
    print(f"Samples: {len(song['samples'])}")
    print(f"Sampled by: {len(song['sampled_by'])}")
