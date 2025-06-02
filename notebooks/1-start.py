# TEST OPENAI WORKS

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You're a helpful assistant."},
        {
            "role": "user",
            "content": "Write a limerick about the Python programming language.",
        },
    ],
)

response = completion.choices[0].message.content
print(response)

# TEST GENIUS API WORKS
import os
import requests

access_token = os.getenv("GENIUS_ACCESS_TOKEN")

if not access_token:
    raise ValueError("GENIUS_ACCESS_TOKEN not found in environment variables")
if access_token:
    print("GENIUS_ACCESS_TOKEN found in environment variables")

base_url = "https://api.genius.com"
# Headers required for authentication
headers = {
    "Authorization": f"Bearer {access_token}",
    "User-Agent": "Trace/1.0",  # It's good practice to identify your application
}

song_id = 109
response = requests.get(f"{base_url}/songs/{song_id}", headers=headers)
response.raise_for_status()  # Raises an HTTPError for bad responses
# Get the song data and store everything under response.song
song_data = response.json()["response"]["song"]

print(f"Title: {song_data['title']}")  # Print some basic info about the song
print(
    [
        song["title"]
        for rel in song_data["song_relationships"]
        if rel["relationship_type"] == "samples"
        for song in rel["songs"]
    ]
)

# STORE MAIN SONG DATA AND SAMPLE DATA
# Extract sample information from song relationships
samples = []
for relationship in song_data["song_relationships"]:
    if relationship["relationship_type"] == "samples":
        for sample in relationship["songs"]:
            # Skip instrumental samples
            if "(Instrumental)" in sample["title"]:
                continue

            sample_info = {
                "id": sample["id"],  # Added song ID
                "title": sample["title"],
                "artist": sample["artist_names"],
                "year": sample["release_date_for_display"].split()[
                    -1
                ],  # Extract year from date string
            }
            samples.append(sample_info)

print("Samples found:", len(samples))
for sample in samples:
    print(
        f"{sample['title']} by {sample['artist']} ({sample['year']}) - ID: {sample['id']}"
    )

main_song = {
    "id": song_data["id"],  # Added song ID
    "title": song_data["title"],
    "artist": song_data["artist_names"],
    "year": song_data["release_date_for_display"].split()[-1],
}

print(main_song)
print(samples)

# TEST THAT I CAN DO NETWORK GRAPHS
import networkx as nx
import plotly.graph_objects as go
import numpy as np

# Create a minimal test graph with just one main song and one sample
G = nx.DiGraph()

# Add one main song
G.add_node("Test Main Song", year=2020, artist="Test Artist", type="main")

# Add one sample
G.add_node("Test Sample Song", year=1990, artist="Sample Artist", type="sample")

# Add one edge
G.add_edge("Test Sample Song", "Test Main Song")

# Create minimal layout
pos = {"Test Main Song": (2020, 0.5), "Test Sample Song": (1990, 0.5)}

# Create minimal visualization
fig = go.Figure()

# Add one edge
fig.add_trace(
    go.Scatter(
        x=[1990, 2020],  # x coordinates for the edge
        y=[0.5, 0.5],  # y coordinates for the edge
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
)

# Add nodes
fig.add_trace(
    go.Scatter(
        x=[1990, 2020],  # x coordinates for the nodes
        y=[0.5, 0.5],  # y coordinates for the nodes
        mode="markers+text",
        hoverinfo="text",
        text=[
            "Test Sample Song<br>Sample Artist<br>1990",
            "Test Main Song<br>Test Artist<br>2020",
        ],
        marker=dict(size=20, color=["blue", "red"], line_width=2),
        textposition="top center",
    )
)

# Show the plot
fig.show()

# TEST SAMPLES WORK IN NETWORK GRAPH
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Create a directed graph (since sampling has a direction - from original to sampled)
G = nx.DiGraph()

# Add the main song as a node
G.add_node(
    main_song["title"],
    year=int(main_song["year"]),
    artist=main_song["artist"],
    type="main",
)

# Add sample songs as nodes
for sample in samples:
    G.add_node(
        sample["title"],
        year=int(sample["year"]),
        artist=sample["artist"],
        type="sample",
    )
    # Add edge from sample to main song (since main song samples from these)
    G.add_edge(sample["title"], main_song["title"])

# Create layout with fixed x-positions based on year
pos = {}
for node in G.nodes():
    year = G.nodes[node]["year"]
    # Use year as x-coordinate, random y-coordinate for initial layout
    pos[node] = (year, np.random.rand())

# Convert to plotly format
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]

# Create the figure
fig = go.Figure()

# Add edges
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
fig.add_trace(
    go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[
            f"{node}<br>{G.nodes[node]['artist']}<br>{G.nodes[node]['year']}"
            for node in G.nodes()
        ],
        marker=dict(
            size=20,
            color=[
                "red" if G.nodes[node]["type"] == "main" else "blue"
                for node in G.nodes()
            ],
            line_width=2,
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
)

# Show the plot
fig.show()


# ADD THE SAMPLES OF THE SAMPLE TO THE NETWORK GRAPH
# Function to get samples for a song
def get_samples_for_song(song_id):
    try:
        response = requests.get(f"{base_url}/songs/{song_id}", headers=headers)
        response.raise_for_status()
        song_data = response.json()["response"]["song"]

        song_samples = []
        for relationship in song_data["song_relationships"]:
            if relationship["relationship_type"] == "samples":
                for sample in relationship["songs"]:
                    if "(Instrumental)" in sample["title"]:
                        continue

                    sample_info = {
                        "id": sample["id"],
                        "title": sample["title"],
                        "artist": sample["artist_names"],
                        "year": sample["release_date_for_display"].split()[-1],
                    }
                    song_samples.append(sample_info)
        return song_samples
    except requests.exceptions.RequestException as e:
        print(f"Error fetching samples for song {song_id}: {e}")
        return []


# Get samples for each sample song
for sample in samples:
    print(f"\nFetching samples for: {sample['title']} (ID: {sample['id']})")
    sample_of_samples = get_samples_for_song(sample["id"])

    # Add each sample-of-sample to the graph
    for sub_sample in sample_of_samples:
        # Add node if it doesn't exist
        if sub_sample["title"] not in G:
            G.add_node(
                sub_sample["title"],
                year=int(sub_sample["year"]),
                artist=sub_sample["artist"],
                type="sub_sample",
                id=sub_sample["id"],  # Add this line to store the ID
            )
            # Add edge from sub-sample to sample
            G.add_edge(sub_sample["title"], sample["title"])

            # Update position for new node
            pos[sub_sample["title"]] = (int(sub_sample["year"]), np.random.rand())

            # Add new edge trace
            x0, y0 = pos[sub_sample["title"]]
            x1, y1 = pos[sample["title"]]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    line=dict(width=0.5, color="#888"),
                    hoverinfo="none",
                    mode="lines",
                )
            )

            # Add new node trace
            fig.add_trace(
                go.Scatter(
                    x=[x0],
                    y=[y0],
                    mode="markers+text",
                    hoverinfo="text",
                    text=[
                        f"{sub_sample['title']}<br>{sub_sample['artist']}<br>{sub_sample['year']}"
                    ],
                    marker=dict(
                        size=15,  # Slightly smaller than main samples
                        color="green",  # Different color for sub-samples
                        line_width=2,
                    ),
                    textposition="top center",
                )
            )

            print(f"Added sub-sample: {sub_sample['title']} ({sub_sample['year']})")

# Update layout and show final plot
fig.update_layout(
    title="Song Sampling Network",
    xaxis_title="Year",
    yaxis_title="",
    showlegend=False,
    height=800,  # Make plot taller to accommodate more nodes
)
fig.show()

# Print some statistics
print("\nNetwork Statistics:")
print(f"Total nodes: {G.number_of_nodes()}")
print(f"Total edges: {G.number_of_edges()}")
print(f"Main song: {main_song['title']}")
print(f"Direct samples: {len(samples)}")
print(
    f"Sub-samples: {G.number_of_nodes() - len(samples) - 1}"
)  # Subtract main song and direct samples

# ADD MORE SAMPLES FROM SAMPLES
