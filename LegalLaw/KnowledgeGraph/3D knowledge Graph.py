import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objs as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def calculate_similarity(documents):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate cosine similarity between all pairs of documents
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

def main():
    st.title("3D Knowledge Graph")

    # Read data from CSV
    df = pd.read_csv("top_10.csv")
    texts = df["text"].tolist()
    names = df["name"].tolist()

    # Process text data (take only 50 words from each text)
    documents = []
    node_names = []  # Store node names
    for text, name in zip(texts, names):
        # Use regex to split text into words
        words = re.findall(r'\b\w+\b', text)
        # Take first 50 words
        selected_words = words[:50]
        document = " ".join(selected_words)
        documents.append(document)
        node_names.append(name)

    # Calculate cosine similarity between all pairs of documents
    similarity_matrix = calculate_similarity(documents)

    # Calculate average similarity for each document
    avg_similarities = np.mean(similarity_matrix, axis=1)

    # Sort documents based on average similarity in descending order
    ranked_indices = np.argsort(avg_similarities)[::-1]

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    top_3_results = []
    for i, idx in enumerate(ranked_indices):
        document = documents[idx]
        similarity = avg_similarities[idx]
        name = node_names[idx]  # Get node name
        G.add_node(name, label=document, avg_similarity=similarity, rank=i+1)
        
        # Store top 3 results
        if i < 3:
            top_3_results.append(f"Rank {i+1}: {document}")

    # Add edges to the graph based on similarity scores
    for i in range(len(documents)):
        for j in range(len(documents)):
            if i != j:
                similarity = similarity_matrix[i][j]
                source_name = node_names[i]
                target_name = node_names[j]
                G.add_edge(source_name, target_name, weight=similarity)

    # Position nodes using Fruchterman-Reingold force-directed algorithm
    pos = nx.spring_layout(G, dim=3)

    # Create edge trace
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(f"Rank: {G.nodes[node]['rank']} \n\n {node}")
        rank = G.nodes[node]['rank']
        if rank == 1:
            node_color.append('red')
        elif rank == 2:
            node_color.append('green')
        elif rank == 3:
            node_color.append('white')
        else:
            node_color.append('blue')

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(symbol='circle',
                    size=10,
                    color=node_color,
                    colorscale='Viridis',
                    colorbar=dict(title='Rank Color'),
                    line=dict(color='rgb(50,50,50)', width=0.5)
                    ),
        text=node_text,
        textposition="bottom center"
    )

    # Plot the 3D knowledge graph
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='3D Knowledge Graph',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        showarrow=False,
                        text="",
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    scene=dict(
                        xaxis=dict(title='', showgrid=False, showticklabels=False, zeroline=False),
                        yaxis=dict(title='', showgrid=False, showticklabels=False, zeroline=False),
                        zaxis=dict(title='', showgrid=False, showticklabels=False, zeroline=False),
                        camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=-1.25, y=-1.25, z=0.1))
                    )
                ),
                frames=[go.Frame(layout=go.Layout(scene_camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=np.cos(theta) * 2.5, y=np.sin(theta) * 2.5, z=0.1)))) for theta in np.linspace(0, 2 * np.pi, 100)]
                )
    # Update the figure's layout to enable animations
    fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                    buttons=[dict(label='Play',
                                                  method='animate',
                                                  args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, transition=dict(duration=0, easing='linear'))]),
                                             dict(label='Pause',
                                                  method='animate',
                                                  args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))])])])

    # Display the animated graph
    st.plotly_chart(fig)

    # Display top 3 results
    st.subheader("Top 3 results are:")
    for result in top_3_results:
        st.write(result)

if _name_ == "_main_":
    main()
