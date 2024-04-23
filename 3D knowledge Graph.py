import numpy as np
import networkx as nx
import plotly.graph_objs as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(documents):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate cosine similarity between all pairs of documents
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

def main():
    st.title("3D Knowledge Graph")

    # Sample documents (you should replace this with your data)
    documents = [
        "Loving these vintage springs on my vintage strat. T",
        "Works great as a guitar bench mat. Not rugged enough for abuse but if you take care of it",
        "kinda flimsy but oit does the job",
        "Great price and good quality.  It didn't quite match the radius of my sound hole but it was close enough.",
        "Loving these vintage springs on my vintage strat. They have a good tension and great stability.",
        "Great price and good quality.  It didn't quite match the radius of my sound hole but it was close enough.",
        "Great nylon strings, just as expected. They worked just fine on my daughter's mini classica ",
        "kinda flimsy but oit does the job",
        "You really cant beat it for the price, however, I would not use it for really heavy lights."
    ]

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
        G.add_node(i, label=document, avg_similarity=similarity, rank=i+1)
        
        # Store top 3 results
        if i < 3:
            top_3_results.append(f"Rank {i+1}: {document}")

    # Add edges to the graph based on similarity scores
    for i in range(len(documents)):
        for j in range(len(documents)):
            if i != j:
                similarity = similarity_matrix[i][j]
                G.add_edge(i, j, weight=similarity)

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
        node_text.append(f"Rank: {G.nodes[node]['rank']}\n\n{G.nodes[node]['label']}")
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