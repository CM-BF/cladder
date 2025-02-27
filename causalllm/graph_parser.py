import networkx as nx
import matplotlib.pyplot as plt
import re
import io
from graphviz import Digraph
from PIL import Image


def parse_graph_from_text(text):
    """
    Parses graph definition from a structured text and returns a NetworkX DiGraph.
    """
    G = nx.DiGraph()
    edges = []

    lines = text.strip().split("\n")
    node_pattern = re.compile(r"(\w+): (.+)")
    edge_pattern = re.compile(r"(\w+) -> (\w+)")

    nodes = {}

    for line in lines:
        # Match node definitions
        node_match = node_pattern.match(line)
        if node_match:
            node, description = node_match.groups()
            nodes[node] = description
            G.add_node(node, label=description)
            continue

        # Match edges
        edge_match = edge_pattern.match(line)
        if edge_match:
            src, dst = edge_match.groups()
            edges.append((src, dst))

    G.add_edges_from(edges)
    return G, nodes


def draw_graph(G, node_labels):
    """
    Draws a directed graph with labeled nodes.
    """
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G)  # Positioning algorithm
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=12)

    # Draw node labels separately to avoid overlapping
    # labels = {node: f"\n{desc}" for node, desc in node_labels.items()}
    # nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, verticalalignment="center")

    plt.show()

def visualize_langgraph(graph, path):
    """Visualize a langgraph graph using Graphviz and display it with matplotlib."""
    dot = graph.get_graph().draw_mermaid_png()

    # Load the image and display it in matplotlib
    img = Image.open(io.BytesIO(dot))
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.title("LangGraph Visualization")
    plt.savefig(path)
    plt.show()



if __name__ == "__main__":
    # Example input text
    graph_text = """ 
X: Input
A: Find all actions
B: Construct a search algorithm
C: Write code to implement searching
D: Conclude the searching results
Y: Output

X -> A
X -> B
A -> C
B -> C
C -> D
D -> Y
"""

    # Parse the graph
    G, node_labels = parse_graph_from_text(graph_text)

    # Draw the graph
    draw_graph(G, node_labels)