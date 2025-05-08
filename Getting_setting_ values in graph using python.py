import network as nx
import matplotlib.pyplot as plt
G = nx.Graph()
G.add_node(1, label='A')
G.add_node(2, label='B')
G.add_node(3, label='C')
G.add_edge(1, 2, weight=5)
G.add_edge(2, 3, weight=3)
G.add_edge(1, 3, weight=8)
print("Label of node 1:", G.nodes[1]['label'])            # A
print("Weight between 1 and 2:", G[1][2]['weight']) 
G.nodes[1]['label'] = 'X'
G[1][2]['weight'] = 10
pos = nx.spring_layout(G)
# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1000)
# Draw edges
nx.draw_networkx_edges(G, pos, width=2)
# Draw node labels
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels, font_size=14)
# Draw edge weights
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# Display the graph
plt.title("Graph with Node Labels and Edge Weights")
plt.axis('off')
plt.show()
