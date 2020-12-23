import networkx as nx
import pickle as pkl
# filenames
data_name = "mon10k"
data_dir = f"../data/mont/labelled/{data_name}"

G = nx.read_edgelist(f"../data/mont/labelled/{data_name}/data.txt", nodetype=int)

pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp", args="-Goverlap=false")

pkl.dump(pos, open(f"{data_dir}/pos.p", "wb"))
