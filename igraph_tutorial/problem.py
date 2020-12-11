from igraph import Graph, plot

filename = '../data/mont/mon1k.csv'
# # Must be space delimited

g : Graph = Graph()
delimiter = ','
ID = {}
# load graph data from edge list
with open(filename, "r") as f:
    for i, line in enumerate(f):
        split = line.split(delimiter)
        id1 = int(split[0])
        id2 = int(split[1])

        # print("line {}: {} {}".format(i, id1, id2))
        # Retrieve vertices or create a new vertex
        if id1 not in ID:
            # Create new vertex
            v1 = g.add_vertex()
            # Link ID with lib id
            ID[id1] = v1.index
            v1["id"] = id1
            v1_id = ID[id1]
        else:
            v1_id = ID[id1]

        if id2 not in ID:
            v2 = g.add_vertex()
            ID[id2] = v2.index
            v2["id"] = id2
            v2_id = ID[id2]
        else:
            v2_id = ID[id2]

        g.add_edge(v1_id, v2_id)

print("done loading")



# style = {
#     "vertex_size": 2,
#     "vertex_frame_width": .5,
#     "edge_width": 2,
#     "layout": g.layout("rt"),
# }
#
# plot(g, "monspace1k.pdf", **style)

