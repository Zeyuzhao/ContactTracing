from igraph import Graph, plot

g: Graph = Graph.Barabasi(n=100, m=1)

# g.layout_graphopt()
style = {
    "vertex_size": 3,
    "vertex_frame_width": .5,
    "edge_arrow_size": 0.2,
    "edge_arrow_width": 1,
    "edge_width": 1,
    "layout": g.layout("fr")
}

g.vs["label"] = g.hub_score()
plot(g, "new.pdf", **style)
