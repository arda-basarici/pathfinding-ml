"""pathfinding-ml — learning a search heuristic and measuring what it costs.

Package layout (each module has one job):

    maze/        grid representation and maze generation
    search/      search algorithms (heuristic-pluggable) + instrumentation
    data/        ground-truth labels, feature extraction, dataset assembly
    model/       baseline + trained cost-to-go regressor
    evaluation/  heuristic quality and downstream search benchmarks

Data flows one direction:
    generate mazes -> label (true cost-to-go) -> features -> dataset
    -> train -> evaluate.
"""
