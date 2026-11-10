"""
Entry point of Antenna Solver MoM.

There is no graphical interface yet: the front end will live in interface/
(see interface/README.md). The previous Qt/QML entry point imported a
`frontend` package that no longer exists; it is kept in the git history
(commit 8704b65) for when the interface is built.

The solver is used from Python, typically in a Jupyter notebook:

    from solver.utils.file_path import setup_save_file_paths
    from solver.src.radiation_algorithm.radiation_algorithm import radiation_algorithm

    path = setup_save_file_paths("my_antenna")      # expects data/gmsh_files/my_antenna.msh
    radiation_algorithm(path, frequency=75e6, feed_point=[0, 0, 0],
                        excitation_unit_vector='y', gap_width=0.05)

Examples: solver/antenna_simulation/ (one notebook per antenna).
"""


def main():
    print(__doc__)


if __name__ == "__main__":
    main()
