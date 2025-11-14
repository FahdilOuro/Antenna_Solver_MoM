## Antenna Solver MoM in Python

### Project Overview

This project aims to develop a Python-based solver for the Method of Moments (MoM) to simulate antennas and their electromagnetic behavior. The initiative is inspired by the Matlab code presented in Sergey Makarov's 2002 book, *Antenna and EM Modeling with Matlab* (AEMM). By translating and enhancing this code, the project provides a modular and object-oriented framework for antenna design and analysis.

### Goals

- **Code Translation**: Convert the Matlab code from Makarov's book into Python while maintaining accuracy and performance.
- **Modular Design**: Structure the code using Object-Oriented Programming (OOP) principles for better modularity, readability, and reusability.
- **Antenna Design Exploration**: Enable users to experiment with antenna designs and analyze their behavior effectively.

### Features

- Translation of the Makarov AEMM algorithms into Python.
- Modular and class-based implementation for easy extensibility.
- Support for triangular meshing and memory-efficient computation.
- Tools to perform antenna design, simulation, and analysis.

### Inspirations and References

- Sergey Makarov's *Antenna and EM Modeling with Matlab* (2002).
- A Python implementation of the MoM solver previously developed by Nguyen Tran Quang Khai (@Khainguyen1349), which serves as a reference and inspiration for this project.

### Project Documentation

In this project, you can also find the **organigramme MoM python.pdf** file, which provides a detailed flowchart outlining the structure and steps of the Method of Moments (MoM) solver implementation in Python.

### Data Files

The antenna mesh data files should be placed in the data/antennas_mesh/ directory.

### How to Get Started

1. **Clone this repository**:

    ```bash
    git clone https://github.com/FahdilOuro/Antenna_Solver_MoM.git
    cd Antenna_Solver_MoM
    ```

2. **Set up a virtual environment**:

    If you're using **Conda**, create an environment with Python 3.12:

    ```bash
    conda create --name antenna_solver python=3.12
    ```

3. **Activate your virtual environment**:

    - For **Conda**:
        ```bash
        conda activate antenna_solver
        ```

4. **Install the required dependencies**:

    - If you have a `requirements.txt`, run:
        ```bash
        pip install -r requirements.txt
        ```
    - Alternatively, you can install the project and dependencies in editable mode with:
        ```bash
        pip install -e .
        ```

5. **Explore the example scripts**:

    Explore the example scripts in the `examples` folder to learn how to use the solver for different antenna designs.

## Jupyter Notebooks

Some notebooks in this project include dynamic plots or interactive outputs that may not display correctly on GitHub.  
To properly view them, please use [nbviewer](https://nbviewer.jupyter.org/).

For example, one can copy the link of the plate_2_antenna notebook and add to viewer to open:

https://nbviewer.org/github/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/receiving_antenna/plate_2_antenna.ipynb

<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/plate_2_antenna_receiving_mode.png" width=50% height=50%>

This is the earliest version of the simulator where one can still find the generation of the antenna mesh, calculation of the RWG basis functions, and the analysis of the antenna performances.

Some result from the simulation of ifa antenna in scattering mode: https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/receiving_antenna/ifa_antenna.ipynb

<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/ifaCurrent_receive.png" width=30% height=30%> <img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/ifaEfield_receive.png" width=35% height=35%> 
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/ifaGainDistribution_receive.png" width=25% height=25%>

## Tutorial

**Coming Soon**

## Contributing

Contributions are welcome! If you find any issues or want to add new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

Special thanks to:

- Sergey Makarov for the foundational work in antenna modeling.
- Nguyen Tran Quang Khai (@Khainguyen1349) for the Python implementation that inspired this project.