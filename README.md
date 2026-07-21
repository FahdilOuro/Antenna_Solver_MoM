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

The solver is a sequential pipeline: each stage saves its output to disk (as `.mat` files) for the next stage to reuse.

```mermaid
flowchart LR
    A["Gmsh geometry<br/>(.msh)"] --> B["rwg1<br/>Points / Triangles / Edges"]
    B --> C["rwg2<br/>Barycentric centers, rho vectors"]
    C --> D["rwg3<br/>Impedance matrix Z"]
    D --> E["Gap source / port excitation<br/>Voltage vector V"]
    E --> F["rwg4<br/>Solve Z*I = V -> current I"]
    F --> G["rwg5<br/>Surface current density J"]
    G --> H["efield1-4<br/>Far/near fields, gain, RCS, S-parameters"]
```

See the Module Map in `CLAUDE.md` for which file implements each stage.

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

## Gallery

All images below come straight out of the current solver, generated from the notebooks they're linked under (GPU-accelerated impedance assembly where the mesh is large enough — see `backend/utils/compute_backend.py`).

### 4-port circularly polarized CubeSat antenna

Notebook: [`backend/antenna_simulation/Nice_Cube/Nice_Cube_radiation.ipynb`](backend/antenna_simulation/Nice_Cube/Nice_Cube_radiation.ipynb) — a CubeSat-shaped structure with 4 solar panels, each fed 90° out of phase to produce circular polarization (RHCP/LHCP).

<p align="center">
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/nice_cube_surface_current.png" width=48%>
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/nice_cube_gain_pattern_3d.png" width=48%>
</p>
<p align="center">
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/nice_cube_rhcp_gain_3d.png" width=48%>
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/nice_cube_lhcp_gain_3d.png" width=48%>
</p>
<p align="center">
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/nice_cube_directivity_polar.png" width=38%>
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/nice_cube_s11_sweep.png" width=58%>
</p>

*Left to right, top to bottom: mesh + surface current at the 4 feed points; total 3D gain pattern; RHCP and LHCP gain patterns (the two lobes reflect the 4-port phasing); elevation/azimuth directivity cuts; S-parameters swept over the 4 ports.*

### Bowtie antenna

Notebook: [`backend/antenna_simulation/radiation_antenna/bowtie_antenna.ipynb`](backend/antenna_simulation/radiation_antenna/bowtie_antenna.ipynb) — a classic single-feed bowtie, resonant near 75 MHz.

<p align="center">
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/bowtie_surface_current.png" width=48%>
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/bowtie_gain_pattern_3d.png" width=48%>
</p>
<p align="center">
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/bowtie_directivity_polar.png" width=38%>
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/bowtie_s11_sweep.png" width=58%>
</p>

*The S11 sweep dips below -10 dB around 65-105 MHz — a clean, well-matched resonance.*

### Multi-turn loop antenna

Notebook: [`backend/antenna_simulation/radiation_antenna/loop4_antenna.ipynb`](backend/antenna_simulation/radiation_antenna/loop4_antenna.ipynb) — a small multi-turn loop (mesh translated directly from Makarov's original MATLAB book examples).

<p align="center">
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/loop4_surface_current.png" width=48%>
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/loop4_gain_pattern_3d.png" width=48%>
</p>
<p align="center">
<img src="https://github.com/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/plot_figures/loop4_directivity_polar.png" width=40%>
</p>

## Jupyter Notebooks

Some notebooks in this project include dynamic Plotly outputs that may not render correctly on GitHub.
To properly view them, please use [nbviewer](https://nbviewer.jupyter.org/) — for example, paste in:

https://nbviewer.org/github/FahdilOuro/Antenna_Solver_MoM/blob/main/backend/antenna_simulation/Nice_Cube/Nice_Cube_radiation.ipynb

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