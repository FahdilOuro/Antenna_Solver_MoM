Analysis of the relation between the mesh size ; the gap_width and the impedance of the antenna. 

by comparing with the standards, find the best choice for the gap_width with a given mesh size :

file backend/antenna_simulation/gap_voltage_implementation/strip_gap_radiation.ipynb :

if mesh_size == 0.1 | initial_mesh_size = mesh_size:
    if gap_width == 0.1 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Warning: Port 0 — no edges found within gap. Gap width (0.01) may be smaller than mesh size.
        Total input power P_in = 0.5*Re(V·I*) = 0.000000e+00 W
        Warning: Port 0 at [0 0 0] has no feeding edges.

    if gap_width == 0.3 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Warning: Port 0 — no edges found within gap. Gap width (0.03) may be smaller than mesh size.
        Total input power P_in = 0.5*Re(V·I*) = 0.000000e+00 W
        Warning: Port 0 at [0 0 0] has no feeding edges.

    if gap_width == 0.6 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Warning: Port 0 — no edges found within gap. Gap width (0.06) may be smaller than mesh size.
        Total input power P_in = 0.5*Re(V·I*) = 0.000000e+00 W
        Warning: Port 0 at [0 0 0] has no feeding edges.

    if gap_width == 0.65 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Warning: Port 0 — no edges found within gap. Gap width (0.065) may be smaller than mesh size.
        Total input power P_in = 0.5*Re(V·I*) = 0.000000e+00 W
        Warning: Port 0 at [0 0 0] has no feeding edges.

    if gap_width == 0.66 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Warning: Port 0 — no edges found within gap. Gap width (0.066) may be smaller than mesh size.
        Total input power P_in = 0.5*Re(V·I*) = 0.000000e+00 W
        Warning: Port 0 at [0 0 0] has no feeding edges.

    if gap_width == 0.67 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Total input power P_in = 0.5*Re(V·I*) = 1.040738e-02 W
        Port 0: P_in = 1.0407e-02 W  |  Q_in = 5.1091e-03 VAR  |  Z_in = 38.7133+19.0047j Ω

    if gap_width == 0.7 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Total input power P_in = 0.5*Re(V·I*) = 9.534436e-03 W
        Port 0: P_in = 9.5344e-03 W  |  Q_in = 4.6805e-03 VAR  |  Z_in = 42.2578+20.7447j Ω

    if gap_width == 1.1 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Total input power P_in = 0.5*Re(V·I*) = 3.861053e-03 W
        Port 0: P_in = 3.8611e-03 W  |  Q_in = 1.8954e-03 VAR  |  Z_in = 104.3508+51.2266j Ω

    if gap_width == 3 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Total input power P_in = 0.5*Re(V·I*) = 4.603129e-03 W
        Port 0: P_in = 4.6031e-03 W  |  Q_in = 2.4250e-03 VAR  |  Z_in = 85.0247+44.7921j Ω

    if gap_width == 5 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Total input power P_in = 0.5*Re(V·I*) = 4.468251e-03 W
        Port 0: P_in = 4.4683e-03 W  |  Q_in = 2.4560e-03 VAR  |  Z_in = 85.9372+47.2358j Ω

    if gap_width == 10 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Total input power P_in = 0.5*Re(V·I*) = 3.867945e-03 W
        Port 0: P_in = 3.8679e-03 W  |  Q_in = 2.2646e-03 VAR  |  Z_in = 96.2673+56.3635j Ω

    if gap_width == 10 * initial_mesh_size :
        Number of points: 42
        Number of triangles: 40
        Number of edges: 39
        Total input power P_in = 0.5*Re(V·I*) = 3.261837e-04 W
        Port 0: P_in = 3.2618e-04 W  |  Q_in = 1.9710e-04 VAR  |  Z_in = 1122.8672+678.5191j Ω