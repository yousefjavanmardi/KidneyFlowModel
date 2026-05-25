# KidneyFlowModel

Python code for simulating blood flow, pressure distribution, filtration, and vessel wall compliance effects in a whole-human-kidney vascular network reconstructed from hierarchical HiP-CT imaging.

This repository contains a haemodynamic model of the renal vascular tree. The model represents vessel segments using a network-based formulation based on Poiseuille resistance, with optional vessel wall compliance effects for exploring how pressure-dependent changes in vessel calibre influence renal blood flow, pressure distribution, and filtration.

## Overview

The code was developed for whole-organ kidney haemodynamic modelling using vascular geometries derived from hierarchical phase-contrast tomography. The model combines image-derived vascular structure with pressure boundary conditions at the renal artery and terminal/glomerular level to simulate pressure and flow throughout the kidney vascular tree.

The repository includes:

- a Python implementation of the vascular network solver;
- a graphical user interface for setting pressures and running simulations;
- a 3D viewer for visualising the vascular network and simulation outputs;
- example input files for nodes, vessel elements, terminal/glomerular information, and pressure boundary conditions;
- optional functionality for exploring vessel wall compliance effects.

## Repository contents

```text
KidneyFlowModel/
├── vascular_tree.py                 Core vascular network model and solver
├── kidney_flow_gui.py               Tkinter-based graphical interface
├── kidney_viewer_qt.py              Qt/PyVista-based 3D vascular viewer
├── requirements.txt                 Required Python packages
├── Nodes.txt                        Node coordinates of the vascular network
├── Elements.txt                     Vessel connectivity and geometry
├── ET.txt                           Terminal/glomerular or terminal element data
├── Boundary_Condition_pressure.txt  Example pressure boundary conditions
├── Boundary_Condition_pressure1.txt Alternative pressure boundary-condition file
├── Boundary_Condition_pressure2.txt Alternative pressure boundary-condition file
├── Run_Config.json                  Example run configuration
├── Camera_View.json                 Saved camera view for visualisation
├── Partition_Frame.json             Saved partition/frame settings
└── README.md