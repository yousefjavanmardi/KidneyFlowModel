import json
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, BooleanVar, StringVar, Tk, filedialog, messagebox, ttk

import numpy as np

from vascular_tree import (
    adjust_terminal_element_pressure,
    adjust_terminal_element_pressure_per_vessel,
    apply_inlet_waveform,
    calculate_gfr,
    calculate_gfr_per_glomerulus,
    plot_flowrate_vs_strahler,
    plot_pressure_vs_strahler,
    sample_glomerular_kf,
    sample_glomerular_resistance,
    add_glomerular_resistance_per_vessel,
    solve_compliance,
    solve_unit,
    strahler_order,
    vessel_resistance,
)


def load_text_matrix(filename, usecols=None, skiprows=0):
    try:
        return np.loadtxt(filename, delimiter=",", usecols=usecols, skiprows=skiprows, ndmin=2)
    except ValueError:
        return np.loadtxt(filename, usecols=usecols, skiprows=skiprows, ndmin=2)


def save_vector(filename, values):
    np.savetxt(filename, np.asarray(values), fmt="%.12e")


def run_single_case(case_name, elements_path, params, log, output_dir):
    nodes_path = Path(params["nodes_path"])
    bc_path = Path(params["bc_path"])
    et_path = Path(params["et_path"]) if params["et_path"] else None
    strahler_file_path = Path(params["strahler_file_path"]) if params["strahler_file_path"] else None
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[{case_name}] Reading Elements file...")
    elements = load_text_matrix(elements_path, usecols=(0, 1, 2, 3))

    if nodes_path:
        if not nodes_path.exists():
            raise FileNotFoundError(f"Nodes file does not exist: {nodes_path}")
        log(f"[{case_name}] Nodes file found. It will be used by the plotting script.")

    et_data = None
    if et_path:
        if not et_path.exists():
            raise FileNotFoundError(f"ET file does not exist: {et_path}")
        log(f"[{case_name}] Reading ET file...")
        et_data = load_text_matrix(et_path)

    log(f"[{case_name}] Reading inlet boundary condition...")
    boundary_condition = params["boundary_condition"]
    bc_table = load_text_matrix(bc_path)

    n_time = int(params["n_time"])
    p_ef = float(params["p_ef"])
    r_gc_mean = float(params["r_gc_mean"])
    r_gc_std = float(params["r_gc_std"])
    blood_viscosity = float(params["blood_viscosity"])
    p_bowman = float(params["p_bowman"])
    p_osmotic = float(params["p_osmotic"])
    kf = float(params["kf"])
    kf_std = float(params["kf_std"])
    include_compliance = bool(params["include_compliance"])
    calculate_sgfr = bool(params["calculate_sgfr"])
    save_visualisation_data = bool(params["save_visualisation_data"])

    if params["strahler_mode"] == "file":
        if strahler_file_path is None or not strahler_file_path.exists():
            raise FileNotFoundError("Selected Strahler-order file does not exist.")
        log(f"[{case_name}] Reading Strahler order from file...")
        so = np.loadtxt(strahler_file_path, dtype=np.int64, ndmin=1)
        if so.shape[0] != elements.shape[0]:
            raise ValueError("Strahler-order file must contain one value per vessel.")
    else:
        log(f"[{case_name}] Calculating Strahler order...")
        so = strahler_order(elements[:, :2])
    if save_visualisation_data:
        np.savetxt(output_dir / "StrahlerOrder.txt", so, fmt="%d")

    log(f"[{case_name}] Calculating vessel resistance...")
    resistance, compliance, radius_m, length_m, effective_viscosity = vessel_resistance(
        elements,
        blood_viscosity,
        et_data=et_data,
    )
    glomerular_resistance = sample_glomerular_resistance(so, r_gc_mean, r_gc_std)
    glomerular_kf = sample_glomerular_kf(so, kf, kf_std)
    modified_resistance, glomerular_resistance_si = add_glomerular_resistance_per_vessel(
        resistance,
        glomerular_resistance,
    )
    vessel_characteristics = np.column_stack(
        (
            radius_m,
            length_m,
            effective_viscosity,
            resistance,
            modified_resistance,
            compliance,
            glomerular_resistance,
            glomerular_resistance_si,
        )
    )
    if save_visualisation_data:
        np.savetxt(
            output_dir / "Vessel_Characteristics.txt",
            vessel_characteristics,
            fmt="%.12e",
            delimiter=",",
            header="Radius_m,Length_m,Effective_Viscosity_Pa_s,Resistance_Pa_s_per_m3,Modified_Resistance_Pa_s_per_m3,Compliance_m3_per_Pa,Glomerular_Resistance_mmHg_min_per_mL,Glomerular_Resistance_Pa_s_per_m3",
            comments="",
        )

    log(f"[{case_name}] Solving unit response...")
    pressure_unit, flowrate_unit, equivalent_resistance, flow_fraction = solve_unit(
        elements[:, :2],
        modified_resistance,
        boundary_condition=boundary_condition,
        inlet_value=1.0,
        outlet_pressure=0.0,
    )

    if include_compliance:
        log(f"[{case_name}] Solving compliance-aware transient response...")
        (
            times,
            pressure_t,
            flowrate_t,
            element_pressure_t,
            inlet_amplitude,
            _pressure_unit_transient,
            _flowrate_unit_transient,
            _equivalent_resistance_transient,
            _flow_fraction_transient,
        ) = solve_compliance(
            elements[:, :2],
            modified_resistance,
            compliance,
            boundary_condition,
            bc_table,
            n_time,
            p_ef=p_ef,
        )
    else:
        log(f"[{case_name}] Interpolating inlet waveform and calculating time solution...")
        times, pressure_t, flowrate_t, element_pressure_t, inlet_amplitude = apply_inlet_waveform(
            elements[:, :2],
            pressure_unit,
            flowrate_unit,
            boundary_condition,
            bc_table,
            n_time,
            p_ef=p_ef,
        )

    if save_visualisation_data:
        save_vector(output_dir / "Pressure_Unit.txt", pressure_unit)
        save_vector(output_dir / "Flowrate_Unit.txt", flowrate_unit)
        save_vector(output_dir / "Equivalent_Resistance.txt", equivalent_resistance)
        save_vector(output_dir / "Flow_Fraction.txt", flow_fraction)

    afferent_element_pressure_t = adjust_terminal_element_pressure(
        element_pressure_t,
        flowrate_t,
        so,
        r_gc_mean,
    )
    if r_gc_std != 0.0:
        afferent_element_pressure_t = adjust_terminal_element_pressure_per_vessel(
            element_pressure_t,
            flowrate_t,
            so,
            glomerular_resistance,
        )

    if save_visualisation_data:
        log(f"[{case_name}] Saving time-dependent results...")
        save_vector(output_dir / "Solution_Times.txt", times)
        save_vector(output_dir / "Inlet_Amplitude.txt", inlet_amplitude)
        np.savetxt(output_dir / "Pressure_Nodes_Time.txt", pressure_t, fmt="%.12e", delimiter=",")
        np.savetxt(output_dir / "Flowrate_Elements_Time.txt", flowrate_t, fmt="%.12e", delimiter=",")
        np.savetxt(output_dir / "Pressure_Elements_Time.txt", element_pressure_t, fmt="%.12e", delimiter=",")
        np.savetxt(
            output_dir / "Pressure_Elements_Afferent_Time.txt",
            afferent_element_pressure_t,
            fmt="%.12e",
            delimiter=",",
        )

    log(f"[{case_name}] Saving flow-rate versus Strahler-order summary...")
    flowrate_plot_path = output_dir / "Flowrate_vs_Strahler.svg" if save_visualisation_data else output_dir / "_tmp_Flowrate_vs_Strahler.svg"
    orders, mean_flow_by_order, std_flow_by_order = plot_flowrate_vs_strahler(
        flowrate_t,
        so,
        flowrate_plot_path,
    )
    np.savetxt(
        output_dir / "Flowrate_vs_Strahler.txt",
        np.column_stack((orders, mean_flow_by_order, std_flow_by_order)),
        fmt=["%d", "%.12e", "%.12e"],
        delimiter=",",
        header="Strahler_Order,Mean_Flowrate_mL_per_min,Std_Flowrate_mL_per_min",
        comments="",
    )
    if not save_visualisation_data and flowrate_plot_path.exists():
        flowrate_plot_path.unlink()

    log(f"[{case_name}] Saving pressure versus Strahler-order summary...")
    pressure_plot_path = output_dir / "Pressure_vs_Strahler.svg" if save_visualisation_data else output_dir / "_tmp_Pressure_vs_Strahler.svg"
    orders, mean_pressure_by_order, std_pressure_by_order = plot_pressure_vs_strahler(
        element_pressure_t,
        so,
        pressure_plot_path,
    )
    np.savetxt(
        output_dir / "Pressure_vs_Strahler.txt",
        np.column_stack((orders, mean_pressure_by_order, std_pressure_by_order)),
        fmt=["%d", "%.12e", "%.12e"],
        delimiter=",",
        header="Strahler_Order,Mean_Pressure_mmHg,Std_Pressure_mmHg",
        comments="",
    )
    if not save_visualisation_data and pressure_plot_path.exists():
        pressure_plot_path.unlink()

    np.savetxt(
        output_dir / "RBF_Time.txt",
        np.column_stack((times, flowrate_t[0, :])),
        fmt="%.12e",
        delimiter=",",
        header="Time,RBF_mL_per_min",
        comments="",
    )

    n_glomeruli = int(np.count_nonzero(so == 1))
    if calculate_sgfr:
        log(f"[{case_name}] Calculating GFR...")
        if r_gc_std == 0.0:
            if kf_std == 0.0:
                gfr_t, n_glomeruli = calculate_gfr(
                    flowrate_t,
                    so,
                    r_gc_mean,
                    p_ef,
                    p_bowman=p_bowman,
                    p_osmotic=p_osmotic,
                    kf=kf,
                )
            else:
                gfr_t, n_glomeruli = calculate_gfr_per_glomerulus(
                    flowrate_t,
                    so,
                    glomerular_resistance,
                    p_ef,
                    p_bowman=p_bowman,
                    p_osmotic=p_osmotic,
                    kf=glomerular_kf,
                )
        else:
            gfr_t, n_glomeruli = calculate_gfr_per_glomerulus(
                flowrate_t,
                so,
                glomerular_resistance,
                p_ef,
                p_bowman=p_bowman,
                p_osmotic=p_osmotic,
                kf=glomerular_kf if kf_std != 0.0 else kf,
            )
        np.savetxt(
            output_dir / "GFR_Time.txt",
            np.column_stack((times, gfr_t)),
            fmt="%.12e",
            delimiter=",",
            header="Time,GFR_mL_per_min",
            comments="",
        )

    if save_visualisation_data:
        config = dict(params)
        config["glomerular_resistance_mean_si"] = float(r_gc_mean) * 8.0e9
        config["n_glomeruli"] = n_glomeruli
        with open(output_dir / "Run_Config.json", "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)

    log(f"[{case_name}] Done.")


def run_pipeline(params, log):
    elements_path = Path(params["elements_path"])
    base_output_dir = Path(params["output_dir"]) / "OUTPUTS"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    if elements_path.is_dir():
        batch_params = dict(params)
        batch_params["save_visualisation_data"] = False
        if batch_params.get("strahler_mode") == "file":
            raise ValueError("Batch mode currently requires 'Calculate Strahler order' instead of 'Read from file'.")

        element_files = sorted(path for path in elements_path.glob("*.txt") if path.is_file())
        if not element_files:
            raise FileNotFoundError(f"No .txt files were found in Elements folder: {elements_path}")

        log("Batch mode detected. 'Save visualisation data' is turned off automatically.")
        log(f"Found {len(element_files)} element files.")
        for element_file in element_files:
            case_output_dir = base_output_dir / element_file.stem
            run_single_case(element_file.stem, element_file, batch_params, log, case_output_dir)
    else:
        run_single_case(elements_path.stem, elements_path, params, log, base_output_dir)


class KidneyFlowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kidney 0D Flow Solver")
        self.root.geometry("760x620")

        self.elements_path = StringVar()
        self.nodes_path = StringVar()
        self.bc_path = StringVar()
        self.et_path = StringVar()
        self.output_dir = StringVar(value=str(Path.cwd()))
        self.strahler_mode = StringVar(value="calculate")
        self.strahler_file_path = StringVar()
        self.boundary_condition = StringVar(value="Pressure")
        self.n_time = StringVar(value="10")
        self.p_ef = StringVar(value="50")
        self.r_gc_mean = StringVar(value="7014.7")
        self.r_gc_std = StringVar(value="4094.0")
        self.blood_viscosity = StringVar(value="0.001")
        self.p_bowman = StringVar(value="15")
        self.p_osmotic = StringVar(value="34.75")
        self.kf = StringVar(value="27.5e-6")
        self.kf_std = StringVar(value="0")
        self.include_compliance = BooleanVar(value=False)
        self.save_visualisation_data = BooleanVar(value=True)
        self.calculate_sgfr = BooleanVar(value=True)

        self.build_ui()
        self.set_default_paths()

    def set_default_paths(self):
        defaults = [
            (self.elements_path, Path("Elements.txt")),
            (self.nodes_path, Path("Nodes.txt")),
            (self.bc_path, Path("Boundary_Condition_pressure.txt")),
            (self.et_path, Path("ET.txt")),
            (self.strahler_file_path, Path("OUTPUTS") / "StrahlerOrder.txt"),
        ]
        for variable, path in defaults:
            if path.exists():
                variable.set(str(path.resolve()))

    def build_ui(self):
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill=BOTH, expand=True)

        input_frame = ttk.LabelFrame(main, text="Input files", padding=8)
        input_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Elements file").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(input_frame, textvariable=self.elements_path, width=60).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        elements_button_bar = ttk.Frame(input_frame)
        elements_button_bar.grid(row=0, column=2, sticky="e", pady=6)
        ttk.Button(elements_button_bar, text="File", command=self.pick_elements_file).pack(side=LEFT)
        ttk.Button(elements_button_bar, text="Folder", command=self.pick_elements_folder).pack(side=LEFT, padx=(6, 0))
        self.add_file_row(input_frame, 1, "Nodes file", self.nodes_path, self.pick_nodes)
        self.add_file_row(input_frame, 2, "Young's modulus (Pa) and thickness (um)", self.et_path, self.pick_et)
        self.add_file_row(input_frame, 3, "Output folder", self.output_dir, self.pick_output_dir)

        bc_frame = ttk.LabelFrame(main, text="Boundary conditions", padding=8)
        bc_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        bc_frame.columnconfigure(1, weight=1)
        bc_frame.columnconfigure(3, weight=1)

        ttk.Label(bc_frame, text="Boundary condtion type at inlet").grid(row=0, column=0, sticky="w", pady=6)
        bc_menu = ttk.Combobox(
            bc_frame,
            textvariable=self.boundary_condition,
            values=("Pressure", "Flowrate"),
            state="readonly",
            width=18,
        )
        bc_menu.grid(row=0, column=1, sticky="w", padx=8, pady=6)

        ttk.Label(bc_frame, text="Pressure at outlets (mmHg)").grid(row=0, column=2, sticky="w", pady=6)
        ttk.Entry(bc_frame, textvariable=self.p_ef, width=20).grid(row=0, column=3, sticky="w", padx=8, pady=6)

        ttk.Label(bc_frame, text="Boundary condition file").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(bc_frame, textvariable=self.bc_path, width=72).grid(row=1, column=1, columnspan=2, sticky="ew", padx=8, pady=6)
        ttk.Button(bc_frame, text="Browse", command=self.pick_bc).grid(row=1, column=3, sticky="e", pady=6)

        ttk.Label(main, text="Blood viscosity (Pa.s)").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(main, textvariable=self.blood_viscosity, width=20).grid(row=2, column=1, sticky="w", padx=8, pady=6)
        ttk.Label(main, text="Number of time frames").grid(row=2, column=2, sticky="w", pady=6)
        ttk.Entry(main, textvariable=self.n_time, width=20).grid(row=2, column=3, sticky="w", padx=8, pady=6)

        ttk.Checkbutton(
            main,
            text="Include compliance effect",
            variable=self.include_compliance,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Checkbutton(
            main,
            text="Save visualisation data",
            variable=self.save_visualisation_data,
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=6)

        strahler_frame = ttk.LabelFrame(main, text="Strahler order", padding=8)
        strahler_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        strahler_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            strahler_frame,
            text="Calculate Strahler order",
            variable=self.strahler_mode,
            value="calculate",
            command=self.update_strahler_state,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=4)

        ttk.Radiobutton(
            strahler_frame,
            text="Read from file",
            variable=self.strahler_mode,
            value="file",
            command=self.update_strahler_state,
        ).grid(row=1, column=0, sticky="w", pady=4)
        self.strahler_file_entry = ttk.Entry(strahler_frame, textvariable=self.strahler_file_path, width=60)
        self.strahler_file_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(strahler_frame, text="Browse", command=self.pick_strahler_file).grid(row=1, column=2, sticky="e", pady=4)

        sgfr_frame = ttk.LabelFrame(main, text="Calculate sGFR", padding=8)
        sgfr_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 4))
        sgfr_frame.columnconfigure(1, weight=1)
        sgfr_frame.columnconfigure(3, weight=1)

        ttk.Checkbutton(
            sgfr_frame,
            text="Enable sGFR calculation",
            variable=self.calculate_sgfr,
            command=self.update_sgfr_state,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        ttk.Label(sgfr_frame, text="Glomerular resistance (mmHg.min/mL)").grid(row=1, column=0, sticky="w", pady=6)
        self.r_gc_entry = ttk.Entry(sgfr_frame, textvariable=self.r_gc_mean, width=18)
        self.r_gc_entry.grid(row=1, column=1, sticky="w", padx=8, pady=6)
        ttk.Label(sgfr_frame, text="Pressure at Bowman's space (mmHg)").grid(row=1, column=2, sticky="w", pady=6)
        self.p_bowman_entry = ttk.Entry(sgfr_frame, textvariable=self.p_bowman, width=18)
        self.p_bowman_entry.grid(row=1, column=3, sticky="w", padx=8, pady=6)

        ttk.Label(sgfr_frame, text="Standard deviation (mmHg.min/mL)").grid(row=2, column=0, sticky="w", pady=6)
        self.r_gc_std_entry = ttk.Entry(sgfr_frame, textvariable=self.r_gc_std, width=18)
        self.r_gc_std_entry.grid(row=2, column=1, sticky="w", padx=8, pady=6)
        ttk.Label(sgfr_frame, text="Oncotic pressure (mmHg)").grid(row=2, column=2, sticky="w", pady=6)
        self.p_osmotic_entry = ttk.Entry(sgfr_frame, textvariable=self.p_osmotic, width=18)
        self.p_osmotic_entry.grid(row=2, column=3, sticky="w", padx=8, pady=6)

        ttk.Label(sgfr_frame, text="Kf (mL/(min.mmHg))").grid(row=3, column=0, sticky="w", pady=6)
        self.kf_entry = ttk.Entry(sgfr_frame, textvariable=self.kf, width=18)
        self.kf_entry.grid(row=3, column=1, sticky="w", padx=8, pady=6)
        ttk.Label(sgfr_frame, text="Kf standard deviation (mL/(min.mmHg))").grid(row=3, column=2, sticky="w", pady=6)
        self.kf_std_entry = ttk.Entry(sgfr_frame, textvariable=self.kf_std, width=18)
        self.kf_std_entry.grid(row=3, column=3, sticky="w", padx=8, pady=6)

        button_row = ttk.Frame(main)
        button_row.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(16, 8))
        self.run_button = ttk.Button(button_row, text="Run Solver", command=self.start_solver)
        self.run_button.pack(side=LEFT)
        ttk.Button(button_row, text="Quit", command=self.root.destroy).pack(side=RIGHT)

        ttk.Label(main, text="Log").grid(row=7, column=0, sticky="w", pady=(8, 4))
        self.log_box = ttk.Treeview(main, columns=("message",), show="headings", height=12)
        self.log_box.heading("message", text="Message")
        self.log_box.column("message", width=700, anchor="w")
        self.log_box.grid(row=8, column=0, columnspan=4, sticky="nsew")

        main.columnconfigure(1, weight=1)
        main.columnconfigure(3, weight=1)
        main.rowconfigure(8, weight=1)
        self.update_strahler_state()
        self.update_sgfr_state()

    def add_file_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=variable, width=60).grid(row=row, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="e", pady=6)

    def update_sgfr_state(self):
        state = "normal" if self.calculate_sgfr.get() else "disabled"
        for entry in (
            self.r_gc_entry,
            self.r_gc_std_entry,
            self.p_bowman_entry,
            self.p_osmotic_entry,
            self.kf_entry,
            self.kf_std_entry,
        ):
            entry.config(state=state)

    def update_strahler_state(self):
        state = "normal" if self.strahler_mode.get() == "file" else "disabled"
        self.strahler_file_entry.config(state=state)

    def pick_elements_file(self):
        filename = filedialog.askopenfilename(title="Select Elements file")
        if filename:
            self.elements_path.set(filename)

    def pick_elements_folder(self):
        directory = filedialog.askdirectory(title="Select Elements folder")
        if directory:
            self.elements_path.set(directory)
            self.save_visualisation_data.set(False)

    def pick_nodes(self):
        filename = filedialog.askopenfilename(title="Select Nodes file")
        if filename:
            self.nodes_path.set(filename)

    def pick_bc(self):
        filename = filedialog.askopenfilename(title="Select boundary condition file")
        if filename:
            self.bc_path.set(filename)

    def pick_et(self):
        filename = filedialog.askopenfilename(title="Select ET file")
        if filename:
            self.et_path.set(filename)

    def pick_strahler_file(self):
        filename = filedialog.askopenfilename(title="Select Strahler-order file")
        if filename:
            self.strahler_file_path.set(filename)

    def pick_output_dir(self):
        directory = filedialog.askdirectory(title="Select output folder")
        if directory:
            self.output_dir.set(directory)

    def log(self, message):
        self.root.after(0, self._append_log, message)

    def _append_log(self, message):
        self.log_box.insert("", END, values=(message,))
        children = self.log_box.get_children()
        if children:
            self.log_box.see(children[-1])

    def collect_params(self):
        if not self.elements_path.get():
            raise ValueError("Please select an Elements file.")
        if not self.nodes_path.get():
            raise ValueError("Please select a Nodes file.")
        if not self.bc_path.get():
            raise ValueError("Please select a boundary condition file.")
        if not self.et_path.get():
            raise ValueError("Please select an ET file.")
        elements_path = Path(self.elements_path.get())
        if not elements_path.exists():
            raise ValueError("The selected Elements path does not exist.")
        if self.strahler_mode.get() == "file" and not self.strahler_file_path.get():
            raise ValueError("Please select a Strahler-order file or choose Calculate Strahler order.")
        if int(self.n_time.get()) < 2:
            raise ValueError("Number of time frames must be at least 2.")

        save_visualisation_data = self.save_visualisation_data.get()
        if elements_path.is_dir():
            save_visualisation_data = False

        return {
            "elements_path": self.elements_path.get(),
            "nodes_path": self.nodes_path.get(),
            "bc_path": self.bc_path.get(),
            "et_path": self.et_path.get(),
            "output_dir": self.output_dir.get(),
            "strahler_mode": self.strahler_mode.get(),
            "strahler_file_path": self.strahler_file_path.get(),
            "boundary_condition": self.boundary_condition.get(),
            "n_time": self.n_time.get(),
            "p_ef": self.p_ef.get(),
            "r_gc_mean": self.r_gc_mean.get(),
            "r_gc_std": self.r_gc_std.get(),
            "blood_viscosity": self.blood_viscosity.get(),
            "p_bowman": self.p_bowman.get(),
            "p_osmotic": self.p_osmotic.get(),
            "kf": self.kf.get(),
            "kf_std": self.kf_std.get(),
            "include_compliance": self.include_compliance.get(),
            "save_visualisation_data": save_visualisation_data,
            "calculate_sgfr": self.calculate_sgfr.get(),
        }

    def start_solver(self):
        try:
            params = self.collect_params()
        except Exception as exc:
            messagebox.showerror("Input error", str(exc))
            return

        self.run_button.config(state="disabled")
        if Path(params["elements_path"]).is_dir():
            self.save_visualisation_data.set(False)
        self._append_log("Starting solver...")
        worker = threading.Thread(target=self.run_solver_worker, args=(params,), daemon=True)
        worker.start()

    def run_solver_worker(self, params):
        success = False
        try:
            run_pipeline(params, self.log)
            success = True
        except Exception:
            error_text = traceback.format_exc()
            self.log(error_text)
            self.root.after(0, messagebox.showerror, "Solver error", error_text)
        finally:
            self.root.after(0, self.run_button.config, {"state": "normal"})
            if success and params.get("save_visualisation_data", True):
                self.root.after(0, self.open_viewer_after_run, params)

    def open_viewer_after_run(self, params):
        output_dir = Path(params["output_dir"]) / "OUTPUTS"
        if getattr(sys, "frozen", False):
            base_dir = Path(sys.executable).resolve().parent
            candidates = [
                base_dir / "KidneyViewerQt.exe",
                base_dir / "Viewer" / "KidneyViewerQt.exe",
                base_dir.parent / "KidneyViewerQt" / "KidneyViewerQt.exe",
                base_dir.parent / "KidneyFlowSuite" / "KidneyViewerQt" / "KidneyViewerQt.exe",
            ]
            viewer_exe = next((path for path in candidates if path.exists()), None)
            if viewer_exe is None:
                raise FileNotFoundError(
                    "Could not find KidneyViewerQt.exe beside the packaged solver. "
                    "Please keep the packaged solver and viewer together in the generated suite folder."
                )
            command = [
                str(viewer_exe),
                "--elements",
                str(Path(params["elements_path"]).resolve()),
                "--nodes",
                str(Path(params["nodes_path"]).resolve()),
                "--outputs",
                str(output_dir.resolve()),
            ]
        else:
            viewer_script = Path(__file__).with_name("kidney_viewer_qt.py")
            command = [
                sys.executable,
                str(viewer_script),
                "--elements",
                str(Path(params["elements_path"]).resolve()),
                "--nodes",
                str(Path(params["nodes_path"]).resolve()),
                "--outputs",
                str(output_dir.resolve()),
            ]
        try:
            subprocess.Popen(command)
        except Exception as exc:
            messagebox.showerror(
                "Viewer error",
                f"The solver finished, but the Qt viewer could not be opened.\n\n{exc}",
            )


if __name__ == "__main__":
    root = Tk()
    app = KidneyFlowApp(root)
    root.mainloop()
