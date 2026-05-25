"""Utilities for 0D vascular-tree flow analysis."""

import json
from typing import Optional

import numpy as np


def left_right_elements(
    element_characteristics: np.ndarray,
    n_nodes: Optional[int] = None,
):
    """Find the upstream and downstream elements connected to each node.

    Parameters
    ----------
    element_characteristics:
        Vessel/element matrix. Only the first two columns are used:
        ``begin_node`` and ``end_node``. Node numbers are expected to follow
        the MATLAB convention: 1, 2, ..., N.
    n_nodes:
        Optional number of nodes. If omitted, it is inferred from the largest
        node number in the first two columns.

    Returns
    -------
    left_elem:
        Integer array of shape ``(n_nodes,)``. ``left_elem[node_id - 1]`` is
        the single upstream element ending at that node. A value of 0 means
        no upstream element, which should normally only happen for node 1.
        Element numbers are also MATLAB-style: 1, 2, ..., N_elem.
    right_elem:
        List of length ``n_nodes``. ``right_elem[node_id - 1]`` is a list of
        downstream elements starting from that node. Element numbers are
        MATLAB-style: 1, 2, ..., N_elem.
    """
    elements = np.asarray(element_characteristics)

    if elements.ndim != 2 or elements.shape[1] < 2:
        raise ValueError("element_characteristics must be a 2D array with at least 2 columns.")

    if elements.shape[0] == 0:
        if n_nodes is None:
            n_nodes = 0
        return np.zeros(n_nodes, dtype=np.int64), [[] for _ in range(n_nodes)]

    connectivity = elements[:, :2].astype(np.int64, copy=False)

    if np.any(connectivity < 1):
        raise ValueError("Node numbers must be positive 1-based integers.")

    if n_nodes is None:
        n_nodes = int(connectivity.max())
    elif n_nodes < int(connectivity.max()):
        raise ValueError("n_nodes is smaller than the largest node number in element_characteristics.")

    left_elem = np.zeros(n_nodes, dtype=np.int64)
    right_elem = [[] for _ in range(n_nodes)]

    for elem_id, (begin_node, end_node) in enumerate(connectivity, start=1):
        right_elem[begin_node - 1].append(elem_id)
        left_elem[end_node - 1] = elem_id

    return left_elem, right_elem


def left_right_nodes(element_characteristics, n_nodes=None):
    """Find upstream and downstream nodes connected to each node.

    Returns
    -------
    left_node:
        Integer array of shape ``(n_nodes,)``. ``left_node[node_id - 1]`` is
        the upstream node connected to ``node_id``. A value of 0 means no
        upstream node, normally only for node 1.
    right_node:
        List of length ``n_nodes``. ``right_node[node_id - 1]`` contains the
        downstream nodes connected to ``node_id``.
    """
    elements = np.asarray(element_characteristics)

    if elements.ndim != 2 or elements.shape[1] < 2:
        raise ValueError("element_characteristics must be a 2D array with at least 2 columns.")

    if elements.shape[0] == 0:
        if n_nodes is None:
            n_nodes = 0
        return np.zeros(n_nodes, dtype=np.int64), [[] for _ in range(n_nodes)]

    connectivity = elements[:, :2].astype(np.int64, copy=False)

    if np.any(connectivity < 1):
        raise ValueError("Node numbers must be positive 1-based integers.")

    if n_nodes is None:
        n_nodes = int(connectivity.max())
    elif n_nodes < int(connectivity.max()):
        raise ValueError("n_nodes is smaller than the largest node number in element_characteristics.")

    left_node = np.zeros(n_nodes, dtype=np.int64)
    right_node = [[] for _ in range(n_nodes)]

    for begin_node, end_node in connectivity:
        left_node[end_node - 1] = begin_node
        right_node[begin_node - 1].append(end_node)

    return left_node, right_node


def distance_to_root(left_node):
    """Count the number of elements between each node and node 1."""
    left_node = np.asarray(left_node, dtype=np.int64).reshape(-1)
    n_nodes = left_node.shape[0]
    distances = np.full(n_nodes, -1, dtype=np.int64)
    distances[0] = 0

    for node_index in range(n_nodes):
        if distances[node_index] >= 0:
            continue

        path = []
        current = node_index + 1
        while current != 0 and distances[current - 1] < 0:
            path.append(current)
            current = left_node[current - 1]

        if current == 0 and path[-1] != 1:
            raise ValueError("A node is not connected to node 1 through upstream links.")

        distance = distances[current - 1] if current != 0 else 0
        for node_id in reversed(path):
            distance += 1
            distances[node_id - 1] = distance

    return distances


def strahler_order(
    element_characteristics: np.ndarray,
    n_nodes: Optional[int] = None,
):
    """Calculate the Strahler order of each vessel/element.

    The vascular tree is assumed to be directed from upstream to downstream:
    column 1 is ``begin_node`` and column 2 is ``end_node``. The returned
    array uses Python indexing, so element 1 is stored at ``so[0]``.

    For a terminal element, the Strahler order is 1. For an upstream element,
    let the downstream child orders be the orders of elements starting from
    its end node. If the maximum child order appears at least twice, the
    element order is ``max_order + 1``; otherwise it is ``max_order``.
    """
    elements = np.asarray(element_characteristics)

    if elements.ndim != 2 or elements.shape[1] < 2:
        raise ValueError("element_characteristics must be a 2D array with at least 2 columns.")

    n_elem = elements.shape[0]
    if n_elem == 0:
        return np.zeros(0, dtype=np.int64)

    connectivity = elements[:, :2].astype(np.int64, copy=False)
    _left_elem, right_elem = left_right_elements(connectivity, n_nodes=n_nodes)

    if n_nodes is None:
        n_nodes = len(right_elem)

    so = np.zeros(n_elem, dtype=np.int64)
    pending_children_count = np.zeros(n_elem, dtype=np.int64)
    upstream_of_element = np.zeros(n_elem, dtype=np.int64)
    ready_elements = []

    for elem_index, (_, end_node) in enumerate(connectivity):
        child_elements = right_elem[end_node - 1]
        pending_children_count[elem_index] = len(child_elements)

        if len(child_elements) == 0:
            ready_elements.append(elem_index)

        for child_elem_id in child_elements:
            upstream_of_element[child_elem_id - 1] = elem_index + 1

    while ready_elements:
        elem_index = ready_elements.pop()
        _, end_node = connectivity[elem_index]
        child_elements = right_elem[end_node - 1]

        if len(child_elements) == 0:
            so[elem_index] = 1
        else:
            child_orders = so[np.asarray(child_elements, dtype=np.int64) - 1]
            max_order = int(child_orders.max())
            so[elem_index] = max_order + 1 if np.count_nonzero(child_orders == max_order) >= 2 else max_order

        upstream_elem_id = upstream_of_element[elem_index]
        if upstream_elem_id != 0:
            upstream_index = upstream_elem_id - 1
            pending_children_count[upstream_index] -= 1
            if pending_children_count[upstream_index] == 0:
                ready_elements.append(upstream_index)

    if np.any(so == 0):
        unresolved = np.flatnonzero(so == 0) + 1
        raise ValueError(
            "Could not calculate Strahler order for all elements. "
            "Check that the connectivity is a directed tree without cycles. "
            f"Unresolved element IDs: {unresolved.tolist()}"
        )

    return so


def vessel_resistance(element_characteristics, blood_viscosity, et_data=None):
    """Calculate Poiseuille resistance for each vessel segment.

    Parameters
    ----------
    element_characteristics:
        Vessel matrix with columns ``begin_node, end_node, radius, length``.
        Radius and length are expected in micrometers, matching the MATLAB
        code. Only columns 3 and 4 are used.
    blood_viscosity:
        Reference blood viscosity in Pa.s.
    et_data:
        Optional Young's-modulus / wall-thickness table from ``ET.txt``.
        Supported shapes are:
        - one row and two columns: same ``E`` and thickness for all vessels
        - ``n_vessels`` rows and two columns: one ``E`` and thickness per vessel

        ``E`` is expected in Pa. Thickness is expected in micrometers and is
        converted to meters internally.

    Returns
    -------
    resistance:
        Segment resistance in Pa.s/m^3, equivalent to Pa/(m^3/s).
    compliance:
        Segment compliance in m^3/Pa.
    radius_m:
        Vessel radius in meters.
    length_m:
        Vessel length in meters.
    effective_viscosity:
        Diameter-dependent viscosity used in the resistance calculation.
    """
    elements = np.asarray(element_characteristics)

    if elements.ndim != 2 or elements.shape[1] < 4:
        raise ValueError("element_characteristics must be a 2D array with at least 4 columns.")

    radius_m = elements[:, 2].astype(np.float64, copy=False) / 1.0e6
    length_m = elements[:, 3].astype(np.float64, copy=False) / 1.0e6

    if np.any(radius_m <= 0):
        raise ValueError("All vessel radii must be positive.")
    if np.any(length_m <= 0):
        raise ValueError("All vessel lengths must be positive.")

    diameter_um = radius_m * 2.0e6
    if np.any(diameter_um <= 1.1):
        raise ValueError("The viscosity correction requires vessel diameters greater than 1.1 um.")

    diameter_factor = (diameter_um / (diameter_um - 1.1)) ** 2
    viscosity_factor = (
        6.0 * np.exp(-0.085 * diameter_um)
        + 2.2
        - 2.44 * np.exp(-0.06 * diameter_um**0.645)
    )
    effective_viscosity = blood_viscosity * (1.0 + viscosity_factor * diameter_factor) * diameter_factor

    resistance = (8.0 * effective_viscosity / np.pi) * length_m / radius_m**4
    if et_data is None:
        young_modulus = np.full(radius_m.shape, 100000.0, dtype=np.float64)
        thickness = 0.2 * radius_m
    else:
        et_array = np.asarray(et_data, dtype=np.float64)
        if et_array.ndim != 2 or et_array.shape[1] < 2:
            raise ValueError("ET.txt must have at least 2 columns: Young's modulus and thickness.")
        if et_array.shape[0] == 1:
            young_modulus = np.full(radius_m.shape, float(et_array[0, 0]), dtype=np.float64)
            thickness = np.full(radius_m.shape, float(et_array[0, 1]) / 1.0e6, dtype=np.float64)
        elif et_array.shape[0] == radius_m.shape[0]:
            young_modulus = et_array[:, 0].astype(np.float64, copy=False)
            thickness = et_array[:, 1].astype(np.float64, copy=False) / 1.0e6
        else:
            raise ValueError(
                "ET.txt must contain either 1 row or one row per vessel, "
                "with columns: Young's modulus (Pa), thickness (um)."
            )

        if np.any(young_modulus <= 0):
            raise ValueError("Young's modulus values in ET.txt must be positive.")
        if np.any(thickness <= 0):
            raise ValueError("Thickness values in ET.txt must be positive.")

    compliance = 2.0 * np.pi * radius_m**3 * length_m / (young_modulus * thickness)

    return resistance, compliance, radius_m, length_m, effective_viscosity


def sample_boundary_condition(boundary_condition_table, n_frames):
    """Sample a boundary-condition time series using evenly spaced endpoints."""
    bc = np.asarray(boundary_condition_table, dtype=np.float64)
    if bc.ndim != 2 or bc.shape[1] < 2:
        raise ValueError("boundary_condition_table must have at least 2 columns: time and amplitude.")
    if n_frames < 2:
        raise ValueError("n_frames must be at least 2.")

    times = np.linspace(float(bc[0, 0]), float(bc[-1, 0]), int(n_frames), dtype=np.float64)
    inlet_amplitude = np.interp(times, bc[:, 0], bc[:, 1])
    return times, inlet_amplitude


def sample_glomerular_resistance(so, r_gc_mean=5000.0, r_gc_std=0.0, rng=None):
    """Assign a glomerular resistance to each Strahler-order-1 vessel.

    When ``r_gc_std`` is zero, all glomeruli receive the same resistance equal
    to ``r_gc_mean``. Otherwise, each terminal glomerulus receives an
    independent sample from a log-normal distribution whose linear-space mean
    and standard deviation match the given values. This keeps all sampled
    resistances strictly positive.
    """
    so = np.asarray(so, dtype=np.int64).reshape(-1)
    endpoint = so == 1

    glomerular_resistance = np.zeros(so.shape[0], dtype=np.float64)
    if not np.any(endpoint):
        return glomerular_resistance

    r_gc_mean = float(r_gc_mean)
    r_gc_std = float(r_gc_std)
    if r_gc_mean <= 0:
        raise ValueError("Mean glomerular resistance must be positive.")
    if r_gc_std < 0:
        raise ValueError("Standard deviation of glomerular resistance cannot be negative.")

    if r_gc_std == 0.0:
        glomerular_resistance[endpoint] = r_gc_mean
        return glomerular_resistance

    if rng is None:
        rng = np.random.default_rng()

    sigma2_log = np.log(1.0 + (r_gc_std**2) / (r_gc_mean**2))
    sigma_log = np.sqrt(sigma2_log)
    mu_log = np.log(r_gc_mean) - 0.5 * sigma2_log
    sampled = rng.lognormal(mean=mu_log, sigma=sigma_log, size=int(np.count_nonzero(endpoint)))
    glomerular_resistance[endpoint] = sampled
    return glomerular_resistance


def sample_glomerular_kf(so, kf_mean=27.5e-6, kf_std=0.0, rng=None):
    """Assign a filtration coefficient to each Strahler-order-1 vessel.

    When ``kf_std`` is zero, all glomeruli receive the same ``kf_mean``.
    Otherwise, each terminal glomerulus receives an independent sample from a
    log-normal distribution whose linear-space mean and standard deviation
    match the given values.
    """
    so = np.asarray(so, dtype=np.int64).reshape(-1)
    endpoint = so == 1

    glomerular_kf = np.zeros(so.shape[0], dtype=np.float64)
    if not np.any(endpoint):
        return glomerular_kf

    kf_mean = float(kf_mean)
    kf_std = float(kf_std)
    if kf_mean <= 0:
        raise ValueError("Mean Kf must be positive.")
    if kf_std < 0:
        raise ValueError("Standard deviation of Kf cannot be negative.")

    if kf_std == 0.0:
        glomerular_kf[endpoint] = kf_mean
        return glomerular_kf

    if rng is None:
        rng = np.random.default_rng()

    sigma2_log = np.log(1.0 + (kf_std**2) / (kf_mean**2))
    sigma_log = np.sqrt(sigma2_log)
    mu_log = np.log(kf_mean) - 0.5 * sigma2_log
    sampled = rng.lognormal(mean=mu_log, sigma=sigma_log, size=int(np.count_nonzero(endpoint)))
    glomerular_kf[endpoint] = sampled
    return glomerular_kf


def add_glomerular_resistance(resistance, so, r_gc=5000.0):
    """Add glomerular resistance to terminal Strahler-order-1 vessels.

    Parameters
    ----------
    resistance:
        Vessel resistance array in Pa.s/m^3.
    so:
        Strahler order for each vessel.
    r_gc:
        Single-glomerulus resistance in mmHg.min/mL. The default is 5000,
        matching the kidney-specific value in the MATLAB code.

    Returns
    -------
    modified_resistance:
        Copy of ``resistance`` with glomerular resistance added where
        ``so == 1``.
    r_gc_si:
        Glomerular resistance converted to Pa.s/m^3.
    """
    resistance = np.asarray(resistance, dtype=np.float64)
    so = np.asarray(so).reshape(-1)

    if resistance.shape[0] != so.shape[0]:
        raise ValueError("resistance and so must have the same length.")

    r_gc_si = float(r_gc) * 8.0e9
    modified_resistance = resistance.copy()
    modified_resistance[so == 1] += r_gc_si

    return modified_resistance, r_gc_si


def add_glomerular_resistance_per_vessel(resistance, glomerular_resistance):
    """Add vessel-specific glomerular resistance values to segment resistance."""
    resistance = np.asarray(resistance, dtype=np.float64).reshape(-1)
    glomerular_resistance = np.asarray(glomerular_resistance, dtype=np.float64).reshape(-1)

    if resistance.shape != glomerular_resistance.shape:
        raise ValueError("resistance and glomerular_resistance must have the same shape.")
    if np.any(glomerular_resistance < 0):
        raise ValueError("glomerular_resistance values must be non-negative.")

    glomerular_resistance_si = glomerular_resistance * 8.0e9
    modified_resistance = resistance + glomerular_resistance_si
    return modified_resistance, glomerular_resistance_si


def solve_unit(
    element_characteristics,
    resistance,
    boundary_condition="pressure",
    inlet_value=1.0,
    outlet_pressure=0.0,
    n_nodes=None,
):
    """Solve the steady 0D tree response to a unit inlet condition.

    The network is treated as a directed tree from node 1 downstream. Terminal
    vessels should already include any terminal series resistance, such as the
    glomerular resistance added to Strahler-order-1 vessels.

    Parameters
    ----------
    element_characteristics:
        Vessel matrix. Only columns 1 and 2, ``begin_node`` and ``end_node``,
        are used.
    resistance:
        Vessel resistance in Pa.s/m^3. This may be the modified resistance
        that includes glomerular resistance.
    boundary_condition:
        ``"pressure"`` or ``"flowrate"``. For ``"pressure"``, ``inlet_value``
        is the inlet pressure at node 1. For ``"flowrate"``, ``inlet_value``
        is the total inflow entering node 1.
    inlet_value:
        Unit inlet pressure or unit inlet flow rate.
    outlet_pressure:
        Constant terminal pressure, e.g. ``P_ef``. Use SI units consistent
        with the pressure drop implied by resistance and flow. If your
        pressure waveform is in Pa, this should also be in Pa.
    n_nodes:
        Optional number of nodes.

    Returns
    -------
    pressure:
        Pressure at each node.
    flowrate:
        Flow rate in each vessel.
    equivalent_resistance:
        Effective downstream resistance for each vessel, including all
        downstream parallel/series combinations.
    flow_fraction:
        Fraction of parent-vessel flow assigned to each vessel. For vessels
        connected directly to node 1, this is the fraction of total inlet flow.
    """
    elements = np.asarray(element_characteristics)
    resistance = np.asarray(resistance, dtype=np.float64).reshape(-1)

    if elements.ndim != 2 or elements.shape[1] < 2:
        raise ValueError("element_characteristics must be a 2D array with at least 2 columns.")
    if elements.shape[0] != resistance.shape[0]:
        raise ValueError("resistance must contain one value per vessel.")
    if np.any(resistance <= 0):
        raise ValueError("All resistance values must be positive.")

    connectivity = elements[:, :2].astype(np.int64, copy=False)
    n_elem = connectivity.shape[0]

    if n_elem == 0:
        if n_nodes is None:
            n_nodes = 1
        pressure = np.full(n_nodes, float(outlet_pressure), dtype=np.float64)
        return pressure, np.zeros(0), np.zeros(0), np.zeros(0)

    left_elem, right_elem = left_right_elements(connectivity, n_nodes=n_nodes)
    if n_nodes is None:
        n_nodes = len(right_elem)

    equivalent_resistance = resistance.copy()
    flow_fraction = np.zeros(n_elem, dtype=np.float64)
    pending_children_count = np.zeros(n_elem, dtype=np.int64)
    ready_elements = []

    for elem_index, (_, end_node) in enumerate(connectivity):
        child_elem_ids = right_elem[end_node - 1]
        pending_children_count[elem_index] = len(child_elem_ids)
        if len(child_elem_ids) == 0:
            ready_elements.append(elem_index)

    while ready_elements:
        elem_index = ready_elements.pop()
        begin_node, end_node = connectivity[elem_index]
        child_elem_ids = right_elem[end_node - 1]

        if len(child_elem_ids) > 0:
            child_indices = np.asarray(child_elem_ids, dtype=np.int64) - 1
            conductance_sum = np.sum(1.0 / equivalent_resistance[child_indices])
            equivalent_resistance[elem_index] = resistance[elem_index] + 1.0 / conductance_sum
            flow_fraction[child_indices] = (1.0 / equivalent_resistance[child_indices]) / conductance_sum

        upstream_elem_id = left_elem[begin_node - 1]
        if upstream_elem_id != 0:
            upstream_index = upstream_elem_id - 1
            pending_children_count[upstream_index] -= 1
            if pending_children_count[upstream_index] == 0:
                ready_elements.append(upstream_index)

    if np.any(pending_children_count != 0):
        raise ValueError("Could not reduce the network. Check for cycles or disconnected elements.")

    root_elem_ids = right_elem[0]
    if len(root_elem_ids) == 0:
        raise ValueError("Node 1 has no downstream vessels.")

    root_indices = np.asarray(root_elem_ids, dtype=np.int64) - 1
    root_conductance_sum = np.sum(1.0 / equivalent_resistance[root_indices])
    flow_fraction[root_indices] = (1.0 / equivalent_resistance[root_indices]) / root_conductance_sum

    boundary = boundary_condition.lower()
    pressure = np.zeros(n_nodes, dtype=np.float64)
    flowrate = np.zeros(n_elem, dtype=np.float64)

    if boundary in ("pressure", "p"):
        pressure[0] = float(inlet_value)
        total_inflow = (pressure[0] - float(outlet_pressure)) * root_conductance_sum
    elif boundary in ("flowrate", "flow", "q"):
        total_inflow = float(inlet_value)
        pressure[0] = float(outlet_pressure) + total_inflow / root_conductance_sum
    else:
        raise ValueError("boundary_condition must be 'pressure' or 'flowrate'.")

    flowrate[root_indices] = total_inflow * flow_fraction[root_indices]

    current_nodes = [1]
    while current_nodes:
        new_current_nodes = []
        for node_id in current_nodes:
            child_elem_ids = right_elem[node_id - 1]
            for child_elem_id in child_elem_ids:
                child_index = child_elem_id - 1
                end_node = connectivity[child_index, 1]
                pressure[end_node - 1] = pressure[node_id - 1] - resistance[child_index] * flowrate[child_index]
                new_current_nodes.append(end_node)

                grandchild_elem_ids = right_elem[end_node - 1]
                if len(grandchild_elem_ids) > 0:
                    grandchild_indices = np.asarray(grandchild_elem_ids, dtype=np.int64) - 1
                    flowrate[grandchild_indices] = flowrate[child_index] * flow_fraction[grandchild_indices]

        current_nodes = new_current_nodes

    return pressure, flowrate, equivalent_resistance, flow_fraction


def apply_inlet_waveform(
    elements,
    pressure_unit,
    flowrate_unit,
    boundary_condition,
    boundary_condition_table,
    n_frames,
    p_ef=50.0,
):
    """Scale a unit solution by sampled inlet boundary-condition amplitudes.

    Parameters
    ----------
    elements:
        Vessel matrix with columns ``begin_node`` and ``end_node``. Used to
        calculate average pressure in each vessel from its two end nodes.
    pressure_unit:
        Unit-solution pressure at nodes, in Pa for pressure BC or Pa per
        m^3/s for flow-rate BC.
    flowrate_unit:
        Unit-solution flow rate in vessels, in m^3/s for pressure BC or
        dimensionless flow split for flow-rate BC.
    boundary_condition:
        ``"pressure"`` or ``"flowrate"``.
    boundary_condition_table:
        Two-column array: time in column 1 and inlet amplitude in column 2.
        Pressure amplitudes are expected in mmHg. Flow-rate amplitudes are
        expected in mL/min.
    n_frames:
        Number of requested output time frames. Must be at least 2. The first
        and last frames use the first and last times in the boundary-condition
        table.
    p_ef:
        Constant downstream pressure in mmHg.

    Returns
    -------
    times:
        Output times.
    pressure_t:
        Node pressure over time, in mmHg, shape ``(n_nodes, n_times)``.
    flowrate_t:
        Vessel flow rate over time, in mL/min, shape ``(n_elem, n_times)``.
    element_pressure_t:
        Average pressure in each vessel over time, in mmHg.
    inlet_amplitude:
        Interpolated inlet amplitude in the original input units.
    """
    elements = np.asarray(elements)
    pressure_unit = np.asarray(pressure_unit, dtype=np.float64).reshape(-1)
    flowrate_unit = np.asarray(flowrate_unit, dtype=np.float64).reshape(-1)
    if elements.ndim != 2 or elements.shape[1] < 2:
        raise ValueError("elements must be a 2D array with at least 2 columns.")

    times, inlet_amplitude = sample_boundary_condition(boundary_condition_table, n_frames)

    boundary = boundary_condition.lower()

    if boundary in ("pressure", "p"):
        scale = (inlet_amplitude - float(p_ef)) * 133.322
        pressure_t = float(p_ef) + pressure_unit[:, None] * scale[None, :] * 0.0075
        flowrate_t = flowrate_unit[:, None] * scale[None, :] * 6.0e7
    elif boundary in ("flowrate", "flow", "q"):
        scale = inlet_amplitude * 1.66667e-8
        pressure_t = float(p_ef) + pressure_unit[:, None] * scale[None, :] * 0.0075
        flowrate_t = flowrate_unit[:, None] * scale[None, :] * 6.0e7
    else:
        raise ValueError("boundary_condition must be 'pressure' or 'flowrate'.")

    connectivity = elements[:, :2].astype(np.int64, copy=False) - 1
    if np.any(connectivity < 0) or np.any(connectivity >= pressure_t.shape[0]):
        raise ValueError("Element connectivity contains node IDs outside the pressure array.")

    element_pressure_t = 0.5 * (
        pressure_t[connectivity[:, 0], :] + pressure_t[connectivity[:, 1], :]
    )

    return times, pressure_t, flowrate_t, element_pressure_t, inlet_amplitude


def adjust_terminal_element_pressure(element_pressure_t, flowrate_t, so, r_gc):
    """Adjust SO=1 element pressures to represent afferent-arteriole averages.

    The current element-average pressure for a terminal element with lumped
    total resistance ``R_aff + R_GC`` is based on the pressure at the
    beginning of the afferent arteriole and the pressure after the glomerulus.
    To recover the average pressure within the afferent arteriole itself, add
    ``0.5 * R_GC * Q`` in mmHg to Strahler-order-1 elements.
    """
    element_pressure_t = np.asarray(element_pressure_t, dtype=np.float64)
    flowrate_t = np.asarray(flowrate_t, dtype=np.float64)
    so = np.asarray(so, dtype=np.int64).reshape(-1)

    if element_pressure_t.ndim == 1:
        element_pressure_t = element_pressure_t[:, None]
    if flowrate_t.ndim == 1:
        flowrate_t = flowrate_t[:, None]
    if element_pressure_t.shape != flowrate_t.shape:
        raise ValueError("element_pressure_t and flowrate_t must have the same shape.")
    if element_pressure_t.shape[0] != so.shape[0]:
        raise ValueError("so must have one value per vessel.")

    adjusted_pressure = element_pressure_t.copy()
    endpoint = so == 1
    if np.any(endpoint):
        adjusted_pressure[endpoint, :] += 0.5 * float(r_gc) * flowrate_t[endpoint, :]

    return adjusted_pressure


def adjust_terminal_element_pressure_per_vessel(element_pressure_t, flowrate_t, so, glomerular_resistance):
    """Adjust SO=1 element pressures using vessel-specific glomerular resistance."""
    element_pressure_t = np.asarray(element_pressure_t, dtype=np.float64)
    flowrate_t = np.asarray(flowrate_t, dtype=np.float64)
    so = np.asarray(so, dtype=np.int64).reshape(-1)
    glomerular_resistance = np.asarray(glomerular_resistance, dtype=np.float64).reshape(-1)

    if element_pressure_t.ndim == 1:
        element_pressure_t = element_pressure_t[:, None]
    if flowrate_t.ndim == 1:
        flowrate_t = flowrate_t[:, None]
    if element_pressure_t.shape != flowrate_t.shape:
        raise ValueError("element_pressure_t and flowrate_t must have the same shape.")
    if element_pressure_t.shape[0] != so.shape[0] or so.shape[0] != glomerular_resistance.shape[0]:
        raise ValueError("so and glomerular_resistance must have one value per vessel.")

    adjusted_pressure = element_pressure_t.copy()
    endpoint = so == 1
    if np.any(endpoint):
        adjusted_pressure[endpoint, :] += 0.5 * glomerular_resistance[endpoint, None] * flowrate_t[endpoint, :]

    return adjusted_pressure


def solve_compliance(
    element_characteristics,
    resistance,
    compliance,
    boundary_condition,
    boundary_condition_table,
    n_frames,
    p_ef=50.0,
    unit_pressure=1.0e3,
    n_nodes=None,
):
    """Solve the compliance-aware transient problem using a sparse formulation.

    The transient unknowns are gauge pressures relative to the constant outlet
    pressure ``p_ef``. Returned pressures are converted back to absolute mmHg.
    """
    try:
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import factorized
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for the compliance solver. Install it with "
            "'python -m pip install scipy'."
        ) from exc

    elements = np.asarray(element_characteristics)
    resistance = np.asarray(resistance, dtype=np.float64).reshape(-1)
    compliance = np.asarray(compliance, dtype=np.float64).reshape(-1)

    if elements.ndim != 2 or elements.shape[1] < 2:
        raise ValueError("element_characteristics must be a 2D array with at least 2 columns.")
    if elements.shape[0] != resistance.shape[0] or elements.shape[0] != compliance.shape[0]:
        raise ValueError("resistance and compliance must contain one value per vessel.")
    if np.any(resistance <= 0):
        raise ValueError("All resistance values must be positive.")
    if np.any(compliance < 0):
        raise ValueError("Compliance values must be non-negative.")

    connectivity = elements[:, :2].astype(np.int64, copy=False)
    if n_nodes is None:
        n_nodes = int(connectivity.max())

    left_node, right_node = left_right_nodes(connectivity, n_nodes=n_nodes)
    left_elem, right_elem = left_right_elements(connectivity, n_nodes=n_nodes)
    terminal_nodes = np.asarray([len(children) == 0 for children in right_elem], dtype=bool)

    boundary = boundary_condition.lower()
    if boundary in ("pressure", "p"):
        unknown_nodes = np.flatnonzero(~terminal_nodes) + 1
        unknown_nodes = unknown_nodes[unknown_nodes != 1]
        steady_inlet_value = float(unit_pressure)
    elif boundary in ("flowrate", "flow", "q"):
        unknown_nodes = np.flatnonzero(~terminal_nodes) + 1
        steady_inlet_value = 1.0
    else:
        raise ValueError("boundary_condition must be 'pressure' or 'flowrate'.")

    n_unknown = int(unknown_nodes.shape[0])
    if n_unknown == 0:
        raise ValueError("The compliance solver found no unknown nodes.")

    rel_vec = np.zeros(n_nodes, dtype=np.int64)
    rel_vec[unknown_nodes - 1] = np.arange(1, n_unknown + 1, dtype=np.int64)
    rev_vec = unknown_nodes.copy()

    rows_a = []
    cols_a = []
    vals_a = []
    rows_b = []
    cols_b = []
    vals_b = []
    rhs_pressure_unit = np.zeros(n_unknown, dtype=np.float64)
    rhs_flow_unit = np.zeros(n_unknown, dtype=np.float64)

    root_children = right_node[0]
    root_child_elements = right_elem[0]

    for row_index, node_id in enumerate(rev_vec, start=1):
        node_zero = node_id - 1

        if node_id == 1:
            right_indices = np.asarray(root_child_elements, dtype=np.int64) - 1
            conductance = np.sum(1.0 / resistance[right_indices])
            rows_a.append(row_index - 1)
            cols_a.append(row_index - 1)
            vals_a.append(conductance)

            for child_id, elem_id in zip(root_children, root_child_elements):
                child_label = rel_vec[child_id - 1]
                elem_index = elem_id - 1
                if child_label != 0:
                    rows_a.append(row_index - 1)
                    cols_a.append(child_label - 1)
                    vals_a.append(-1.0 / resistance[elem_index])

                rows_b.append(row_index - 1)
                cols_b.append(row_index - 1)
                vals_b.append(compliance[elem_index] / 2.0)

                if child_label != 0:
                    rows_b.append(row_index - 1)
                    cols_b.append(child_label - 1)
                    vals_b.append(compliance[elem_index] / 2.0)

            rhs_flow_unit[row_index - 1] = 1.0
            continue

        neighbor_nodes = [int(left_node[node_zero])] + [int(child) for child in right_node[node_zero]]
        neighbor_elements = left_elem[np.asarray([node_id] + [int(child) for child in right_node[node_zero]], dtype=np.int64) - 1]
        neighbor_indices = np.asarray(neighbor_elements, dtype=np.int64) - 1

        rows_a.append(row_index - 1)
        cols_a.append(row_index - 1)
        vals_a.append(np.sum(1.0 / resistance[neighbor_indices]))

        for neighbor_id, elem_id in zip(neighbor_nodes, neighbor_elements):
            elem_index = elem_id - 1
            neighbor_label = rel_vec[neighbor_id - 1] if neighbor_id > 0 else 0
            if neighbor_label != 0:
                rows_a.append(row_index - 1)
                cols_a.append(neighbor_label - 1)
                vals_a.append(-1.0 / resistance[elem_index])
            elif neighbor_id == 1:
                rhs_pressure_unit[row_index - 1] += float(unit_pressure) / resistance[elem_index]

        for child_id, elem_id in zip(right_node[node_zero], right_elem[node_zero]):
            child_label = rel_vec[child_id - 1]
            elem_index = elem_id - 1
            rows_b.append(row_index - 1)
            cols_b.append(row_index - 1)
            vals_b.append(compliance[elem_index] / 2.0)
            if child_label != 0:
                rows_b.append(row_index - 1)
                cols_b.append(child_label - 1)
                vals_b.append(compliance[elem_index] / 2.0)

    matrix_a = csc_matrix((vals_a, (rows_a, cols_a)), shape=(n_unknown, n_unknown))
    matrix_b = csc_matrix((vals_b, (rows_b, cols_b)), shape=(n_unknown, n_unknown))

    pressure_unit_abs, flowrate_unit_abs, equivalent_resistance, flow_fraction = solve_unit(
        connectivity,
        resistance,
        boundary_condition=boundary_condition,
        inlet_value=steady_inlet_value,
        outlet_pressure=0.0,
        n_nodes=n_nodes,
    )

    times, inlet_amplitude = sample_boundary_condition(boundary_condition_table, n_frames)
    if times.shape[0] < 2:
        raise ValueError("The compliance solver requires at least 2 sampled time points.")

    steady_unknown = pressure_unit_abs[unknown_nodes - 1]
    if boundary in ("pressure", "p"):
        gauge_values = (inlet_amplitude - float(p_ef)) * 133.322
        rhs_at_value = lambda value: rhs_pressure_unit * (value / float(unit_pressure))
        x_current = steady_unknown * (gauge_values[0] / float(unit_pressure))
    else:
        gauge_values = inlet_amplitude * 1.66667e-8
        rhs_at_value = lambda value: rhs_flow_unit * value
        x_current = steady_unknown * gauge_values[0]

    x_t = np.zeros((n_unknown, times.shape[0]), dtype=np.float64)
    x_t[:, 0] = x_current

    delta_t = np.diff(times)
    if np.any(delta_t <= 0):
        raise ValueError("Boundary-condition times must be strictly increasing after sampling.")

    for time_index, dt in enumerate(delta_t, start=1):
        system_matrix = matrix_a + matrix_b * (1.0 / dt)
        solve_linear_system = factorized(system_matrix)
        rhs_next = rhs_at_value(gauge_values[time_index])
        x_current = solve_linear_system(rhs_next + (matrix_b * (1.0 / dt)).dot(x_current))
        x_t[:, time_index] = x_current

    pressure_gauge_pa = np.zeros((n_nodes, times.shape[0]), dtype=np.float64)
    pressure_gauge_pa[terminal_nodes, :] = 0.0
    pressure_gauge_pa[unknown_nodes - 1, :] = x_t
    if boundary in ("pressure", "p"):
        pressure_gauge_pa[0, :] = gauge_values

    flowrate_m3s = np.zeros((connectivity.shape[0], times.shape[0]), dtype=np.float64)
    begin_indices = connectivity[:, 0] - 1
    end_indices = connectivity[:, 1] - 1
    flowrate_m3s[:, :] = (
        pressure_gauge_pa[begin_indices, :] - pressure_gauge_pa[end_indices, :]
    ) / resistance[:, None]

    pressure_t = float(p_ef) + pressure_gauge_pa * 0.00750061683
    flowrate_t = flowrate_m3s * 6.0e7
    element_pressure_t = 0.5 * (
        pressure_t[begin_indices, :] + pressure_t[end_indices, :]
    )

    return (
        times,
        pressure_t,
        flowrate_t,
        element_pressure_t,
        inlet_amplitude,
        pressure_unit_abs,
        flowrate_unit_abs,
        equivalent_resistance,
        flow_fraction,
    )


def calculate_gfr(flowrate_t, so, r_gc, p_ef, p_bowman=15.0, p_osmotic=34.75, kf=27.5e-6):
    """Calculate total GFR over time from terminal-vessel flow rates.

    Parameters
    ----------
    flowrate_t:
        Vessel flow rate over time in mL/min, shape ``(n_elem, n_times)``.
    so:
        Strahler order for each vessel.
    r_gc:
        Glomerular resistance in mmHg.min/mL.
    p_ef:
        Efferent/downstream pressure in mmHg.
    p_bowman:
        Bowman's-space pressure in mmHg.
    p_osmotic:
        Colloid osmotic pressure in mmHg.
    kf:
        Filtration coefficient for one glomerulus in mL/(min.mmHg).

    Returns
    -------
    gfr_t:
        Total GFR over time in mL/min.
    n_glomeruli:
        Number of terminal Strahler-order-1 vessels.
    """
    flowrate_t = np.asarray(flowrate_t, dtype=np.float64)
    so = np.asarray(so).reshape(-1)

    if flowrate_t.ndim == 1:
        flowrate_t = flowrate_t[:, None]
    if flowrate_t.shape[0] != so.shape[0]:
        raise ValueError("flowrate_t and so must have the same number of vessels.")

    endpoint = so == 1
    n_glomeruli = int(np.count_nonzero(endpoint))
    if n_glomeruli == 0:
        raise ValueError("No Strahler-order-1 vessels were found for GFR calculation.")

    q_end_total = np.sum(flowrate_t[endpoint, :], axis=0)
    gfr_t = float(kf) * (
        float(r_gc) * q_end_total / 2.0
        + (float(p_ef) - float(p_bowman) - float(p_osmotic)) * n_glomeruli
    )

    return gfr_t, n_glomeruli


def calculate_gfr_per_glomerulus(
    flowrate_t,
    so,
    glomerular_resistance,
    p_ef,
    p_bowman=15.0,
    p_osmotic=34.75,
    kf=27.5e-6,
):
    """Calculate total GFR using vessel-specific glomerular resistance/Kf values."""
    flowrate_t = np.asarray(flowrate_t, dtype=np.float64)
    so = np.asarray(so, dtype=np.int64).reshape(-1)
    glomerular_resistance = np.asarray(glomerular_resistance, dtype=np.float64).reshape(-1)
    kf_array = np.asarray(kf, dtype=np.float64)

    if flowrate_t.ndim == 1:
        flowrate_t = flowrate_t[:, None]
    if flowrate_t.shape[0] != so.shape[0] or so.shape[0] != glomerular_resistance.shape[0]:
        raise ValueError("flowrate_t, so, and glomerular_resistance must have one value per vessel.")

    endpoint = so == 1
    n_glomeruli = int(np.count_nonzero(endpoint))
    if n_glomeruli == 0:
        raise ValueError("No Strahler-order-1 vessels were found for GFR calculation.")

    p_capillary = float(p_ef) + 0.5 * glomerular_resistance[endpoint, None] * flowrate_t[endpoint, :]
    if kf_array.ndim == 0:
        kf_term = float(kf_array)
    else:
        kf_array = kf_array.reshape(-1)
        if kf_array.shape[0] != so.shape[0]:
            raise ValueError("When Kf is provided per vessel, it must have one value per vessel.")
        kf_term = kf_array[endpoint, None]
    gfr_t = np.sum(
        kf_term * (p_capillary - float(p_bowman) - float(p_osmotic)),
        axis=0,
    )
    return gfr_t, n_glomeruli


def plot_flowrate_vs_strahler(flowrate_t, so, output_path):
    """Save flow-rate versus Strahler-order summary as an SVG figure.

    Flow rate is first averaged over all time steps for each vessel. Then,
    for each Strahler order, the plotted point is the mean of those vessel
    averages and the error bar is their standard deviation.
    """
    from matplotlib.figure import Figure

    flowrate_t = np.asarray(flowrate_t, dtype=np.float64)
    so = np.asarray(so, dtype=np.int64).reshape(-1)

    if flowrate_t.ndim == 1:
        flowrate_t = flowrate_t[:, None]
    if flowrate_t.shape[0] != so.shape[0]:
        raise ValueError("flowrate_t and so must have the same number of vessels.")

    vessel_mean_flowrate = np.mean(np.abs(flowrate_t), axis=1)
    orders = np.unique(so)
    mean_by_order = np.zeros(orders.shape[0], dtype=np.float64)
    std_by_order = np.zeros(orders.shape[0], dtype=np.float64)

    for index, order in enumerate(orders):
        values = vessel_mean_flowrate[so == order]
        mean_by_order[index] = np.mean(values)
        std_by_order[index] = np.std(values)

    if np.any(mean_by_order <= 0):
        raise ValueError("Flow-rate means must be positive for a logarithmic y-axis.")

    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(
        orders,
        mean_by_order,
        yerr=std_by_order,
        fmt="o",
        linestyle="none",
        linewidth=2,
        markersize=9,
        capsize=6,
        color="red",
        ecolor="red",
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=2,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Strahler order", fontsize=28)
    ax.set_ylabel("Flow rate (mL/min)", fontsize=28)
    ax.set_xticks(orders)
    ax.tick_params(axis="both", labelsize=24)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")

    return orders, mean_by_order, std_by_order


def plot_pressure_vs_strahler(element_pressure_t, so, output_path):
    """Save pressure versus Strahler-order summary as an SVG figure.

    Pressure is first averaged over all time steps for each vessel. Then,
    for each Strahler order, the plotted point is the mean of those vessel
    averages and the error bar is their standard deviation.
    """
    from matplotlib.figure import Figure

    element_pressure_t = np.asarray(element_pressure_t, dtype=np.float64)
    so = np.asarray(so, dtype=np.int64).reshape(-1)

    if element_pressure_t.ndim == 1:
        element_pressure_t = element_pressure_t[:, None]
    if element_pressure_t.shape[0] != so.shape[0]:
        raise ValueError("element_pressure_t and so must have the same number of vessels.")

    vessel_mean_pressure = np.mean(element_pressure_t, axis=1)
    orders = np.unique(so)
    mean_by_order = np.zeros(orders.shape[0], dtype=np.float64)
    std_by_order = np.zeros(orders.shape[0], dtype=np.float64)

    for index, order in enumerate(orders):
        values = vessel_mean_pressure[so == order]
        mean_by_order[index] = np.mean(values)
        std_by_order[index] = np.std(values)

    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(
        orders,
        mean_by_order,
        yerr=std_by_order,
        fmt="o",
        linestyle="none",
        linewidth=2,
        markersize=9,
        capsize=6,
        color="red",
        ecolor="red",
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=2,
    )
    ax.set_xlabel("Strahler order", fontsize=28)
    ax.set_ylabel("Pressure (mmHg)", fontsize=28)
    ax.set_xticks(orders)
    ax.tick_params(axis="both", labelsize=24)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")

    return orders, mean_by_order, std_by_order


def get_colormap(cmap):
    """Return a Matplotlib colormap, including a small parula approximation."""
    if str(cmap).lower() != "parula":
        return cmap

    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0.2081, 0.1663, 0.5292),
        (0.1180, 0.3870, 0.7100),
        (0.0000, 0.6800, 0.8400),
        (0.2000, 0.7600, 0.5000),
        (0.7500, 0.7400, 0.2000),
        (0.9769, 0.9839, 0.0805),
    ]
    return LinearSegmentedColormap.from_list("parula", colors, N=256)


def transform_color_values(values, log_color):
    values = np.asarray(values, dtype=np.float64)
    if not log_color:
        return values

    # Logarithmic colouring cannot display negative or zero values.
    # Flow rates can become negative in transient/compliant simulations
    # when the local direction is opposite to the element orientation, so
    # visualise their magnitude on a log scale.
    values = np.abs(values)

    positive = values > 0
    if np.any(positive):
        # Replace exact zeros by the smallest positive value in this frame
        # so that PyVista/VTK log colour mapping always receives positives.
        floor_value = np.nanmin(values[positive])
        values = values.copy()
        values[~positive] = floor_value
    else:
        values = np.full_like(values, 1.0e-12, dtype=np.float64)
    return values


def transform_color_limits(clim, log_color):
    if clim is None:
        return None
    clim = tuple(float(value) for value in clim)
    if not log_color:
        return clim
    if clim[0] <= 0 or clim[1] <= 0:
        raise ValueError("Logarithmic color limits must be positive.")
    return clim


def choose_scale_bar_length(nodes):
    """Choose a round scale-bar length from the spatial extent."""
    nodes = np.asarray(nodes, dtype=np.float64)
    span = np.ptp(nodes[:, :3], axis=0)
    width = np.max(span)
    if width <= 0:
        return 1.0

    lower = width / 10.0
    upper = width / 5.0
    target = np.sqrt(lower * upper)
    exponent = np.floor(np.log10(target))
    base = 10.0**exponent
    candidates = []
    for exp_offset in (-1, 0, 1, 2):
        exp_base = base * (10.0**exp_offset)
        candidates.extend(multiplier * exp_base for multiplier in (1.0, 2.0, 5.0))

    valid = [candidate for candidate in candidates if lower <= candidate <= upper]
    if valid:
        return min(valid, key=lambda candidate: abs(candidate - target))
    return min(candidates, key=lambda candidate: abs(candidate - target))


def add_scale_bar(plotter, nodes, unit=""):
    """Add a fixed screen-space scale bar above the bottom color bar."""
    nodes = np.asarray(nodes, dtype=np.float64)
    length = choose_scale_bar_length(nodes)
    label = f"{length:g} um"

    try:
        import vtk
    except Exception:
        return

    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToNormalizedViewport()

    line = vtk.vtkLineSource()
    line.SetPoint1(0.20, 0.165, 0.0)
    line.SetPoint2(0.36, 0.165, 0.0)

    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputConnection(line.GetOutputPort())
    mapper.SetTransformCoordinate(coordinate)

    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.0, 0.0, 0.0)
    actor.GetProperty().SetLineWidth(5)
    plotter.renderer.AddActor2D(actor)

    text = vtk.vtkTextActor()
    text.SetInput(label)
    text.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text.SetPosition(0.28, 0.18)
    text.GetTextProperty().SetFontSize(18)
    text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
    text.GetTextProperty().SetJustificationToCentered()
    plotter.renderer.AddActor2D(text)


def vascular_tree_display_mask(nodes, elements, so, full_detail_quarter="All", outside_min_strahler=3):
    """Mask vessels for quarter-detail visualization.

    The selected quarter is defined in the X-Y plane using vessel midpoint
    coordinates. Vessels inside that quarter are shown at all Strahler orders.
    Outside that quarter, only vessels with ``SO >= outside_min_strahler`` are
    shown.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements)
    so = np.asarray(so).reshape(-1)

    if str(full_detail_quarter).lower() == "all":
        return np.ones(elements.shape[0], dtype=bool)

    connectivity = elements[:, :2].astype(np.int64, copy=False) - 1
    midpoints = 0.5 * (nodes[connectivity[:, 0], :3] + nodes[connectivity[:, 1], :3])
    center = 0.5 * (np.min(nodes[:, :3], axis=0) + np.max(nodes[:, :3], axis=0))

    quarter = str(full_detail_quarter).lower()
    if quarter in ("upper right", "right upper", "x+ y+"):
        in_quarter = (midpoints[:, 0] >= center[0]) & (midpoints[:, 1] >= center[1])
    elif quarter in ("upper left", "left upper", "x- y+"):
        in_quarter = (midpoints[:, 0] < center[0]) & (midpoints[:, 1] >= center[1])
    elif quarter in ("lower right", "right lower", "x+ y-"):
        in_quarter = (midpoints[:, 0] >= center[0]) & (midpoints[:, 1] < center[1])
    elif quarter in ("lower left", "left lower", "x- y-"):
        in_quarter = (midpoints[:, 0] < center[0]) & (midpoints[:, 1] < center[1])
    else:
        raise ValueError("full_detail_quarter must be All, Upper Right, Upper Left, Lower Right, or Lower Left.")

    return in_quarter | (so >= int(outside_min_strahler))


def camera_view_axes(camera_position):
    """Return camera right/up/view vectors from a PyVista camera_position."""
    camera_location = np.asarray(camera_position[0], dtype=np.float64)
    focal_point = np.asarray(camera_position[1], dtype=np.float64)
    view_up = np.asarray(camera_position[2], dtype=np.float64)

    view_direction = focal_point - camera_location
    view_direction /= np.linalg.norm(view_direction)
    view_up /= np.linalg.norm(view_up)
    right = np.cross(view_direction, view_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, view_direction)
    up /= np.linalg.norm(up)

    return right, up, view_direction


def vascular_tree_display_mask_view_space(
    nodes,
    elements,
    so,
    camera_position,
    full_detail_quarter="All",
    outside_min_strahler=3,
):
    """Mask vessels using quarters defined in the current camera view plane."""
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements)
    so = np.asarray(so).reshape(-1)

    if str(full_detail_quarter).lower() == "all":
        return np.ones(elements.shape[0], dtype=bool)

    connectivity = elements[:, :2].astype(np.int64, copy=False) - 1
    midpoints = 0.5 * (nodes[connectivity[:, 0], :3] + nodes[connectivity[:, 1], :3])
    center = 0.5 * (np.min(nodes[:, :3], axis=0) + np.max(nodes[:, :3], axis=0))
    right, up, _view_direction = camera_view_axes(camera_position)

    relative = midpoints - center
    view_x = relative @ right
    view_y = relative @ up

    quarter = str(full_detail_quarter).lower()
    if quarter == "upper right":
        in_quarter = (view_x >= 0) & (view_y >= 0)
    elif quarter == "upper left":
        in_quarter = (view_x < 0) & (view_y >= 0)
    elif quarter == "lower right":
        in_quarter = (view_x >= 0) & (view_y < 0)
    elif quarter == "lower left":
        in_quarter = (view_x < 0) & (view_y < 0)
    else:
        raise ValueError("full_detail_quarter must be All, Upper Right, Upper Left, Lower Right, or Lower Left.")

    return in_quarter | (so >= int(outside_min_strahler))


def rotation_matrix_from_euler(rotation_degrees):
    """Create a 3D rotation matrix from X/Y/Z Euler angles in degrees."""
    rx, ry, rz = np.deg2rad(np.asarray(rotation_degrees, dtype=np.float64))

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    return rotation_z @ rotation_y @ rotation_x


def default_partition_frame(nodes):
    """Return a partition frame centered on the data with zero rotation."""
    nodes = np.asarray(nodes, dtype=np.float64)
    origin = 0.5 * (np.min(nodes[:, :3], axis=0) + np.max(nodes[:, :3], axis=0))
    return {
        "origin": origin.tolist(),
        "rotation_degrees": [0.0, 0.0, 0.0],
    }


def local_coordinates(points, partition_frame):
    """Transform global points into partition-frame local coordinates."""
    origin = np.asarray(partition_frame["origin"], dtype=np.float64)
    rotation = rotation_matrix_from_euler(partition_frame["rotation_degrees"])
    return (np.asarray(points, dtype=np.float64) - origin) @ rotation


def vascular_tree_display_mask_partition(
    nodes,
    elements,
    so,
    partition_frame,
    region_min_strahler,
):
    """Mask vessels using a local partition frame and per-region SO thresholds."""
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements)
    so = np.asarray(so).reshape(-1)

    connectivity = elements[:, :2].astype(np.int64, copy=False) - 1
    midpoints = 0.5 * (nodes[connectivity[:, 0], :3] + nodes[connectivity[:, 1], :3])
    local = local_coordinates(midpoints, partition_frame)

    region_keys = np.empty(elements.shape[0], dtype=object)
    region_keys[(local[:, 0] >= 0) & (local[:, 1] >= 0)] = "x_plus_y_plus"
    region_keys[(local[:, 0] < 0) & (local[:, 1] >= 0)] = "x_minus_y_plus"
    region_keys[(local[:, 0] < 0) & (local[:, 1] < 0)] = "x_minus_y_minus"
    region_keys[(local[:, 0] >= 0) & (local[:, 1] < 0)] = "x_plus_y_minus"

    keep = np.zeros(elements.shape[0], dtype=bool)
    for region_key, min_so in region_min_strahler.items():
        keep |= (region_keys == region_key) & (so >= int(min_so))

    return keep


def save_partition_frame(partition_frame, frame_path):
    """Save a local partition frame to JSON."""
    from pathlib import Path

    frame_path = Path(frame_path)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frame_path, "w", encoding="utf-8") as file:
        json.dump(partition_frame, file, indent=4)


def load_partition_frame(frame_path):
    """Load a local partition frame from JSON."""
    with open(frame_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_high_resolution_screenshot(plotter, screenshot_path, scale=6):
    """Save a high-resolution screenshot using a magnification scale."""
    from pathlib import Path

    screenshot_path = Path(screenshot_path)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_path = str(screenshot_path)
    try:
        plotter.screenshot(screenshot_path, scale=int(scale))
    except TypeError:
        original_size = list(plotter.window_size)
        plotter.window_size = [original_size[0] * int(scale), original_size[1] * int(scale)]
        plotter.screenshot(screenshot_path)
        plotter.window_size = original_size


def save_camera_position(camera_position, camera_path):
    """Save a PyVista camera position to JSON."""
    from pathlib import Path

    camera_path = Path(camera_path)
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "position": np.asarray(camera_position[0], dtype=float).tolist(),
        "focal_point": np.asarray(camera_position[1], dtype=float).tolist(),
        "view_up": np.asarray(camera_position[2], dtype=float).tolist(),
    }
    with open(camera_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_camera_position(camera_path):
    """Load a PyVista camera position from JSON."""
    with open(camera_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return [
        tuple(data["position"]),
        tuple(data["focal_point"]),
        tuple(data["view_up"]),
    ]


def add_partition_frame_overlay(plotter, nodes, partition_frame):
    """Draw translucent local X/Y/Z planes and axes for the partition frame."""
    try:
        import pyvista as pv
    except ImportError:
        return

    nodes = np.asarray(nodes, dtype=np.float64)
    origin = np.asarray(partition_frame["origin"], dtype=np.float64)
    rotation = rotation_matrix_from_euler(partition_frame["rotation_degrees"])
    axis_x = rotation[:, 0]
    axis_y = rotation[:, 1]
    axis_z = rotation[:, 2]

    span = np.ptp(nodes[:, :3], axis=0)
    size = float(np.max(span))
    if size <= 0:
        size = 1.0

    plane_size = size * 1.15
    axis_length = size * 0.30

    planes = [
        (axis_x, axis_y, "X-Y partition plane", "#4c78a8"),
        (axis_x, axis_z, "X-Z partition plane", "#f58518"),
        (axis_y, axis_z, "Y-Z partition plane", "#54a24b"),
    ]
    for direction_i, direction_j, name, color in planes:
        plane = pv.Plane(
            center=origin,
            direction=np.cross(direction_i, direction_j),
            i_size=plane_size,
            j_size=plane_size,
            i_resolution=1,
            j_resolution=1,
        )
        plotter.add_mesh(plane, color=color, opacity=0.12, name=name, pickable=False)

    axis_specs = [
        (axis_x, "Local X", "red"),
        (axis_y, "Local Y", "green"),
        (axis_z, "Local Z", "blue"),
    ]
    for axis, label, color in axis_specs:
        start = origin - axis * axis_length
        end = origin + axis * axis_length
        plotter.add_mesh(pv.Line(start, end), color=color, line_width=4, pickable=False)
        plotter.add_point_labels(
            np.asarray([end]),
            [label],
            font_size=12,
            text_color=color,
            point_size=0,
            shape_opacity=0.0,
            always_visible=True,
        )


def plot_vascular_tree(
    nodes,
    elements,
    scalar_values,
    strahler_order_values=None,
    scalar_name="Strahler order",
    cmap="jet",
    log_color=False,
    clim=None,
    radius_scale=1.0,
    n_sides=8,
    interactive_quality_toggle=True,
    high_quality_n_sides=24,
    background="white",
    window_size=(1200, 900),
    time_values=None,
    screenshot_path="Vascular_Tree.png",
    screenshot_scale=6,
    scale_bar_unit="",
    full_detail_quarter="All",
    outside_min_strahler=3,
    camera_position=None,
    camera_path=None,
    partition_frame=None,
    region_min_strahler=None,
    show_partition_frame=True,
):
    """Open an interactive PyVista vascular tree viewer.

    ``scalar_values`` can be either one value per vessel or a matrix with
    shape ``(n_elem, n_times)``. When time-dependent values are provided, a
    slider is added to select the displayed time frame.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for plotting. Install it with "
            "'python -m pip install pyvista'."
        ) from exc

    nodes = np.asarray(nodes, dtype=np.float32)
    elements = np.asarray(elements)
    scalar_values = np.asarray(scalar_values, dtype=np.float64)
    strahler_order_values = None if strahler_order_values is None else np.asarray(strahler_order_values).reshape(-1)

    if nodes.ndim != 2 or nodes.shape[1] < 3:
        raise ValueError("nodes must be a 2D array with at least 3 columns: X, Y, Z.")
    if elements.ndim != 2 or elements.shape[1] < 3:
        raise ValueError("elements must be a 2D array with at least 3 columns.")
    if scalar_values.ndim == 1:
        scalar_values = scalar_values[:, None]
    if scalar_values.ndim != 2 or scalar_values.shape[0] != elements.shape[0]:
        raise ValueError("scalar_values must have one row per vessel.")
    if strahler_order_values is not None and strahler_order_values.shape[0] != elements.shape[0]:
        raise ValueError("strahler_order_values must contain one value per vessel.")

    n_times = scalar_values.shape[1]
    if time_values is None:
        time_values = np.arange(n_times, dtype=np.float64)
    else:
        time_values = np.asarray(time_values, dtype=np.float64).reshape(-1)
        if time_values.shape[0] != n_times:
            raise ValueError("time_values length must match the number of scalar time frames.")

    connectivity = elements[:, :2].astype(np.int64, copy=False) - 1
    radii = elements[:, 2].astype(np.float32, copy=False) * np.float32(radius_scale)

    if np.any(connectivity < 0) or np.any(connectivity >= nodes.shape[0]):
        raise ValueError("Element connectivity contains node IDs outside the nodes array.")
    if np.any(radii <= 0):
        raise ValueError("All vessel radii must be positive to create tubes.")

    cmap = get_colormap(cmap)
    display_name = scalar_name
    display_clim = transform_color_limits(clim, log_color)
    quality_cache = {}

    def color_values_at(time_index):
        return transform_color_values(scalar_values[:, time_index], log_color)

    def build_fast_tubes(time_index):
        n_elem = elements.shape[0]
        color_values = color_values_at(time_index)
        tube_points = np.empty((2 * n_elem, 3), dtype=np.float32)
        tube_points[0::2] = nodes[connectivity[:, 0], :3]
        tube_points[1::2] = nodes[connectivity[:, 1], :3]

        lines = np.empty((n_elem, 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(0, 2 * n_elem, 2, dtype=np.int64)
        lines[:, 2] = lines[:, 1] + 1

        line_mesh = pv.PolyData(tube_points, lines=lines.ravel())
        line_mesh.point_data["Radius"] = np.repeat(radii, 2)
        line_mesh.point_data["Color Value"] = np.repeat(color_values, 2)
        line_mesh.cell_data["Color Value"] = color_values

        return line_mesh.tube(
            scalars="Radius",
            absolute=True,
            n_sides=n_sides,
            capping=True,
        )

    def build_high_quality_tubes(time_index):
        if time_index in quality_cache:
            return quality_cache[time_index]

        n_elem = elements.shape[0]
        color_values = color_values_at(time_index)
        lines = np.empty((n_elem, 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1:] = connectivity

        line_mesh = pv.PolyData(nodes[:, :3], lines=lines.ravel())
        line_mesh.cell_data["Radius"] = radii
        line_mesh.cell_data["Color Value"] = color_values

        point_mesh = line_mesh.cell_data_to_point_data()
        tube_mesh = point_mesh.tube(
            scalars="Radius",
            absolute=True,
            n_sides=high_quality_n_sides,
            capping=True,
        )
        quality_cache[time_index] = tube_mesh
        return tube_mesh

    def add_tube_mesh(plotter, tube_mesh, smooth_shading):
        return plotter.add_mesh(
            tube_mesh,
            scalars="Color Value",
            cmap=cmap,
            clim=display_clim,
            log_scale=log_color,
            show_scalar_bar=True,
            smooth_shading=smooth_shading,
            scalar_bar_args={
                "title": "",
                "vertical": False,
                "position_x": 0.20,
                "position_y": 0.04,
                "width": 0.60,
                "height": 0.08,
            },
        )

    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background(background)
    if camera_position is not None:
        plotter.camera_position = camera_position

    original_nodes = nodes.copy()
    if (
        strahler_order_values is not None
        and partition_frame is not None
        and region_min_strahler is not None
    ):
        keep_mask = vascular_tree_display_mask_partition(
            nodes,
            elements,
            strahler_order_values,
            partition_frame,
            region_min_strahler,
        )
        elements = elements[keep_mask]
        scalar_values = scalar_values[keep_mask]
        strahler_order_values = strahler_order_values[keep_mask]
        connectivity = elements[:, :2].astype(np.int64, copy=False) - 1
        radii = elements[:, 2].astype(np.float32, copy=False) * np.float32(radius_scale)
    elif strahler_order_values is not None and full_detail_quarter != "All":
        quarter_camera_position = camera_position if camera_position is not None else plotter.camera_position
        keep_mask = vascular_tree_display_mask_view_space(
            nodes,
            elements,
            strahler_order_values,
            quarter_camera_position,
            full_detail_quarter,
            outside_min_strahler,
        )
        elements = elements[keep_mask]
        scalar_values = scalar_values[keep_mask]
        strahler_order_values = strahler_order_values[keep_mask]
        connectivity = elements[:, :2].astype(np.int64, copy=False) - 1
        radii = elements[:, 2].astype(np.float32, copy=False) * np.float32(radius_scale)

    state = {
        "time_index": 0,
        "quality": False,
        "actor": None,
        "text": None,
    }

    def label_text():
        mode = "Paper quality" if state["quality"] else "Fast mode"
        if n_times == 1:
            return mode
        return f"{mode} | time {state['time_index'] + 1}/{n_times}: {time_values[state['time_index']]:.6g}"

    def current_mesh():
        if state["quality"]:
            return build_high_quality_tubes(state["time_index"])
        return build_fast_tubes(state["time_index"])

    def refresh():
        if state["actor"] is not None:
            plotter.remove_actor(state["actor"])
        if state["text"] is not None:
            plotter.remove_actor(state["text"])

        state["actor"] = add_tube_mesh(plotter, current_mesh(), state["quality"])
        state["text"] = plotter.add_text(
            label_text(),
            position=(60, window_size[1] - 38),
            font_size=10,
            color="black",
            name="vascular_viewer_text",
        )
        plotter.render()

    refresh()

    if interactive_quality_toggle:
        def set_quality_mode(enabled):
            state["quality"] = bool(enabled)
            refresh()

        plotter.add_checkbox_button_widget(
            set_quality_mode,
            value=False,
            position=(10, window_size[1] - 54),
            size=32,
            color_on="green",
            color_off="gray",
            background_color="white",
        )

    if n_times > 1:
        def set_time(value):
            state["time_index"] = int(np.clip(round(value), 0, n_times - 1))
            refresh()

        slider_widget = plotter.add_slider_widget(
            set_time,
            rng=(0, n_times - 1),
            value=0,
            title="Time (s)",
            pointa=(0.94, 0.08),
            pointb=(0.94, 0.94),
            style="modern",
            slider_width=0.01,
            tube_width=0.006,
        )
        try:
            slider_widget.GetRepresentation().GetTitleProperty().SetOrientation(90)
        except AttributeError:
            pass

    def save_screenshot():
        save_high_resolution_screenshot(plotter, screenshot_path, scale=screenshot_scale)

    def save_camera():
        if camera_path is None:
            return
        save_camera_position(plotter.camera_position, camera_path)

    def load_camera():
        if camera_path is None:
            return
        plotter.camera_position = load_camera_position(camera_path)
        plotter.render()

    plotter.add_key_event("s", save_screenshot)
    plotter.add_key_event("c", save_camera)
    plotter.add_key_event("l", load_camera)
    plotter.add_text(
        "Press s to save image | c to save view | l to load view",
        position="lower_left",
        font_size=9,
        color="black",
        name="save_hint_text",
    )
    plotter.add_text(
        display_name,
        position=(window_size[0] // 2 - 90, 8),
        font_size=11,
        color="black",
        name="colorbar_caption_text",
    )
    if partition_frame is not None and show_partition_frame:
        add_partition_frame_overlay(plotter, original_nodes, partition_frame)
    add_scale_bar(plotter, original_nodes, unit=scale_bar_unit)
    plotter.add_axes(labels_off=True)
    plotter.reset_camera()
    plotter.show()

    return plotter


def plot_strahler_tubes(
    nodes,
    elements,
    so,
    radius_scale=1.0,
    cmap="jet",
    n_sides=8,
    interactive_quality_toggle=False,
    high_quality_n_sides=24,
    background="white",
    window_size=(1200, 900),
    show=True,
    screenshot=None,
):
    """Plot vessels as PyVista tubes colored by Strahler order.

    Parameters
    ----------
    nodes:
        Node coordinate matrix with columns ``X, Y, Z``.
    elements:
        Vessel matrix with columns ``begin_node, end_node, radius, length``.
        Node numbers are expected to be 1-based, matching the MATLAB data.
    so:
        Strahler order for each vessel. Its length must match the number of
        rows in ``elements``.
    radius_scale:
        Multiplier applied to the radius column before creating tubes.
    cmap:
        Matplotlib/PyVista colormap name used for Strahler order coloring.
    n_sides:
        Number of sides used around each tube. Lower values are faster and
        use less memory for very large vascular trees.
    interactive_quality_toggle:
        If True, start with a fast segmented tube mesh and add a button that
        switches to a smoother, heavier paper-quality mesh.
    high_quality_n_sides:
        Number of tube sides used for the paper-quality mesh.
    background:
        Plot background color.
    window_size:
        PyVista window size.
    show:
        If True, display the interactive PyVista window.
    screenshot:
        Optional file path for saving a screenshot.

    Returns
    -------
    plotter:
        The PyVista plotter.
    tube_mesh:
        The generated tube mesh.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for plotting. Install it with "
            "'python -m pip install pyvista'."
        ) from exc

    nodes = np.asarray(nodes, dtype=np.float32)
    elements = np.asarray(elements)
    so = np.asarray(so, dtype=np.int64).reshape(-1)

    if nodes.ndim != 2 or nodes.shape[1] < 3:
        raise ValueError("nodes must be a 2D array with at least 3 columns: X, Y, Z.")
    if elements.ndim != 2 or elements.shape[1] < 3:
        raise ValueError("elements must be a 2D array with at least 3 columns.")
    if elements.shape[0] != so.shape[0]:
        raise ValueError("so must contain one Strahler order value per vessel.")

    connectivity = elements[:, :2].astype(np.int64, copy=False) - 1
    radii = elements[:, 2].astype(np.float32, copy=False) * np.float32(radius_scale)

    if np.any(connectivity < 0) or np.any(connectivity >= nodes.shape[0]):
        raise ValueError("Element connectivity contains node IDs outside the nodes array.")
    if np.any(radii <= 0):
        raise ValueError("All vessel radii must be positive to create tubes.")

    def build_fast_tubes():
        """Build separate tubes per vessel for responsive interaction."""
        n_elem = elements.shape[0]
        tube_points = np.empty((2 * n_elem, 3), dtype=np.float32)
        tube_points[0::2] = nodes[connectivity[:, 0], :3]
        tube_points[1::2] = nodes[connectivity[:, 1], :3]

        lines = np.empty((n_elem, 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(0, 2 * n_elem, 2, dtype=np.int64)
        lines[:, 2] = lines[:, 1] + 1

        line_mesh = pv.PolyData(tube_points, lines=lines.ravel())
        line_mesh.point_data["Radius"] = np.repeat(radii, 2)
        line_mesh.point_data["Strahler Order"] = np.repeat(so, 2)
        line_mesh.cell_data["Strahler Order"] = so

        return line_mesh.tube(
            scalars="Radius",
            absolute=True,
            n_sides=n_sides,
            capping=True,
        )

    def build_high_quality_tubes():
        """Build shared-node tubes for smoother-looking junctions."""
        n_elem = elements.shape[0]
        lines = np.empty((n_elem, 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1:] = connectivity

        line_mesh = pv.PolyData(nodes[:, :3], lines=lines.ravel())
        line_mesh.cell_data["Radius"] = radii
        line_mesh.cell_data["Strahler Order"] = so

        point_mesh = line_mesh.cell_data_to_point_data()
        return point_mesh.tube(
            scalars="Radius",
            absolute=True,
            n_sides=high_quality_n_sides,
            capping=True,
        )

    def add_tube_mesh(plotter, tube_mesh, smooth_shading):
        return plotter.add_mesh(
            tube_mesh,
            scalars="Strahler Order",
            cmap=cmap,
            show_scalar_bar=True,
            smooth_shading=smooth_shading,
            scalar_bar_args={"title": "Strahler Order"},
        )

    fast_tube_mesh = build_fast_tubes()
    tube_mesh = fast_tube_mesh

    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background(background)
    tube_actor = add_tube_mesh(plotter, fast_tube_mesh, smooth_shading=False)

    if interactive_quality_toggle:
        quality_state = {
            "actor": tube_actor,
            "fast_mesh": fast_tube_mesh,
            "quality_mesh": None,
            "text": None,
        }

        quality_state["text"] = plotter.add_text(
            "Fast mode",
            position="upper_left",
            font_size=10,
            color="black",
            name="quality_mode_text",
        )

        def set_quality_mode(enabled):
            plotter.remove_actor(quality_state["actor"])
            plotter.remove_actor(quality_state["text"])

            if enabled:
                if quality_state["quality_mesh"] is None:
                    quality_state["quality_mesh"] = build_high_quality_tubes()
                quality_state["actor"] = add_tube_mesh(
                    plotter,
                    quality_state["quality_mesh"],
                    smooth_shading=True,
                )
                label = "Paper quality"
            else:
                quality_state["actor"] = add_tube_mesh(
                    plotter,
                    quality_state["fast_mesh"],
                    smooth_shading=False,
                )
                label = "Fast mode"

            quality_state["text"] = plotter.add_text(
                label,
                position="upper_left",
                font_size=10,
                color="black",
                name="quality_mode_text",
            )
            plotter.render()

        plotter.add_checkbox_button_widget(
            set_quality_mode,
            value=False,
            position=(10, 10),
            size=32,
            color_on="green",
            color_off="gray",
            background_color="white",
        )

    plotter.add_axes()
    plotter.reset_camera()

    if show or screenshot is not None:
        plotter.show(screenshot=screenshot)

    return plotter, tube_mesh
