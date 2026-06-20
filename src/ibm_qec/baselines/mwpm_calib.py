import argparse
import json
import numpy as np
from pathlib import Path
import pymatching
from collections import defaultdict
import re 
import stim
import traceback



def build_stim_syndrome(data_shots, syndrome_shots, stim_to_qasm_map, detector_map):
    """
    Convert raw qasm-formatted shots into stim-style detector outcomes.
    
    Args:
        data_shots: np.ndarray, shape=(n_shots, n_data_bits)
        syndrome_shots: np.ndarray, shape=(n_shots, n_synd_bits)
        stim_to_qasm_map: dict mapping stim_idx -> {'qasm_idx': int, 'register': str, 'qubit': int}
        detector_map: list[list[int]], each inner list = stim indices whose parity defines that detector.
    
    Returns:
        det_out: np.ndarray, shape=(n_shots, n_detectors), dtype=bool
    """
    n_shots = data_shots.shape[0]
    n_meas = max(stim_to_qasm_map.keys()) + 1
    meas_matrix = np.zeros((n_shots, n_meas), dtype=np.uint8)

    # --- Step 1: Rebuild stim measurement record ---
    for stim_idx, info in stim_to_qasm_map.items():
        qasm_idx = info['qasm_idx']
        reg = info['register']

        if reg.startswith("c_data"):
            meas_matrix[:, stim_idx] = data_shots[:, qasm_idx]
        elif reg.startswith("c_syndrome"):
            meas_matrix[:, stim_idx] = syndrome_shots[:, qasm_idx]
        else:
            raise ValueError(f"Unknown register {reg}")

    # --- Step 2: Build detector outcomes ---
    n_det = len(detector_map)
    det_out = np.zeros((n_shots, n_det), dtype=bool)

    for d, stim_indices in enumerate(detector_map):
        det_out[:, d] = np.bitwise_xor.reduce(meas_matrix[:, stim_indices], axis=1).astype(bool)

    return det_out


def parse_backend_properties(backend_json):
    """Parse IBM backend properties JSON into structured dicts."""
    calib = {
        "T1": {}, "T2": {},
        "readout_error": {}, "prob_meas0_prep1": {}, "prob_meas1_prep0": {}, "readout_length": {},
        "single_qubit_gates": {},
        "two_qubit_gates": {}
    }

    # --- Per-qubit properties ---
    
    for q_idx, q_props in enumerate(backend_json['backend_properties'].get("qubits", [])):
        for entry in q_props:
            name, val = entry["name"], entry["value"]
            if name == "T1":
                calib["T1"][q_idx] = val*1000
            elif name == "T2":
                calib["T2"][q_idx] = val*1000
            elif name == "readout_error":
                calib["readout_error"][q_idx] = val
            elif name == "prob_meas0_prep1":
                calib["prob_meas0_prep1"][q_idx] = val
            elif name == "prob_meas1_prep0":
                calib["prob_meas1_prep0"][q_idx] = val
            elif name == "readout_length":
                calib["readout_length"][q_idx] = val

    # --- Gate properties (1Q and 2Q) ---
    for gate in backend_json['backend_properties'].get("gates", []):
        gtype = gate["gate"]
        qubits = tuple(gate["qubits"])
        gdata = {}
        for p in gate["parameters"]:
            gdata[p["name"]] = p["value"]

        if len(qubits) == 1:
            if gtype == 'reset':
                calib["single_qubit_gates"][(qubits[0], gtype)] = gdata
            elif gtype == 'id':
                calib["single_qubit_gates"][(qubits[0], gtype)] = gdata
        elif len(qubits) == 2:
            calib["two_qubit_gates"][(qubits[0], qubits[1], gtype)] = gdata

    return calib

def load_manifest(manifest_path: str):
    """Load manifest.json which maps circuits to metadata."""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    return manifest

def load_metadata(metadata_path: str):
    """Load manifest.json which maps circuits to metadata."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def load_results(results_path: str):
    """Load results.json which maps circuits to metadata."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results
def compute_average_error_rate(calib):
    """Calculate the mean error probability across calibration entries."""
    error_values = []
    def _maybe_add(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if np.isnan(v):
            return
        if 0 < v < 1:
            error_values.append(v)
    for val in calib.get("readout_error", {}).values():
        _maybe_add(val)
    for val in calib.get("prob_meas0_prep1", {}).values():
        _maybe_add(val)
    for val in calib.get("prob_meas1_prep0", {}).values():
        _maybe_add(val)
    for params in calib.get("single_qubit_gates", {}).values():
        _maybe_add(params.get("gate_error"))
    for params in calib.get("two_qubit_gates", {}).values():
        _maybe_add(params.get("gate_error"))
    if not error_values:
        print('no statistic parse')
        return 0.02
    return float(np.mean(error_values))

def sanitize_error_rate(value, fallback):
    """Return a reasonable error rate, falling back when data is missing/outlier."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    if np.isnan(v) or v <= 0 or v >= 1:
        return fallback
    return v

def add_idle_error(circuit, q, T1, T2, duration_ns):
    """Insert idle error channel for qubit q."""
    t = duration_ns 
    if T1 <= 0 or T2 <= 0:
        return
    px = py = 0.25 * (1 - np.exp(-t/T1))
    pz = 0.5 * (1 - np.exp(-t/T2)) - 0.25 * (1 - np.exp(-t/T1))
    px, py, pz = max(px, 0), max(py, 0), max(pz, 0)
    if px + py + pz > 0:
        circuit.append_operation("PAULI_CHANNEL_1", [q], [px, py, pz])



def map_stim_to_qasm(stim_history, syndrome_qasm, data_qasm, reg_names):
    """
    Build mapping: {stim_idx: {"qasm_idx": int, "register": str, "qubit": int}}
    - stim_history: {qubit: [stim_indices]}
    - syndrome_qasm: {qubit: [qasm_indices]}
    - data_qasm: {qubit: [qasm_indices]}
    - reg_names: dict with keys {"syndrome": <str>, "data": <str>}
    """
    mapping = {}

    # Handle data qubits
    for qubit, qasm_idxs in data_qasm.items():
        stim_idxs = stim_history.get(qubit, [])
        for qasm_idx, stim_idx in zip(qasm_idxs, stim_idxs):
            mapping[stim_idx] = {
                "qasm_idx": qasm_idx,
                "register": reg_names["data"],
                "qubit": qubit
            }

    # Handle syndrome qubits
    for qubit, qasm_idxs in syndrome_qasm.items():
        stim_idxs = stim_history.get(qubit, [])
        for qasm_idx, stim_idx in zip(qasm_idxs, stim_idxs):
            mapping[stim_idx] = {
                "qasm_idx": qasm_idx,
                "register": reg_names["syndrome"],
                "qubit": qubit
            }

    return mapping



def MWPM_decoding(
    job_dirs,
    target_distance=None,
    target_rounds=None,
    target_basis=None,
    allowed_logical_states=None,
    quiet=False,
):
    if allowed_logical_states is not None:
        allowed_logical_states = {int(s) for s in allowed_logical_states}
    LER_stats = defaultdict(list)
    aggregate_counts = defaultdict(lambda: {"shots": 0, "errors": 0})
    processed_jobs = []
    for jd in job_dirs:
        try:
            job_dir = jd
            # Load metadata and results using helper functions
            hardware_data = load_metadata(job_dir / "metadata.json")
            results_data = load_results(job_dir / "results.json")
            calib = parse_backend_properties(hardware_data)
            avg_error_rate = compute_average_error_rate(calib)
            job_used = False

            # Find the QASM file that starts with "transpiled_1"
            qasm_files = list(job_dir.glob("transpiled_0*.qasm"))
            if not qasm_files:
                raise FileNotFoundError("No QASM file starting with 'transpiled_1' found.")
            qasm_path = qasm_files[0]  # first match

            # Read QASM content
            with open(qasm_path, "r") as f:
                qasm_str = f.read()
            for circuit_result in results_data:

                all_chains = set()
                metadata_shot = circuit_result.get('metadata', {})
                intended_logical_state = metadata_shot.get('logical_state')
                if allowed_logical_states is not None and intended_logical_state not in allowed_logical_states:
                    continue
                basis = metadata_shot.get('basis')
                if target_basis and str(basis).upper() != str(target_basis).upper():
                    continue
                num_rounds = metadata_shot.get('n_syndrome_rounds')
                if target_rounds is not None and num_rounds != target_rounds:
                    continue
                meta_distance = metadata_shot.get('D')
                if target_distance is not None and meta_distance is not None and meta_distance != target_distance:
                    continue
                for reg_name in circuit_result['per_shot_cregs'].keys():
                    match = re.search(r"c_data_([\d_]+)", reg_name)
                    if match:
                        all_chains.add(match.group(1))
                bad_qubits = [key[0] for key, val in calib['single_qubit_gates'].items()
                            if val.get('gate_error', 0) == 1]
                for chain_str in all_chains:

                    # --- Step 1: Identify chain registers ---
                    data_decl = re.search(rf"bit\[\d+\] c_data_{chain_str};", qasm_str)
                    synd_decl = re.search(rf"bit\[\d+\] c_syndrome_{chain_str};", qasm_str)
                    if not data_decl or not synd_decl:
                        raise ValueError(f"Chain {chain_str} not found in QASM.")


                    qubits = [int(x) for x in chain_str.split("_")]

                    if any(q in bad_qubits for q in qubits):
                        continue
                    # --- Step 2: Filter operations ---
                    # Keep only lines with qubits in our chain

                    lines = []
                    for line in qasm_str.splitlines():
                        if not line or line.startswith("OPENQASM") or line.startswith("include") or line.startswith("bit["):
                            continue
                        # Extract qubit indices referenced in this line
                        q_in_line = [int(x) for x in re.findall(r"\$(\d+)", line)]
                        if q_in_line and all(q in qubits for q in q_in_line):
                            lines.append(line.strip())



                    data_qubits = set()
                    syndrome_qubits = set()
                    circuit = stim.Circuit()
                    data_meas_history_qasm = defaultdict(list)
                    syndrome_meas_history_qasm = defaultdict(list)
                    for line in lines:
                        if line.startswith("rz(pi/2)"):
                            q = int(re.search(r"\$(\d+)", line).group(1))
                            circuit.append_operation("S", [q])
                        elif line.startswith("sx"):
                            q = int(re.search(r"\$(\d+)", line).group(1))
                            circuit.append("SQRT_X", [q])
                        elif line.startswith("cz"):
                            q1, q2 = map(int, re.findall(r"\$(\d+)", line))
                            circuit.append("CZ", [q1, q2])
                        elif line.startswith("reset"):
                            q = int(re.search(r"\$(\d+)", line).group(1))
                            circuit.append("R", [q])
                        elif "measure" in line:
                            # Example: c_syndrome_6_5_4_3_16[4] = measure $5;
                            q = int(re.search(r"\$(\d+)", line).group(1))
                            idx = int(re.search(r"\[(\d+)\]", line).group(1))
                            circuit.append("M", [q])

                            if "c_data" in line:
                                data_qubits.add(q)
                                data_meas_history_qasm[q].append(idx)
                            elif "c_syndrome" in line:
                                syndrome_qubits.add(q)
                                syndrome_meas_history_qasm[q].append(idx)


                    ##########try optimal number of stabilizer style###########
                    noisy = stim.Circuit()
                    D = len(data_qubits)
                    current_cycle_qubits_measure = set()
                    T_cycle_ns = {q: [] for q in qubits}
                    # Keep track of measurement indices for each qubit
                    meas_history = {q: [] for q in qubits}
                    meas_counter = -1
                    detector_map = []
                    syndrome_neighbor = {q: set() for q in syndrome_qubits}



                    for inst in circuit:
                        name = inst.name
                        targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
                        # Gate noise
                        if name in ["S", "SQRT_X", "H"]:
                            noisy.append(name, targets)
                            for q in targets:
                                raw_p = calib["single_qubit_gates"].get((q,"id"), {}).get("gate_error")
                                p = sanitize_error_rate(raw_p, avg_error_rate)
                                noisy.append("DEPOLARIZE1", [q], p)
                                # print('noisy r?#ate',p)
                                # T_cycle_ns += calib["single_qubit_gates"].get((q,"id"), {}).get("gate_length", 20)
                        elif name == "CZ":
                            noisy.append(name, targets)
                            for i in range(0, len(targets), 2):
                                q1, q2 = targets[i], targets[i+1]
                                key = (min(q1, q2), max(q1, q2), "cz")
                                raw_p = calib["two_qubit_gates"].get(key, {}).get("gate_error")
                                p = sanitize_error_rate(raw_p, avg_error_rate)
                                if p > 0.5:
                                    p = avg_error_rate
                                noisy.append("DEPOLARIZE2", [q1, q2], p)
                                # T_cycle_ns += calib["two_qubit_gates"].get(key, {}).get("gate_length", 0.01)
                                if q1 in syndrome_qubits and q2 not in syndrome_neighbor[q1]:
                                    syndrome_neighbor[q1].add(q2)
                                elif q2 in syndrome_qubits and q1 not in syndrome_neighbor[q2]:
                                    syndrome_neighbor[q2].add(q1)
                              #  print('CZ noisy rate',p)
                        elif name == "M":
                            for q in targets:
                                raw_p = calib["readout_error"].get(q)
                                p = sanitize_error_rate(raw_p, avg_error_rate)
                                noisy.append("X_ERROR", [q],p)
                                noisy.append("M", [q])
                             #   print('M noisy rate',p)
                                # --- Temporal detectors ---
                                meas_counter += 1
                                meas_history[q].append(meas_counter)
                                if len(meas_history[q]) > 1 and q in syndrome_qubits:
                                    prev = meas_history[q][-2] - meas_counter -1
                                    noisy.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(prev)])
                                    detector_map.append([meas_history[q][-2],meas_counter])
                                elif len(meas_history[q]) == 1 and q in syndrome_qubits:
                                    noisy.append("DETECTOR", [stim.target_rec(-1)])
                                    detector_map.append([meas_counter])
                                noisy.append_operation('R',q)

                                for other in qubits:
                                    T_cycle_ns = calib["single_qubit_gates"].get((other,"reset"), {}).get("gate_length", 1220)

                                    if other != q:
                                        T1 = calib["T1"].get(q, 100e-6)
                                        T2 = calib["T2"].get(q, 100e-6)
                                        add_idle_error(noisy, other, T1, T2, T_cycle_ns)


                    # --- Final round detectors ---
                    for a in syndrome_qubits:
                        if not meas_history[a]:
                            continue
                        last_synd_meas = meas_history[a][-1]

                        neighbors = syndrome_neighbor[a]  # implement this mapping
                        rec_targets = [stim.target_rec(last_synd_meas-meas_counter-1)]
                        for d in neighbors:
                            if meas_history[d]:
                                data_meas = meas_history[d][0]
                                rec_targets.append(stim.target_rec(data_meas-meas_counter-1))

                        noisy.append("DETECTOR", rec_targets)
                        detector_map.append([last_synd_meas] + [meas_history[d][0] for d in neighbors])

                    # noisy.append("OBSERVABLE_INCLUDE",[stim.target_rec(data_meas-meas_counter-1)] ,0)
                    syndrome_reg_name = f"c_syndrome_{chain_str}"
                    data_reg_name = f"c_data_{chain_str}"
                    reg_names     = {"syndrome": syndrome_reg_name, "data": data_reg_name}

                    mapping = map_stim_to_qasm(meas_history, syndrome_meas_history_qasm, data_meas_history_qasm, reg_names)
                    # for q, d in mapping.items():
                    #     print(q, d)
                    # find stim index whose qasm_idx == 0 and register starts with c_data
                    obs_stim_idx = next(
                        stim_idx for stim_idx, info in mapping.items()
                        if info['qasm_idx'] == 0 and info['register'].startswith("c_data")
                    )

                    # add observable definition
                    noisy.append(
                        "OBSERVABLE_INCLUDE",
                        [stim.target_rec(obs_stim_idx - meas_counter - 1)],
                        0  # observable id
                    )


                    syndrome_shots = np.array(circuit_result['per_shot_cregs'][syndrome_reg_name])
                    data_shots = np.array(circuit_result['per_shot_cregs'][data_reg_name])
                    # shape: (n_shots,)
                    logical_obs = data_shots[:, 0]

                    det_out = build_stim_syndrome(data_shots, syndrome_shots, mapping, detector_map)


                    model = noisy.detector_error_model(decompose_errors=True,approximate_disjoint_errors=True)
                    matching = pymatching.Matching.from_detector_error_model(model)
                    predictions = matching.decode_batch(det_out)
                    error = 0
                    for shot in range(len(det_out)):
                        if predictions[shot][0] != logical_obs[shot]:
                            error += 1
                    shots = len(det_out)
                    key = (D, num_rounds, str(basis).upper(), int(intended_logical_state))
                    if target_distance is not None and D != target_distance:
                        continue
                    LER_stats[key].append(error / shots if shots else 0.0)
                    aggregate_counts[key]["shots"] += shots
                    aggregate_counts[key]["errors"] += error
                    job_used = True
        except Exception as e:
            if not quiet:
                print(f"Error processing job directory: {jd}")
                print("Detailed error:")
                traceback.print_exc()
                print("Skipping...")
            continue
        if job_used:
            processed_jobs.append(job_dir)
    return {
        "ler_samples": {k: v for k, v in LER_stats.items()},
        "aggregate": {k: val for k, val in aggregate_counts.items()},
        "jobs": [job for job in processed_jobs],
    }


def catalog_results(results_root: Path):
    combos = defaultdict(set)
    for job_dir in sorted(results_root.iterdir()):
        if not job_dir.is_dir():
            continue
        results_path = job_dir / "results.json"
        metadata_path = job_dir / "metadata.json"
        if not results_path.is_file() or not metadata_path.is_file():
            continue
        try:
            circuits = load_results(results_path)
        except Exception:
            continue
        for circuit in circuits:
            metadata = circuit.get("metadata", {}) or {}
            basis = metadata.get("basis")
            rounds = metadata.get("n_syndrome_rounds")
            logical_state = metadata.get("logical_state")
            distance = metadata.get("D")
            if distance is None:
                for key in circuit.get("per_shot_cregs", {}):
                    if key.startswith("c_data_"):
                        distance = len(key[len("c_data_"):].split("_"))
                        break
            try:
                distance = int(distance)
                rounds = int(rounds)
                logical_state = int(logical_state)
            except (TypeError, ValueError):
                continue
            if basis is None:
                continue
            combos[(distance, rounds, str(basis).upper(), logical_state)].add(job_dir)
    return {key: sorted(paths, key=lambda p: str(p)) for key, paths in combos.items()}


def summarise_decoding(decoder_output, key):
    distance, rounds, basis, logical_state = key
    samples = decoder_output["ler_samples"].get(key, [])
    counts = decoder_output["aggregate"].get(key, {"shots": 0, "errors": 0})
    shots = counts["shots"]
    errors = counts["errors"]
    if logical_state == 1:
        errors = shots - errors
    ler = errors / shots if shots else None
    ler_mean = float(np.mean(samples)) if samples else None
    ler_std = float(np.std(samples)) if len(samples) > 1 else None
    return {
        "distance": distance,
        "rounds": rounds,
        "basis": basis,
        "logical_state": logical_state,
        "shots": shots,
        "errors": errors,
        "ler": ler,
        "ler_mean": ler_mean,
        "ler_std": ler_std,
        "ler_samples": samples,
        "jobs_processed": [str(p) for p in decoder_output["jobs"]],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run MWPM decoding on IBM repetition-code experiment results."
    )
    parser.add_argument("-d", "--distance", type=int, help="Target code distance.")
    parser.add_argument("-t", "--rounds", type=int, help="Number of syndrome rounds.")
    parser.add_argument(
        "-b",
        "--basis",
        type=str,
        choices=("X", "Z"),
        help="Preparation basis to analyse.",
    )
    parser.add_argument(
        "-s",
        "--logical-state",
        type=int,
        choices=(0, 1),
        help="Logical state to analyse (0 or 1). Default: all states present.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results_testing/",
        help="Directory containing job subdirectories with results.json/metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "MWPM_calib_results_training_data",
        help="Directory where summary JSON files are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of job directories per combination.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="If set, do not write summary JSON files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-job warnings.",
    )
    args = parser.parse_args()

    results_root = args.results_root
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root '{results_root}' does not exist.")

    catalog = catalog_results(results_root)
    if not catalog:
        print(f"No valid job directories found in '{results_root}'.")
        return

    requested_combos = []
    if args.distance is not None or args.rounds is not None or args.basis is not None:
        if args.distance is None or args.rounds is None or args.basis is None:
            parser.error("Please specify distance, rounds, and basis together.")
        basis = args.basis.upper()
        candidate_states = (
            [args.logical_state]
            if args.logical_state is not None
            else sorted({key[3] for key in catalog if key[:3] == (args.distance, args.rounds, basis)})
        )
        if not candidate_states:
            candidate_states = [0, 1]
        requested_combos = [(args.distance, args.rounds, basis, state) for state in candidate_states]
    else:
        requested_combos = sorted(catalog.keys())
        if args.logical_state is not None:
            requested_combos = [combo for combo in requested_combos if combo[3] == args.logical_state]

    if not requested_combos:
        print("No matching configurations found.")
        return
    requested_combos = sorted(set(requested_combos))

    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for combo in requested_combos:
        distance, rounds, basis, logical_state = combo
        job_dirs = catalog.get(combo, [])
        if args.limit is not None:
            job_dirs = job_dirs[: args.limit]
        if not job_dirs:
            if not args.quiet:
                print(f"No job directories found for D={distance}, r={rounds}, basis={basis}, state={logical_state}.")
            continue

        decoder_output = MWPM_decoding(
            job_dirs,
            target_distance=distance,
            target_rounds=rounds,
            target_basis=basis,
            allowed_logical_states={logical_state},
            quiet=args.quiet,
        )
        summary = summarise_decoding(decoder_output, combo)
        shots = summary["shots"]
        ler_display = f"{summary['ler']:.4%}" if summary["ler"] is not None else "N/A"
        print(f"\nD={distance}, r={rounds}, basis={basis}, state={logical_state}")
        print(f"  Jobs processed: {len(summary['jobs_processed'])}")
        print(f"  Shots: {shots:,}")
        print(f"  Errors: {summary['errors']:,}")
        print(f"  LER: {ler_display}")
        if summary["ler_mean"] is not None:
            mean = f"{summary['ler_mean']:.4%}"
            std = f"{summary['ler_std']:.4%}" if summary["ler_std"] is not None else "N/A"
            print(f"  Sample mean: {mean}  Sample std: {std}")

        if not args.no_save:
            filename = f"mwpm_test_D{distance}_T{rounds}_{basis}_S{logical_state}.json"
            output_path = args.output_dir / filename
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            if not args.quiet:
                print(f"  Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
