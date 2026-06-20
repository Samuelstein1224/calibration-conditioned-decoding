# Preprocess raw IBM experiment data into ML-ready tensors (QASM3 compatible)

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import warnings

import numpy as np
import torch
# --- MODIFIED: Import the correct QASM3 parser ---
from qiskit import qasm3
from qiskit.converters import circuit_to_dag
from torch_geometric.data import Data
from tqdm import tqdm
from ibm_qec.data.prepare_oneshot import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare experimental data for ML decoding.")
    parser.add_argument("job_dir", type=str, help="Path to job results directory.")
    parser.add_argument("--out-dir", type=str, default="processed_data", help="Directory to save processed tensors.")
    args = parser.parse_args()
    process_and_save_data(args.job_dir, args.out_dir)