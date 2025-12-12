#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import List, Tuple, Optional

import numpy as np


def _read_mtx_coordinate(path: str) -> Tuple[int, int, int, List[Tuple[int, int, float]]]:
    """Read MatrixMarket coordinate real general.

    Returns: (M, K, nnz, entries) where entries are 0-based (r, c, v).
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("%")]

    if not lines:
        raise ValueError(f"Empty or comment-only mtx: {path}")

    header = lines[0].split()
    if len(header) < 3:
        raise ValueError(f"Invalid mtx header line: {lines[0]}")

    m, k, nnz = map(int, header[:3])
    entries: List[Tuple[int, int, float]] = []

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        r1 = int(float(parts[0]))
        c1 = int(float(parts[1]))
        v = float(parts[2])
        # MatrixMarket is 1-based
        r0 = r1 - 1
        c0 = c1 - 1
        if 0 <= r0 < m and 0 <= c0 < k:
            entries.append((r0, c0, v))

    # nnz in file can be trusted, but we use parsed entries length for safety.
    return m, k, nnz, entries


def _dense_a_from_entries(m: int, k: int, entries: List[Tuple[int, int, float]], dtype: np.dtype) -> np.ndarray:
    a = np.zeros((m, k), dtype=dtype)
    for r0, c0, v in entries:
        a[r0, c0] = dtype.type(v) if hasattr(dtype, "type") else dtype(v)
    return a


def _infer_n_from_bin(path: str, elem_bytes: int, leading: int, name: str) -> int:
    size = os.path.getsize(path)
    if size % elem_bytes != 0:
        raise ValueError(f"{name} bin size {size} not divisible by elem_bytes={elem_bytes}: {path}")
    elems = size // elem_bytes
    if leading <= 0:
        raise ValueError(f"Invalid leading dimension for {name}: {leading}")
    if elems % leading != 0:
        raise ValueError(
            f"{name} bin elements {elems} not divisible by leading={leading}. "
            f"Cannot infer N from {path}"
        )
    return elems // leading


def _load_mnk_txt(path: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    m = k = n = None
    if not os.path.exists(path):
        return m, k, n
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip().upper()
            val = val.strip()
            if key == "M":
                m = int(val)
            elif key == "K":
                k = int(val)
            elif key == "N":
                n = int(val)
    return m, k, n


def _configure_print(full: bool, linewidth: int) -> None:
    if full:
        np.set_printoptions(threshold=np.inf, linewidth=linewidth)
    else:
        np.set_printoptions(linewidth=linewidth)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print full A(.mtx), B(x2_gm.bin float16), C(golden.bin float32) matrices"
    )
    ap.add_argument(
        "--mtx",
        help="Path to A .mtx (e.g. temp_input/aligned_32.mtx)",
        required=True,
    )
    ap.add_argument(
        "--b",
        help="Path to B bin (float16, shape KxN). Default: <mtx_dir>/<name>/x2_gm.bin",
        default=None,
    )
    ap.add_argument(
        "--c",
        help="Path to C golden bin (float32, shape MxN). Default: <mtx_dir>/<name>/golden.bin",
        default=None,
    )
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional N override (otherwise inferred from bins or mnk.txt)",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Print full arrays (no truncation)",
    )
    ap.add_argument(
        "--linewidth",
        type=int,
        default=200,
        help="Numpy print linewidth",
    )

    args = ap.parse_args()

    mtx_path = os.path.abspath(args.mtx)
    if not os.path.exists(mtx_path):
        raise FileNotFoundError(mtx_path)

    mtx_dir = os.path.dirname(mtx_path)
    base = os.path.splitext(os.path.basename(mtx_path))[0]
    sample_dir = os.path.join(mtx_dir, base)

    b_path = os.path.abspath(args.b) if args.b else os.path.join(sample_dir, "x2_gm.bin")
    c_path = os.path.abspath(args.c) if args.c else os.path.join(sample_dir, "golden.bin")

    if not os.path.exists(b_path):
        raise FileNotFoundError(f"B bin not found: {b_path}")
    if not os.path.exists(c_path):
        raise FileNotFoundError(f"golden bin not found: {c_path}")

    m_mtx, k_mtx, nnz_file, entries = _read_mtx_coordinate(mtx_path)

    # Optional metadata file.
    mnk_path = os.path.join(sample_dir, "mnk.txt")
    m_txt, k_txt, n_txt = _load_mnk_txt(mnk_path)

    m = m_txt if m_txt is not None else m_mtx
    k = k_txt if k_txt is not None else k_mtx

    if m != m_mtx or k != k_mtx:
        raise ValueError(
            f"M/K mismatch: mtx says M={m_mtx},K={k_mtx} but mnk.txt says M={m},K={k}. "
            f"Check {mtx_path} and {mnk_path}."
        )

    n = args.n if args.n is not None else n_txt

    if n is None:
        # Prefer infer from B (float16, KxN)
        n = _infer_n_from_bin(b_path, elem_bytes=2, leading=k, name="B")

    # Load matrices
    a = _dense_a_from_entries(m, k, entries, dtype=np.float16)

    b = np.fromfile(b_path, dtype=np.float16)
    expected_b = k * n
    if b.size != expected_b:
        # Try infer N from actual B size if user-provided N is wrong.
        if b.size % k == 0:
            n2 = b.size // k
            raise ValueError(f"B size mismatch: got {b.size} elems, expected {expected_b}. Inferred N={n2} from file.")
        raise ValueError(f"B size mismatch: got {b.size} elems, expected {expected_b} (K={k}, N={n}).")
    b = b.reshape((k, n))

    c = np.fromfile(c_path, dtype=np.float32)
    expected_c = m * n
    if c.size != expected_c:
        if c.size % m == 0:
            n2 = c.size // m
            raise ValueError(f"C size mismatch: got {c.size} elems, expected {expected_c}. Inferred N={n2} from file.")
        raise ValueError(f"C size mismatch: got {c.size} elems, expected {expected_c} (M={m}, N={n}).")
    c = c.reshape((m, n))

    _configure_print(full=args.full, linewidth=args.linewidth)

    print(f"[A] {mtx_path}")
    print(f"  shape=({m},{k}), nnz(file_header)={nnz_file}, nnz(parsed)={len(entries)}, dtype=float16")
    print(a)
    print()

    print(f"[B] {b_path}")
    print(f"  shape=({k},{n}), dtype=float16")
    print(b)
    print()

    print(f"[C] {c_path}")
    print(f"  shape=({m},{n}), dtype=float32")
    print(c)


if __name__ == "__main__":
    main()
