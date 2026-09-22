import argparse
import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

# Define amino acid scale values for a basic sequence-based detectability score (native fallback)
# These represent typical scales for hydrophobicity (Kyte-Doolittle) and ionizability
AA_GRAVY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def calculate_native_detectability(seq: str) -> float:
    """
    Calculates an in-silico peptide detectability score (0.1 to 1.0) based on sequence properties.
    This acts as a high-quality native fallback representing peptide 'flyability'
    using sequence length, hydrophobicity (GRAVY), and charge-inducing residues.
    """
    seq = str(seq).upper().strip()
    if not seq or any(c not in AA_GRAVY for c in seq):
        return 0.5 # Default fallback for modified or invalid sequences
    
    # 1. Length factor: Tryptic peptides fly best between 7 and 25 residues
    length = len(seq)
    if length < 6:
        len_score = 0.2
    elif length <= 15:
        len_score = 0.9
    elif length <= 25:
        len_score = 0.7
    else:
        len_score = 0.4
        
    # 2. Hydrophobicity (GRAVY) factor: Extreme hydrophobicity or hydrophilicity reduces flight
    gravy = sum(AA_GRAVY[aa] for aa in seq) / length
    # Optimal GRAVY is typically close to neutral-to-slightly hydrophobic (-0.5 to 1.0)
    if -0.5 <= gravy <= 1.0:
        gravy_score = 0.95
    elif -1.5 <= gravy < -0.5 or 1.0 < gravy <= 2.0:
        gravy_score = 0.60
    else:
        gravy_score = 0.25
        
    # 3. Mass Spectrometry Friendly Residues: Proline (P) and hydrophobic C-termini fly well.
    # High counts of Acidic residues (D, E) or internal Basic residues (without cleavage) can reduce positive ionization.
    p_count = seq.count('P')
    e_count = seq.count('E')
    d_count = seq.count('D')
    ionized_score = 0.5 + (0.1 * p_count) - (0.05 * (e_count + d_count))
    ionized_score = max(0.2, min(1.0, ionized_score))
    
    # Combine scores into a final probability-like detectability metric [0.1, 1.0]
    score = (len_score * 0.4) + (gravy_score * 0.4) + (ionized_score * 0.2)
    return max(0.1, min(1.0, score))


# ---------------------------------------------------------------------------
# DeepMSPeptide (Serrano et al., Bioinformatics 2020) - direct in-process use
# ---------------------------------------------------------------------------
# Same encoding as DeepMSPeptide.py in the original repo (max length 81, post-padding with 0).
DEEPMS_AA_CODES = {
    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11,
    'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19, 'W': 20, 'Y': 21, 'V': 22
}
DEEPMS_MAX_LEN = 81


def _load_deepms_model(model_path: Path):
    """
    The pretrained model is a legacy Keras 2 .h5 file (TF 1.13 era). It does NOT load with Keras 3
    (default in recent TensorFlow / Colab), so we use the `tf_keras` package (pip install tf_keras).
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    try:
        import tf_keras as keras
    except ImportError:
        from tensorflow import keras  # works only if this TensorFlow still ships Keras 2
    return keras.models.load_model(str(model_path), compile=False)


def predict_detectability_deepms(peptides: list, model_path: Path, min_score: float = 0.05,
                                 batch_size: int = 2048) -> dict:
    """
    Detectability = 1 - model output, exactly as DeepMSPeptide.py writes it in its 'Prob' column
    (verified against the repo's inputExample_Predictions.txt).
    - Modification annotations like (Oxidation) or [+15.99] are stripped: the model is sequence-only.
    - Peptides that still contain non-standard letters or are longer than 81 aa are not predicted;
      they receive the median score of the predicted peptides (neutral).
    - Scores are clipped to [min_score, 1.0] so that dividing intensity by the score cannot explode.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"DeepMSPeptide model not found: {model_path}")

    clean = {p: re.sub(r"\(.*?\)|\[.*?\]", "", str(p)).upper().strip() for p in peptides}
    valid = {p: c for p, c in clean.items()
             if c and len(c) <= DEEPMS_MAX_LEN and all(a in DEEPMS_AA_CODES for a in c)}
    unique_seqs = sorted(set(valid.values()))
    if not unique_seqs:
        raise ValueError("No valid peptide sequences to predict.")

    X = np.zeros((len(unique_seqs), DEEPMS_MAX_LEN), dtype="int32")
    for i, seq in enumerate(unique_seqs):
        X[i, :len(seq)] = [DEEPMS_AA_CODES[a] for a in seq]

    print(f"[INFO] Loading DeepMSPeptide model: {model_path}")
    model = _load_deepms_model(Path(model_path))
    print(f"[INFO] Predicting {len(unique_seqs)} unique cleaned sequences...")
    raw = model.predict(X, batch_size=batch_size, verbose=0)[:, 0]
    seq_to_score = dict(zip(unique_seqs, 1.0 - raw))

    scores = {p: float(np.clip(seq_to_score[c], min_score, 1.0)) for p, c in valid.items()}
    n_skipped = len(peptides) - len(valid)
    if n_skipped:
        neutral = float(np.median(list(scores.values())))
        for p in peptides:
            scores.setdefault(p, neutral)
        print(f"[WARNING] {n_skipped} peptides could not be encoded (non-standard letters or >81 aa); "
              f"assigned the median score {neutral:.4f}.")
    return scores


def predict_detectability_scores(peptides: list, use_deep_learning: bool = False,
                                 model_path: Path = None, min_score: float = 0.05) -> dict:
    """
    use_deep_learning=True  -> DeepMSPeptide only. If it fails, the script STOPS (no silent fallback),
                               so that results can never be mixed up with the heuristic model.
    use_deep_learning=False -> native sequence-property heuristic (GRAVY + length + ionization).
    """
    if use_deep_learning:
        try:
            return predict_detectability_deepms(peptides, model_path, min_score=min_score)
        except Exception as e:
            sys.exit(f"FATAL: DeepMSPeptide prediction failed: {type(e).__name__}: {e}\n"
                     f"       Run without --use_deep_learning to use the native heuristic instead.")

    return {pep: calculate_native_detectability(pep) for pep in peptides}


def main():
    parser = argparse.ArgumentParser(
        description="Step 2 - Block 2: Detectability (Flyability) Correction & PPM Normalization"
    )
    parser.add_argument(
        "--input_tsv",
        default="step2_out/step2_peptides_lca_assigned.tsv",
        help="Path to the LCA-assigned peptide TSV generated by Block 1."
    )
    parser.add_argument(
        "--outdir",
        default="./step2_out",
        help="Directory to save the corrected peptide TSV output."
    )
    parser.add_argument(
        "--use_deep_learning",
        action="store_true",
        help="Toggle to execute deep-learning prediction via DeepMSPeptide."
    )
    parser.add_argument(
        "--model_path",
        default="./DeepMSPeptide/DeepMSPeptide/model_2_1D.h5",
        help="Path to the DeepMSPeptide pretrained model (model_2_1D.h5)."
    )
    parser.add_argument(
        "--min_detectability",
        type=float,
        default=0.05,
        help="Lower bound applied to DeepMSPeptide scores before dividing intensities (default 0.05)."
    )
    args = parser.parse_args()

    input_path = Path(args.input_tsv)
    outdir = Path(args.outdir)

    # 1. Verify input file exists
    if not input_path.exists():
        sys.exit(f"FATAL: LCA assigned peptide file not found at: {input_path}\n"
                 f"        Please make sure to run Step 2 Block 1 first!")

    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LCA-assigned peptide table: {input_path}...")
    df = pd.read_csv(input_path, sep="\t")

    # Verify expected columns
    required_cols = ["run", "condition", "proteins", "peptide", "intensity", "tree_node"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(f"FATAL: Missing required columns in input table: {missing}")

    # 2. Extract unique non-decoy peptides to predict
    # We do not predict decoys; they will receive a default neutral detectability score of 1.0
    is_decoy = df["tree_node"].astype(str).str.contains("DECOY_")
    unique_peptides = df.loc[~is_decoy, "peptide"].unique().tolist()
    
    print(f"Total rows in dataset: {len(df)}")
    print(f"Unique target peptides to predict: {len(unique_peptides)}")

    # 3. Predict detectability scores
    print("Calculating/Predicting peptide detectability scores (flyability)...")
    peptide_to_score = predict_detectability_scores(
        unique_peptides,
        use_deep_learning=args.use_deep_learning,
        model_path=Path(args.model_path),
        min_score=args.min_detectability,
    )

    # Map scores back to the DataFrame (decoys get 1.0, preserving raw intensity ratio)
    df["detectability_score"] = df["peptide"].map(peptide_to_score)
    df.loc[is_decoy, "detectability_score"] = 1.0
    df["detectability_score"] = df["detectability_score"].fillna(1.0)

    # 4. Apply Physical Correction: Intensity_corrected = Intensity_observed / Detectability_score
    print("Applying ionization efficiency correction to peptide intensities...")
    df["intensity_corrected"] = df["intensity"] / df["detectability_score"]

    # 5. Apply PPM Normalization on Corrected Intensities (Independent per MS Run)
    print("Calculating run-specific totals and scaling to parts per million (PPM)...")
    # Group by run and calculate total sum of corrected intensities
    df["run_total_corrected"] = df.groupby("run")["intensity_corrected"].transform("sum")
    
    # Scale to PPM
    df["intensity_ppm"] = (df["intensity_corrected"] / df["run_total_corrected"]) * 1e6

    # 6. Quality Control Metrics and Summary
    print("\n" + "="*50)
    print("         BLOCK 2: CORRECTION & NORMALIZATION QUALITY CONTROL")
    print("="*50)
    avg_score = df.loc[~is_decoy, "detectability_score"].mean()
    min_score = df.loc[~is_decoy, "detectability_score"].min()
    max_score = df.loc[~is_decoy, "detectability_score"].max()
    print(f"Target Detectability Scores:")
    print(f"  -> Mean: {avg_score:.4f}")
    print(f"  -> Min:  {min_score:.4f}")
    print(f"  -> Max:  {max_score:.4f}")
    
    # Verify that each run normalizes perfectly to 1,000,000 PPM
    run_ppm_sums = df.groupby("run")["intensity_ppm"].sum()
    runs_ok = np.allclose(run_ppm_sums, 1e6)
    print(f"\nPPM Normalization Status:")
    print(f"  -> All {len(run_ppm_sums)} runs sum to exactly 1,000,000 PPM: {runs_ok}")
    if not runs_ok:
        print(f"  -> WARNING: Run sums range from {run_ppm_sums.min():,.2f} to {run_ppm_sums.max():,.2f}")
    print("="*50 + "\n")

    # 7. Save output table
    output_path = outdir / "step2_peptides_corrected_ppm.tsv"
    # Keep key columns to keep file size optimized
    columns_to_save = [
        "run", "condition", "proteins", "peptide", "tree_node", 
        "intensity", "detectability_score", "intensity_corrected", "intensity_ppm"
    ]
    df[columns_to_save].to_csv(output_path, sep="\t", index=False)
    print(f"[SUCCESS] Block 2 finished! Processed table saved to: {output_path}")

if __name__ == "__main__":
    main()
