import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try importing ete3 for taxonomy. If not present, we will fallback to Simulation Mode
try:
    from ete3 import NCBITaxa
    HAS_TAXONOMY_LIB = True
except ImportError:
    HAS_TAXONOMY_LIB = False

def load_species_metadata(known_path: Path) -> pd.DataFrame:
    sep = "\t" if str(known_path).lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(known_path, sep=sep)
    required = {"Label", "Species"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"FATAL: Species metadata missing columns {missing}; found {list(df.columns)}")
    df['Label'] = df['Label'].astype(str).str.strip()
    df['Species'] = df['Species'].astype(str).str.strip()
    return df

def get_descendants_map(df_corrected: pd.DataFrame, species_df: pd.DataFrame):
    """
    Creates a mapping from tree_node name (3-letter Label for leaves, or scientific name for LCA)
    to a list of descendant species Labels.
    """
    node_to_descendants = {}
    unique_nodes = df_corrected["tree_node"].dropna().unique().tolist()

    unique_nodes = [n for n in unique_nodes if "DECOY_" not in str(n) and n not in ["Unknown", "Error_Calculating_LCA"]]

    species_to_label = dict(zip(species_df["Species"], species_df["Label"]))
    label_to_species = dict(zip(species_df["Label"], species_df["Species"]))

    if not HAS_TAXONOMY_LIB:
        for node in unique_nodes:
            node_str = str(node)
            if node_str.startswith("LCA_Node_of_"):
                labels = node_str.replace("LCA_Node_of_", "").split("_")
                node_to_descendants[node] = [l for l in labels if l in label_to_species]
            else:
                if node in label_to_species:
                    node_to_descendants[node] = [node]
                elif node in species_to_label:
                    node_to_descendants[node] = [species_to_label[node]]
                else:
                    node_to_descendants[node] = []
        return node_to_descendants

    try:
        ncbi = NCBITaxa()
        species_list = species_df['Species'].dropna().unique().tolist()
        name2taxid = ncbi.get_name_translator(species_list)

        prefix_to_taxid = {}
        taxid_to_label = {}
        for _, row in species_df.iterrows():
            prefix = row['Label']
            species_name = row['Species']
            if species_name in name2taxid:
                tid = name2taxid[species_name][0]
                prefix_to_taxid[prefix] = tid
                taxid_to_label[tid] = prefix

        valid_taxids = [tid for tid in prefix_to_taxid.values()]
        tree = ncbi.get_topology(valid_taxids)
        tree_leaf_names = set(tree.get_leaf_names())

        all_node_taxids = [int(node.name) for node in tree.traverse()]
        translator = ncbi.get_taxid_translator(all_node_taxids)

        taxid_to_descendants = {}
        for node in tree.traverse():
            taxid = int(node.name)
            leaves = [int(leaf.name) for leaf in node.get_leaves()]
            descendant_labels = [taxid_to_label[l] for l in leaves if l in taxid_to_label]
            taxid_to_descendants[taxid] = descendant_labels

        name_to_descendants = {}
        for taxid, desc in taxid_to_descendants.items():
            sc_name = translator.get(taxid, f"TaxID_{taxid}")
            name_to_descendants[sc_name] = desc

        for node in unique_nodes:
            if node in name_to_descendants:
                node_to_descendants[node] = name_to_descendants[node]
            else:
                # Leaf node: checking if it is already a 3-letter label
                if node in label_to_species:
                    node_to_descendants[node] = [node]
                elif node in species_to_label:
                    node_to_descendants[node] = [species_to_label[node]]
                else:
                    node_to_descendants[node] = []

        return node_to_descendants
    except Exception as e:
        print(f"[WARNING] NCBI Tree navigation failed ({e}). Falling back to text-pattern mapping.")
        for node in unique_nodes:
            node_str = str(node)
            if node_str.startswith("LCA_Node_of_"):
                labels = node_str.replace("LCA_Node_of_", "").split("_")
                node_to_descendants[node] = [l for l in labels if l in label_to_species]
            else:
                if node in label_to_species:
                    node_to_descendants[node] = [node]
                elif node in species_to_label:
                    node_to_descendants[node] = [species_to_label[node]]
                else:
                    node_to_descendants[node] = []
        return node_to_descendants

def main():
    parser = argparse.ArgumentParser(
        description="Step 2 - Block 4: Mixture Model vs Unique Only vs LCA Rígido Benchmark Pipeline (V3 with Ultra-Clean Labels)"
    )
    parser.add_argument(
        "--input_tsv",
        default="step2_out/step2_peptides_corrected_ppm.tsv",
        help="Path to the corrected peptide PPM TSV file generated by Block 2."
    )
    parser.add_argument(
        "--known",
        default="suplementarydata1_speciesamounts.csv",
        help="Path to the physical recipe CSV."
    )
    parser.add_argument(
        "--sdrf",
        default="fileSDRF_Run_1_2.sdrf.tsv",
        help="Path to the SDRF experimental design file."
    )
    parser.add_argument(
        "--outdir",
        default="./step2_out",
        help="Directory to save final comparison results and plots."
    )
    args = parser.parse_args()

    input_path = Path(args.input_tsv)
    known_path = Path(args.known)
    sdrf_path = Path(args.sdrf)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        sys.exit(f"FATAL: Input file not found at '{input_path}'. Please run Block 1 and 2 first!")
    if not known_path.exists():
        sys.exit(f"FATAL: Ground truth metadata file not found at: {known_path}")

    print("Loading datasets...")
    df = pd.read_csv(input_path, sep="\t")
    species_df = load_species_metadata(known_path)

    # Replicate mapping
    if sdrf_path.exists():
        print(f"Loading SDRF replicate mapping from: {sdrf_path}...")
        sdrf = pd.read_csv(sdrf_path, sep="\t")
        rep_map = pd.DataFrame()
        rep_map["run"] = range(1, len(sdrf) + 1)
        rep_map["biological_replicate"] = sdrf["characteristics[biological replicate]"]
        rep_map["technical_replicate"] = sdrf["comment[technical replicate]"]
        rep_map["condition"] = sdrf["factor value[condition]"]
    else:
        print("[WARNING] SDRF file not found. Generating default 24-run replicate mapping fallback...")
        conditions = ["Constant"] * 8 + ["Protein"] * 8 + ["Uneven"] * 8
        bioreps = [1, 2, 3, 4, 1, 2, 3, 4] * 3
        techreps = ["Run1", "Run1", "Run1", "Run1", "Run2", "Run2", "Run2", "Run2"] * 3
        rep_map = pd.DataFrame({
            "run": range(1, 25),
            "condition": conditions,
            "biological_replicate": bioreps,
            "technical_replicate": techreps
        })

    if "biological_replicate" not in df.columns or "technical_replicate" not in df.columns:
        df = df.drop(columns=["biological_replicate", "technical_replicate"], errors="ignore")
        df = df.merge(rep_map[["run", "biological_replicate", "technical_replicate"]], on="run", how="left")

    # Clean decoys/unknowns
    is_decoy = df["tree_node"].astype(str).str.contains("DECOY_")
    df_clean = df[~is_decoy & ~df["tree_node"].isin(["Unknown", "Error_Calculating_LCA"])].copy()

    print("Mapping taxonomic nodes to descendant species...")
    node_to_desc = get_descendants_map(df_clean, species_df)

    # Let's map species name to label and vice versa
    species_to_label = dict(zip(species_df["Species"], species_df["Label"]))
    label_to_species = dict(zip(species_df["Label"], species_df["Species"]))
    label_to_normalized_ug_C = dict(zip(species_df["Label"], species_df["Input Protein Amount in C (ug) normalized"]))
    label_to_normalized_ug_P = dict(zip(species_df["Label"], species_df["Input Protein Amount in P (ug) normalized"]))
    label_to_normalized_ug_U = dict(zip(species_df["Label"], species_df["Input Protein Amount in U (ug) normalized"]))

    # ==========================================================================
    # METHOD 1: UNIQUE ONLY
    # ==========================================================================
    print("Processing Method 1: Unique Only...")
    # Find strictly unique peptides (where descendants has exactly 1 species)
    df_clean["num_descendants"] = df_clean["tree_node"].map(lambda n: len(node_to_desc.get(n, [])))
    df_unique = df_clean[df_clean["num_descendants"] == 1].copy()

    # Map tree_node directly to descendant label
    df_unique["species_label"] = df_unique["tree_node"].map(lambda n: node_to_desc[n][0])

    # Collapse replicates for Unique Only
    # A. Technical collapse (mean)
    unique_biorep = df_unique.groupby(["species_label", "condition", "biological_replicate"], as_index=False)["intensity_ppm"].mean()
    # B. Biological average
    unique_avg = unique_biorep.groupby(["species_label", "condition"], as_index=False)["intensity_ppm"].mean()

    # Pivot
    unique_wide = unique_avg.pivot_table(index="species_label", columns="condition", values="intensity_ppm", fill_value=0.0).reset_index()

    # ==========================================================================
    # METHOD 2: LCA RÍGIDO (Sum of signals under nodes)
    # ==========================================================================
    print("Processing Method 2: LCA Rígido...")
    # A. Technical collapse per node
    lca_biorep = df_clean.groupby(["tree_node", "condition", "biological_replicate"], as_index=False)["intensity_ppm"].mean()
    # B. Biological average
    lca_avg = lca_biorep.groupby(["tree_node", "condition"], as_index=False)["intensity_ppm"].mean()

    lca_wide = lca_avg.pivot_table(index="tree_node", columns="condition", values="intensity_ppm", fill_value=0.0).reset_index()

    # ==========================================================================
    # METHOD 3: MIXTURE MODEL (Expectation-Maximization)
    # ==========================================================================
    print("Processing Method 3: Mixture Model (EM Algorithm)...")
    runs = sorted(df_clean["run"].unique())
    mixture_rows = []

    # EM hyperparameters
    max_iters = 50
    tolerance = 1e-4

    for run in runs:
        df_run = df_clean[df_clean["run"] == run].copy()
        cond = df_run["condition"].iloc[0]

        # Initial M-step: Set initial species abundances using unique peptides
        df_run_unique = df_run[df_run["num_descendants"] == 1].copy()
        df_run_unique["species_label"] = df_run_unique["tree_node"].map(lambda n: node_to_desc[n][0])

        abundances = df_run_unique.groupby("species_label")["intensity_ppm"].sum().to_dict()

        all_labels_in_run = set()
        for d in df_run["tree_node"].map(lambda n: node_to_desc.get(n, [])):
            all_labels_in_run.update(d)

        for label in all_labels_in_run:
            if label not in abundances:
                abundances[label] = 1e-9 # Small non-zero start if no unique peptides

        # EM Loop
        for iteration in range(max_iters):
            prev_abundances = abundances.copy()

            # Reset accumulated abundances for the next M-step
            new_abundances = {lbl: 0.0 for lbl in abundances}
            run_splits = [] # Store splits for this iteration to potentially use as final

            # E-step: Allocate peptide intensities based on current species abundances
            for _, row in df_run.iterrows():
                node = row["tree_node"]
                intensity = row["intensity_ppm"]
                peptide = row["peptide"]

                descendants = node_to_desc.get(node, [])
                if not descendants:
                    continue

                if len(descendants) == 1:
                    lbl = descendants[0]
                    new_abundances[lbl] += intensity
                    run_splits.append({
                        "run": run, "condition": cond,
                        "species_label": lbl, "peptide": peptide,
                        "split_intensity": intensity
                    })
                else:
                    desc_abunds = {lbl: abundances.get(lbl, 0.0) for lbl in descendants}
                    total_abund = sum(desc_abunds.values())

                    if total_abund > 0:
                        for lbl in descendants:
                            weight = desc_abunds[lbl] / total_abund
                            alloc = intensity * weight
                            new_abundances[lbl] += alloc
                            if alloc > 0:
                                run_splits.append({
                                    "run": run, "condition": cond,
                                    "species_label": lbl, "peptide": peptide,
                                    "split_intensity": alloc
                                })
                    else:
                        # Fallback equal split
                        eq_weight = 1.0 / len(descendants)
                        alloc = intensity * eq_weight
                        for lbl in descendants:
                            new_abundances[lbl] += alloc
                            run_splits.append({
                                "run": run, "condition": cond,
                                "species_label": lbl, "peptide": peptide,
                                "split_intensity": alloc
                            })

            # Check convergence (M-step updates are implicit in new_abundances)
            max_diff = 0.0
            for lbl in abundances:
                diff = abs(new_abundances[lbl] - prev_abundances.get(lbl, 0.0))
                if prev_abundances.get(lbl, 0.0) > 0:
                    diff /= prev_abundances[lbl]
                max_diff = max(max_diff, diff)

            abundances = new_abundances

            if max_diff < tolerance:
                break

        # Append converged splits for this run
        mixture_rows.extend(run_splits)

    df_mixture = pd.DataFrame(mixture_rows)

    if df_mixture.empty:
        # Prevent errors if dataframe is empty
        df_mixture = pd.DataFrame(columns=["run", "condition", "species_label", "peptide", "split_intensity"])

    # Sum peptide split intensities per species and run
    species_run_mixture = df_mixture.groupby(["species_label", "run", "condition"], as_index=False)["split_intensity"].sum()
    species_run_mixture = species_run_mixture.merge(rep_map[["run", "biological_replicate", "technical_replicate"]], on="run", how="left")

    # Collapse replicates for Mixture Model
    # A. Technical collapse (mean)
    mix_biorep = species_run_mixture.groupby(["species_label", "condition", "biological_replicate"], as_index=False)["split_intensity"].mean()
    # B. Biological average
    mix_avg = mix_biorep.groupby(["species_label", "condition"], as_index=False)["split_intensity"].mean()

    # Pivot
    mix_wide = mix_avg.pivot_table(index="species_label", columns="condition", values="split_intensity", fill_value=0.0).reset_index()

    # ==========================================================================
    # BENCHMARK COMPARISON ASSEMBLY & EVALUATION
    # ==========================================================================
    print("Assembling final evaluation datasets...")

    conditions = ["Constant", "Protein", "Uneven"]
    all_evaluation_rows = []

    for node, descendants in node_to_desc.items():
        if not descendants:
            continue

        num_species = len(descendants)
        node_type = "Leaf" if num_species == 1 else "Internal Node"

        # Expected Relative Abundances (%)
        sum_C_recipe = sum(label_to_normalized_ug_C.get(l, 0.0) for l in descendants)
        sum_P_recipe = sum(label_to_normalized_ug_P.get(l, 0.0) for l in descendants)
        sum_U_recipe = sum(label_to_normalized_ug_U.get(l, 0.0) for l in descendants)

        recipe_sums = {"Constant": sum_C_recipe, "Protein": sum_P_recipe, "Uneven": sum_U_recipe}

        for cond in conditions:
            expected_pct = recipe_sums[cond]

            # --- MEASURED VAL 1: UNIQUE ONLY ---
            # Sum descendant leaf values
            unique_desc = unique_wide[unique_wide["species_label"].isin(descendants)]
            unique_val = unique_desc[cond].sum() if not unique_desc.empty else 0.0

            # --- MEASURED VAL 2: LCA RÍGIDO ---
            lca_sub = lca_wide[lca_wide["tree_node"].map(lambda n: set(node_to_desc.get(n, [])).issubset(set(descendants)))]
            lca_val = lca_sub[cond].sum() if not lca_sub.empty else 0.0

            # --- MEASURED VAL 3: MIXTURE MODEL ---
            # Sum descendant leaf values
            mix_desc = mix_wide[mix_wide["species_label"].isin(descendants)]
            mix_val = mix_desc[cond].sum() if not mix_desc.empty else 0.0

            all_evaluation_rows.append({
                "Node": node,
                "Node_Type": node_type,
                "Num_Species": num_species,
                "Condition": cond,
                "Expected_Pct": expected_pct,
                "Unique_PPM": unique_val,
                "LCA_PPM": lca_val,
                "Mix_PPM": mix_val
            })

    eval_df = pd.DataFrame(all_evaluation_rows)

    if eval_df.empty:
        eval_df = pd.DataFrame(columns=["Node", "Node_Type", "Num_Species", "Condition", "Expected_Pct", "Unique_PPM", "LCA_PPM", "Mix_PPM", "Unique_Pct", "LCA_Pct", "Mix_Pct"])

    # Convert PPM to Percentages (%) per method so they are in the same scale [0.0, 100.0] as the Expected Recipe
    for cond in conditions:
        if eval_df.empty:
            break
        sub_idx = eval_df["Condition"] == cond
        # Normalize Unique
        total_unique = eval_df.loc[sub_idx & (eval_df["Node_Type"] == "Leaf"), "Unique_PPM"].sum()
        if total_unique > 0:
            eval_df.loc[sub_idx, "Unique_Pct"] = (eval_df.loc[sub_idx, "Unique_PPM"] / total_unique) * 100.0
        else:
            eval_df.loc[sub_idx, "Unique_Pct"] = 0.0

        # Normalize LCA
        total_lca = eval_df.loc[sub_idx & (eval_df["Node_Type"] == "Leaf"), "LCA_PPM"].sum()
        if total_lca > 0:
            eval_df.loc[sub_idx, "LCA_Pct"] = (eval_df.loc[sub_idx, "LCA_PPM"] / total_lca) * 100.0
        else:
            eval_df.loc[sub_idx, "LCA_Pct"] = 0.0

        # Normalize Mixture Model
        total_mix = eval_df.loc[sub_idx & (eval_df["Node_Type"] == "Leaf"), "Mix_PPM"].sum()
        if total_mix > 0:
            eval_df.loc[sub_idx, "Mix_Pct"] = (eval_df.loc[sub_idx, "Mix_PPM"] / total_mix) * 100.0
        else:
            eval_df.loc[sub_idx, "Mix_Pct"] = 0.0

    # Save results to disk
    eval_df.to_csv(outdir / "step2_mixture_model_benchmark_results.csv", index=False)
    print(f"[SUCCESS] Tabular evaluation dataset saved to: {outdir}/step2_mixture_model_benchmark_results.csv")

    # ==========================================================================
    # GENERATE 3X3 DIAGNOSTIC PLOT MATRIX
    # ==========================================================================
    print("Generating 3x3 Benchmark comparison plot matrix...")
    fig, axs = plt.subplots(3, 3, figsize=(22, 21))

    methods = [
        ("Unique_Pct", "Unique Peptides Only (Discard Shared)", "Blues"),
        ("LCA_Pct", "Rigid LCA Assignment (Paso 2)", "Oranges"),
        ("Mix_Pct", "Mixture Model (Proportional Split)", "Greens")
    ]

    for row_idx, cond in enumerate(conditions):
        cond_df = eval_df[eval_df["Condition"] == cond].copy()

        # Determine shared max depth for color scaling
        max_depth = cond_df["Num_Species"].max()
        norm = plt.Normalize(1, max_depth)

        for col_idx, (col_name, method_title, cmap_base) in enumerate(methods):
            ax = axs[row_idx, col_idx]

            # Filter non-zero expected to avoid log(0)
            plot_df = cond_df[(cond_df["Expected_Pct"] > 0.0) & (cond_df[col_name] > 0.0)].copy()

            if plot_df.empty:
                ax.set_title(f"No Data\n{cond} - {method_title}")
                continue

            x_vals = plot_df["Expected_Pct"]
            y_vals = plot_df[col_name]

            # Log10-scale correlations
            pearson_r = np.log10(x_vals).corr(np.log10(y_vals), method="pearson")
            spearman_R = np.log10(x_vals).corr(np.log10(y_vals), method="spearman")

            # Separate Leaf and Internal
            leaves = plot_df[plot_df["Node_Type"] == "Leaf"]
            internal = plot_df[plot_df["Node_Type"] == "Internal Node"]

            # Plot Leaves (circles)
            sc_leaves = ax.scatter(
                leaves["Expected_Pct"], leaves[col_name],
                c=leaves["Num_Species"], cmap=cmap_base, norm=norm,
                marker="o", edgecolors="black", s=130, alpha=0.9, zorder=4, label="Leaf (Species)"
            )

            # Plot Internal Nodes (triangles)
            sc_internal = ax.scatter(
                internal["Expected_Pct"], internal[col_name],
                c=internal["Num_Species"], cmap=cmap_base, norm=norm,
                marker="^", edgecolors="black", s=150, alpha=0.9, zorder=4, label="Internal (Ancestor)"
            )

            # Identity line
            min_lim = 0.005
            max_lim = 105.0
            ax.plot([min_lim, max_lim], [min_lim, max_lim], "r--", alpha=0.7, linewidth=1.5, label="1:1 Identity", zorder=2)

            # Formatting
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(min_lim, max_lim)
            ax.set_ylim(min_lim, max_lim)
            ax.grid(True, which="both", linestyle="--", alpha=0.4, zorder=1)

            # Labels
            ax.set_xlabel("Expected Recipe Abundance (%)", fontsize=10, fontweight="bold")
            ax.set_ylabel(f"Measured {method_title.split(' ')[0]} Abundance (%)", fontsize=10, fontweight="bold")

            # Add titles with correlations
            ax.set_title(f"{cond} | {method_title}\nLog Pearson r: {pearson_r:.4f} | Spearman R: {spearman_R:.4f}",
                         fontsize=11, fontweight="bold", pad=8)

            # Local colorbar
            cbar = plt.colorbar(sc_leaves, ax=ax, pad=0.02)
            cbar.set_label("Descendant Species Count", fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor="white")

            # Selected Annotations for visual tracing (V3 with Clean short labels and fewer ancestors)
            for _, row in plot_df.iterrows():
                name = str(row["Node"])
                is_leaf = row["Node_Type"] == "Leaf"

                # 1. Annotate ALL circles (Species/Leaf nodes) because their labels are short 3-letter codes
                if is_leaf:
                    # Let's offset slightly to avoid overlapping the marker
                    ax.text(
                        row["Expected_Pct"] * 1.25,
                        row[col_name] * 0.85,
                        name,
                        fontsize=7.5,
                        alpha=0.9,
                        color="black",
                        fontweight="bold"
                    )
                else:
                    # 2. Only annotate ancestors (triangles) if they are biologically significant (Expected_Pct > 2.0)
                    # and truncate their names aggressively to avoid messy overlays
                    if row["Expected_Pct"] > 2.0:
                        short_ancestor = name
                        if short_ancestor.startswith("LCA_Node_of_"):
                            short_ancestor = short_ancestor.replace("LCA_Node_of_", "LCA_")

                        if len(short_ancestor) > 12:
                            short_ancestor = short_ancestor[:10] + ".."

                        ax.text(
                            row["Expected_Pct"] * 1.25,
                            row[col_name] * 0.85,
                            short_ancestor,
                            fontsize=7,
                            alpha=0.8,
                            color="darkred",
                            fontweight="semibold",
                            style="italic"
                        )

    plt.suptitle("METAPROTEOMICS TAXONOMIC QUANTIFICATION BENCHMARK MATRIX\n"
                 "Unique Peptides Only vs. Strict LCA vs. Proportional Mixture Model across all 3 conditions (Log-Log Scale)",
                 fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()

    plot_path = outdir / "step2_mixture_model_benchmark_matrix.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] 3x3 Benchmark plot matrix saved to: {plot_path}")

    # ==========================================================================
    # CONSOLE DIAGNOSTIC BENCHMARK SUMMARY
    # ==========================================================================
    print("\n" + "="*85)
    print("           METAPROTEOMICS QUANTIFICATION METHODOLOGY COMPARATIVE BENCHMARK (V2)")
    print("="*85)
    for cond in conditions:
        sub = eval_df[(eval_df["Condition"] == cond) & (eval_df["Expected_Pct"] > 0.0)].copy()

        print(f"\nCondition: {cond}")
        print("-" * 50)

        for col_name, title, _ in methods:
            valid = sub[sub[col_name] > 0.0]
            # Calculate Log Pearson
            log_pears = np.log10(valid["Expected_Pct"]).corr(np.log10(valid[col_name]), method="pearson")
            log_spear = np.log10(valid["Expected_Pct"]).corr(np.log10(valid[col_name]), method="spearman")

            # Leaves only
            leaves = valid[valid["Node_Type"] == "Leaf"]
            r_leaves = np.log10(leaves["Expected_Pct"]).corr(np.log10(leaves[col_name]), method="pearson") if len(leaves) > 2 else np.nan

            print(f"  -> {title.split(' ')[0]:<14} | Total Nodes Pearson r: {log_pears:.4f} | Spearman R: {log_spear:.4f} | Leaves Pearson r: {r_leaves:.4f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
