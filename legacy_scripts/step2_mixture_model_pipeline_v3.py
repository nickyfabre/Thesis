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
    # METHOD 3: MIXTURE MODEL (Proportional Split of Shared Peptides)
    # ==========================================================================
    print("Processing Method 3: Mixture Model...")
    # We do the Mixture Model calculation *separately for each individual run* to preserve run-level physical ratios.
    runs = sorted(df_clean["run"].unique())
    mixture_rows = []

    for run in runs:
        df_run = df_clean[df_clean["run"] == run].copy()
        cond = df_run["condition"].iloc[0]
        
        # 1. Calculate unique peptide sum (anchor) for each species in this run
        df_run_unique = df_run[df_run["num_descendants"] == 1].copy()
        df_run_unique["species_label"] = df_run_unique["tree_node"].map(lambda n: node_to_desc[n][0])
        
        anchors = df_run_unique.groupby("species_label")["intensity_ppm"].sum().to_dict()
        # Fill missing species with 0.0 anchors
        for label in species_df["Label"]:
            if label not in anchors:
                anchors[label] = 0.0
                
        # 2. Iterate through all peptide measurements in this run and split them
        for _, row in df_run.iterrows():
            node = row["tree_node"]
            intensity = row["intensity_ppm"]
            peptide = row["peptide"]
            
            descendants = node_to_desc.get(node, [])
            if not descendants:
                continue
                
            if len(descendants) == 1:
                # Strictly unique peptide - goes 100% to this leaf
                mixture_rows.append({
                    "run": run,
                    "condition": cond,
                    "species_label": descendants[0],
                    "peptide": peptide,
                    "split_intensity": intensity
                })
            else:
                # Shared peptide - split proportionally according to unique anchors
                desc_anchors = {lbl: anchors.get(lbl, 0.0) for lbl in descendants}
                total_anchor = sum(desc_anchors.values())
                
                if total_anchor > 0.0:
                    for lbl in descendants:
                        weight = desc_anchors[lbl] / total_anchor
                        if weight > 0:
                            mixture_rows.append({
                                "run": run,
                                "condition": cond,
                                "species_label": lbl,
                                "peptide": peptide,
                                "split_intensity": intensity * weight
                            })
                else:
                    # If no candidate species has any unique peptides in this run, split equally
                    eq_weight = 1.0 / len(descendants)
                    for lbl in descendants:
                        mixture_rows.append({
                            "run": run,
                            "condition": cond,
                            "species_label": lbl,
                            "peptide": peptide,
                            "split_intensity": intensity * eq_weight
                        })

    df_mixture = pd.DataFrame(mixture_rows)
    
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

    # Convert PPM to Percentages (%) per method so they are in the same scale [0.0, 100.0] as the Expected Recipe
    for cond in conditions:
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
    # CONSOLE DIAGNOSTIC BENCHMARK SUMMARY & ERROR METRICS
    # ==========================================================================
    print("\n" + "="*85)
    print("           METAPROTEOMICS QUANTIFICATION METHODOLOGY COMPARATIVE BENCHMARK (V3)")
    print("="*85)

    # Store metrics for plotting
    metrics_records = []

    for cond in conditions:
        sub = eval_df[(eval_df["Condition"] == cond) & (eval_df["Expected_Pct"] > 0.0)].copy()
        
        print(f"\nCondition: {cond}")
        print("-" * 75)
        
        for col_name, title, _ in methods:
            # We use all valid expected > 0 for error metrics, not just those where measured > 0,
            # to penalize methods that completely fail to quantify an expected species (measured=0).
            expected_vals = sub["Expected_Pct"]
            measured_vals = sub[col_name]

            # Calculate MAE and RMSE
            mae = np.mean(np.abs(expected_vals - measured_vals))
            rmse = np.sqrt(np.mean((expected_vals - measured_vals)**2))

            valid_log = sub[sub[col_name] > 0.0]
            if not valid_log.empty:
                # Calculate Log Pearson
                log_pears = np.log10(valid_log["Expected_Pct"]).corr(np.log10(valid_log[col_name]), method="pearson")
                log_spear = np.log10(valid_log["Expected_Pct"]).corr(np.log10(valid_log[col_name]), method="spearman")

                # Leaves only
                leaves = valid_log[valid_log["Node_Type"] == "Leaf"]
                r_leaves = np.log10(leaves["Expected_Pct"]).corr(np.log10(leaves[col_name]), method="pearson") if len(leaves) > 2 else np.nan
            else:
                log_pears, log_spear, r_leaves = np.nan, np.nan, np.nan
            
            
            # Calculate Coefficient of Variation (CV)
            cv = np.std(measured_vals) / np.mean(measured_vals) * 100 if np.mean(measured_vals) > 0 else np.nan

            method_short = title.split(' ')[0]
            print(f"  -> {method_short:<14} | Pearson r: {log_pears:6.4f} | MAE: {mae:6.2f}% | RMSE: {rmse:6.2f}% | CV: {cv:6.2f}%")

            metrics_records.append({
                "Condition": cond,
                "Method": method_short,
                "MAE": mae,
                "RMSE": rmse,
                "Pearson_R": log_pears,
                "CV": cv
            })

    print("="*85 + "\n")

    # ==========================================================================
    # ERROR METRICS VISUALIZATION (MAE/RMSE Bar Charts)
    # ==========================================================================
    print("Generating Error Metrics Visualization (MAE & RMSE)...")
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(outdir / "step2_mixture_model_error_metrics.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Set up bar positions
    x = np.arange(len(conditions))
    width = 0.25

    method_labels = [m[1].split(' ')[0] for m in methods]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green matching the scatterplots

    for i, method_short in enumerate(method_labels):
        method_data = metrics_df[metrics_df["Method"] == method_short]

        # MAE Bars
        rects1 = ax1.bar(x + (i - 1) * width, method_data["MAE"], width, label=method_short, color=colors[i], edgecolor='black')
        # RMSE Bars
        rects2 = ax2.bar(x + (i - 1) * width, method_data["RMSE"], width, label=method_short, color=colors[i], edgecolor='black')

        # Add values on top of bars
        ax1.bar_label(rects1, padding=3, fmt='%.1f')
        ax2.bar_label(rects2, padding=3, fmt='%.1f')

    # Formatting MAE plot
    ax1.set_ylabel('Mean Absolute Error (%)', fontweight="bold")
    ax1.set_title('Mean Absolute Error (MAE) by Condition', fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Formatting RMSE plot
    ax2.set_ylabel('Root Mean Square Error (%)', fontweight="bold")
    ax2.set_title('Root Mean Square Error (RMSE) by Condition', fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle("Error Metrics Benchmark: Lower Error indicates better quantification accuracy", fontsize=14, fontweight="bold")
    plt.tight_layout()

    error_plot_path = outdir / "step2_mixture_model_error_barchart.png"
    plt.savefig(error_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Error metrics bar chart saved to: {error_plot_path}")

    # ==========================================================================
    # ERROR METRICS VISUALIZATION (Bland-Altman/Residuals Plot)
    # ==========================================================================
    print("Generating Residuals Plot...")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, (col_name, method_title, color) in enumerate(zip([m[0] for m in methods], method_labels, colors)):
        ax = axs[i]

        # Gather data for this method across all conditions
        all_expected = []
        all_diffs = []

        for cond in conditions:
            sub = eval_df[(eval_df["Condition"] == cond) & (eval_df["Expected_Pct"] > 0.0)]
            expected = sub["Expected_Pct"]
            measured = sub[col_name]
            diff = measured - expected

            # Use scatter for each point
            ax.scatter(expected, diff, label=cond, alpha=0.7, edgecolor='black', s=50)

            all_expected.extend(expected.tolist())
            all_diffs.extend(diff.tolist())

        # Draw a horizontal line at 0 (perfect agreement)
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Perfect Agreement')

        # Calculate overall mean difference and limits of agreement
        if all_diffs:
            mean_diff = np.mean(all_diffs)
            std_diff = np.std(all_diffs)

            ax.axhline(mean_diff, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label=f'Mean Bias ({mean_diff:.1f}%)')
            ax.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='+1.96 SD')
            ax.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='-1.96 SD')

        ax.set_title(f'{method_title} Residuals', fontweight="bold")
        ax.set_xlabel('Expected Abundance (%)', fontweight="bold")
        if i == 0:
            ax.set_ylabel('Difference (Measured - Expected) %', fontweight="bold")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 2:
            # Only put legend on the last plot so it doesn't clutter all of them
            ax.legend(loc='upper right', fontsize='small')

    plt.suptitle("Residuals Analysis: Agreement between Measured and Expected Abundances", fontsize=14, fontweight="bold")
    plt.tight_layout()

    residuals_plot_path = outdir / "step2_mixture_model_residuals_plot.png"
    plt.savefig(residuals_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Residuals plot saved to: {residuals_plot_path}")

if __name__ == "__main__":
    main()
