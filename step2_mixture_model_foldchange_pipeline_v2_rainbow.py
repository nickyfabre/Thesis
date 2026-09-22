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
        description="Step 2 - Block 5: Log2 Fold Change Comparison Pipeline (V2_Rainbow - Uniform Rainbow Scale)"
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
        print(f"[INFO] Input file '{input_path}' not found. Check if you run Block 1 and 2.")
        sys.exit(1)
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

    label_to_normalized_ug_C = dict(zip(species_df["Label"], species_df["Input Protein Amount in C (ug) normalized"]))
    label_to_normalized_ug_P = dict(zip(species_df["Label"], species_df["Input Protein Amount in P (ug) normalized"]))
    label_to_normalized_ug_U = dict(zip(species_df["Label"], species_df["Input Protein Amount in U (ug) normalized"]))

    comparisons = [
        ("Uneven", "Constant", "U", "C"),
        ("Protein", "Constant", "P", "C"),
        ("Uneven", "Protein", "U", "P")
    ]

    # ==========================================================================
    # METHOD 1: UNIQUE ONLY LOG2 FC
    # ==========================================================================
    print("Processing Method 1: Unique Only...")
    df_clean["num_descendants"] = df_clean["tree_node"].map(lambda n: len(node_to_desc.get(n, [])))
    df_unique = df_clean[df_clean["num_descendants"] == 1].copy()
    df_unique["species_label"] = df_unique["tree_node"].map(lambda n: node_to_desc[n][0])

    # Collapse replicates at peptide level
    unique_pep_biorep = df_unique.groupby(["peptide", "species_label", "condition", "biological_replicate"], as_index=False)["intensity_ppm"].mean()
    unique_pep_avg = unique_pep_biorep.groupby(["peptide", "species_label", "condition"], as_index=False)["intensity_ppm"].mean()
    unique_pep_wide = unique_pep_avg.pivot_table(index=["peptide", "species_label"], columns="condition", values="intensity_ppm", fill_value=0.0).reset_index()

    # Calculate unique peptide sum (anchor) per species to reconstitute ancestor clados
    # A. Collapse technical replicates first
    leaves_run_sum = df_unique.groupby(["species_label", "run", "condition", "biological_replicate"], as_index=False)["intensity_ppm"].sum()
    # B. Collapse biological replicates
    leaves_avg_unique = leaves_run_sum.groupby(["species_label", "condition"], as_index=False)["intensity_ppm"].mean()
    leaves_unique_wide = leaves_avg_unique.pivot_table(index="species_label", columns="condition", values="intensity_ppm", fill_value=0.0).reset_index()

    # ==========================================================================
    # METHOD 2: RIGID LCA LOG2 FC (Traditional Median of directly assigned peptides)
    # ==========================================================================
    print("Processing Method 2: Rigid LCA...")
    # Collapse replicates at peptide level
    lca_pep_biorep = df_clean.groupby(["peptide", "tree_node", "condition", "biological_replicate"], as_index=False)["intensity_ppm"].mean()
    lca_pep_avg = lca_pep_biorep.groupby(["peptide", "tree_node", "condition"], as_index=False)["intensity_ppm"].mean()
    lca_pep_wide = lca_pep_avg.pivot_table(index=["peptide", "tree_node"], columns="condition", values="intensity_ppm", fill_value=0.0).reset_index()

    # ==========================================================================
    # METHOD 3: MIXTURE MODEL LOG2 FC (Proportional Split)
    # ==========================================================================
    print("Processing Method 3: Mixture Model...")
    runs = sorted(df_clean["run"].unique())
    mixture_rows = []

    for run in runs:
        df_run = df_clean[df_clean["run"] == run].copy()
        cond = df_run["condition"].iloc[0]
        
        df_run_unique = df_run[df_run["num_descendants"] == 1].copy()
        df_run_unique["species_label"] = df_run_unique["tree_node"].map(lambda n: node_to_desc[n][0])
        anchors = df_run_unique.groupby("species_label")["intensity_ppm"].sum().to_dict()
        for label in species_df["Label"]:
            if label not in anchors:
                anchors[label] = 0.0
                
        for _, row in df_run.iterrows():
            node = row["tree_node"]
            intensity = row["intensity_ppm"]
            peptide = row["peptide"]
            
            descendants = node_to_desc.get(node, [])
            if not descendants:
                continue
                
            if len(descendants) == 1:
                mixture_rows.append({
                    "run": run,
                    "condition": cond,
                    "species_label": descendants[0],
                    "peptide": peptide,
                    "split_intensity": intensity
                })
            else:
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
    df_mixture = df_mixture.merge(rep_map[["run", "biological_replicate", "technical_replicate"]], on="run", how="left")
    
    # Collapse technical replicates for Mixture Model at Peptide level
    mix_pep_biorep = df_mixture.groupby(["peptide", "species_label", "condition", "run", "biological_replicate"], as_index=False)["split_intensity"].sum()
    mix_pep_biorep_mean = mix_pep_biorep.groupby(["peptide", "species_label", "condition", "biological_replicate"], as_index=False)["split_intensity"].mean()
    # Collapse biological replicates
    mix_pep_avg = mix_pep_biorep_mean.groupby(["peptide", "species_label", "condition"], as_index=False)["split_intensity"].mean()
    mix_pep_wide = mix_pep_avg.pivot_table(index=["peptide", "species_label"], columns="condition", values="split_intensity", fill_value=0.0).reset_index()

    # Reconstitute leaf averages in Mixture Model to evaluate ancestor clados
    mix_run_sum = df_mixture.groupby(["species_label", "run", "condition", "biological_replicate"], as_index=False)["split_intensity"].sum()
    mix_avg_sum = mix_run_sum.groupby(["species_label", "condition"], as_index=False)["split_intensity"].mean()
    mix_sum_wide = mix_avg_sum.pivot_table(index="species_label", columns="condition", values="split_intensity", fill_value=0.0).reset_index()

    # ==========================================================================
    # BENCHMARK COMPARISON ASSEMBLY & EVALUATION FOR LOG2 FOLD CHANGE
    # ==========================================================================
    print("Calculating observed vs. expected log2 fold changes...")
    all_fc_rows = []

    for node, descendants in node_to_desc.items():
        if not descendants:
            continue
            
        num_species = len(descendants)
        node_type = "Leaf" if num_species == 1 else "Internal Node"
        
        sub_known = species_df[species_df["Label"].isin(descendants)]
        
        for num_cond, den_cond, num_s, den_s in comparisons:
            comp_label = f"{num_cond}_vs_{den_cond}"
            
            # --- EXPECTED LOG2 FC (Recipe) ---
            exp_num_ug = sub_known[f"Input Protein Amount in {num_s} (ug) normalized"].sum()
            exp_den_ug = sub_known[f"Input Protein Amount in {den_s} (ug) normalized"].sum()
            
            # CRITICAL BIOLOGICAL FILTER (STRICT NO-IMPUTATION):
            # If the species/node expected amount is 0 in either condition, there is NO fold change.
            # We strictly exclude it from the evaluation dataset to prevent mathematical noise or infinity errors.
            if exp_num_ug == 0 or exp_den_ug == 0:
                continue
                
            expected_log2fc = np.log2(exp_num_ug / exp_den_ug)
            
            # --- METHOD 1: UNIQUE ONLY OBSERVED LOG2 FC ---
            if node_type == "Leaf":
                node_peps_u = unique_pep_wide[unique_pep_wide["species_label"] == node]
                p_num = node_peps_u[num_cond]
                p_den = node_peps_u[den_cond]
                valid = (p_num > 0) & (p_den > 0)
                if valid.any():
                    unique_fc = np.median(np.log2(p_num[valid] / p_den[valid]))
                    is_excl_u = False
                else:
                    unique_fc = np.nan
                    is_excl_u = True
            else:
                # For clados in unique only: log2 of ratio of reconstructed sum of descendant leaves
                sub_leaves = leaves_unique_wide[leaves_unique_wide["species_label"].isin(descendants)]
                sum_num = sub_leaves[num_cond].sum()
                sum_den = sub_leaves[den_cond].sum()
                if sum_num > 0 and sum_den > 0:
                    unique_fc = np.log2(sum_num / sum_den)
                    is_excl_u = False
                else:
                    unique_fc = np.nan
                    is_excl_u = True
                    
            # --- METHOD 2: RIGID LCA OBSERVED LOG2 FC ---
            # Traditional median of directly assigned peptides
            node_peps_l = lca_pep_wide[lca_pep_wide["tree_node"] == node]
            p_num = node_peps_l[num_cond]
            p_den = node_peps_l[den_cond]
            valid = (p_num > 0) & (p_den > 0)
            if valid.any():
                lca_fc = np.median(np.log2(p_num[valid] / p_den[valid]))
                is_excl_l = False
            else:
                lca_fc = np.nan
                is_excl_l = True
                
            # --- METHOD 3: MIXTURE MODEL OBSERVED LOG2 FC ---
            if node_type == "Leaf":
                node_peps_m = mix_pep_wide[mix_pep_wide["species_label"] == node]
                p_num = node_peps_m[num_cond]
                p_den = node_peps_m[den_cond]
                valid = (p_num > 0) & (p_den > 0)
                if valid.any():
                    mix_fc = np.median(np.log2(p_num[valid] / p_den[valid]))
                    is_excl_m = False
                else:
                    mix_fc = np.nan
                    is_excl_m = True
            else:
                # For clados in mixture model: log2 of ratio of reconstructed sum of descendant leaves
                sub_leaves = mix_sum_wide[mix_sum_wide["species_label"].isin(descendants)]
                sum_num = sub_leaves[num_cond].sum()
                sum_den = sub_leaves[den_cond].sum()
                if sum_num > 0 and sum_den > 0:
                    mix_fc = np.log2(sum_num / sum_den)
                    is_excl_m = False
                else:
                    mix_fc = np.nan
                    is_excl_m = True
                    
            all_fc_rows.append({
                "Node": node,
                "Node_Type": node_type,
                "Num_Species": num_species,
                "Comparison": comp_label,
                "Expected_Log2FC": expected_log2fc,
                "Unique_Log2FC": unique_fc,
                "Unique_Excluded": is_excl_u,
                "LCA_Log2FC": lca_fc,
                "LCA_Excluded": is_excl_l,
                "Mix_Log2FC": mix_fc,
                "Mix_Excluded": is_excl_m
            })

    eval_df = pd.DataFrame(all_fc_rows)
    eval_df.to_csv(outdir / "step2_mixture_model_foldchange_results.csv", index=False)
    print(f"[SUCCESS] Tabular log2 fold change dataset saved to: {outdir}/step2_mixture_model_foldchange_results.csv")

    # ==========================================================================
    # GENERATE 3X3 DIAGNOSTIC PLOT MATRIX (LOG2 FC COMPARISON)
    # ==========================================================================
    print("Generating 3x3 Log2 Fold Change Benchmark comparison plot matrix...")
    fig, axs = plt.subplots(3, 3, figsize=(22, 21))
    
    # Set a global normalization scale so colors are perfectly comparable across all rows/methods
    global_max_depth = eval_df["Num_Species"].max()
    norm = plt.Normalize(1, global_max_depth)

    methods = [
        ("Unique_Log2FC", "Unique_Excluded", "Unique Peptides Only (Discard Shared)", "rainbow"),
        ("LCA_Log2FC", "LCA_Excluded", "Rigid LCA Assignment (Paso 2)", "rainbow"),
        ("Mix_Log2FC", "Mix_Excluded", "Mixture Model (Proportional Split)", "rainbow")
    ]
    
    conditions_labels = {
        "Uneven_vs_Constant": "Uneven vs Constant",
        "Protein_vs_Constant": "Protein vs Constant",
        "Uneven_vs_Protein": "Uneven vs Protein"
    }

    comp_keys = ["Uneven_vs_Constant", "Protein_vs_Constant", "Uneven_vs_Protein"]

    for row_idx, comp_key in enumerate(comp_keys):
        comp_df = eval_df[eval_df["Comparison"] == comp_key].copy()
        
        for col_idx, (col_name, excl_name, method_title, cmap_base) in enumerate(methods):
            ax = axs[row_idx, col_idx]
            
            # Filter non-excluded values
            plot_df = comp_df[~comp_df[excl_name] & comp_df["Expected_Log2FC"].notna() & comp_df[col_name].notna()].copy()
            
            if plot_df.empty:
                ax.set_title(f"No Data\n{conditions_labels[comp_key]} - {method_title}")
                continue
                
            x_vals = plot_df["Expected_Log2FC"]
            y_vals = plot_df[col_name]
            
            # Calculate linear correlations
            pearson_r = x_vals.corr(y_vals, method="pearson")
            spearman_R = x_vals.corr(y_vals, method="spearman")
            
            # Separate Leaf and Internal
            leaves = plot_df[plot_df["Node_Type"] == "Leaf"]
            internal = plot_df[plot_df["Node_Type"] == "Internal Node"]
            
            # Plot Leaves (circles)
            sc_leaves = ax.scatter(
                leaves["Expected_Log2FC"], leaves[col_name],
                c=leaves["Num_Species"], cmap=cmap_base, norm=norm,
                marker="o", edgecolors="black", s=130, alpha=0.9, zorder=4, label="Leaf (Species)"
            )
            
            # Plot Internal Nodes (triangles)
            sc_internal = ax.scatter(
                internal["Expected_Log2FC"], internal[col_name],
                c=internal["Num_Species"], cmap=cmap_base, norm=norm,
                marker="^", edgecolors="black", s=150, alpha=0.9, zorder=4, label="Internal (Ancestor)"
            )
            
            # Dynamic identity line limits tailored to the cleaned data (usually spans -7 to 7 now!)
            min_lim = -7.0
            max_lim = 7.0
            ax.plot([min_lim, max_lim], [min_lim, max_lim], "r--", alpha=0.7, linewidth=1.5, label="1:1 Identity", zorder=2)
            
            # Formatting
            ax.set_xlim(min_lim, max_lim)
            ax.set_ylim(min_lim, max_lim)
            ax.grid(True, linestyle="--", alpha=0.4, zorder=1)
            
            # Labels
            ax.set_xlabel("Expected Log2 Fold Change", fontsize=10, fontweight="bold")
            ax.set_ylabel(f"Measured {method_title.split(' ')[0]} Log2 FC", fontsize=10, fontweight="bold")
            
            # Add titles with correlations
            ax.set_title(f"{conditions_labels[comp_key]} | {method_title}\nPearson r: {pearson_r:.4f} | Spearman R: {spearman_R:.4f}", 
                         fontsize=11, fontweight="bold", pad=8)
            
            # Local colorbar
            cbar = plt.colorbar(sc_leaves, ax=ax, pad=0.02)
            cbar.set_label("Descendant Species Count", fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor="white")
            
            # Annotating labels cleanly
            for _, row in plot_df.iterrows():
                name = str(row["Node"])
                is_leaf = row["Node_Type"] == "Leaf"
                
                # Annotate leaf nodes (circles) using black bold text
                if is_leaf:
                    ax.text(
                        row["Expected_Log2FC"] + 0.15,
                        row[col_name] - 0.10,
                        name,
                        fontsize=7.5,
                        alpha=0.9,
                        color="black",
                        fontweight="bold"
                    )
                else:
                    # Only annotate ancestors (triangles) if their expected magnitude of change is notable
                    if abs(row["Expected_Log2FC"]) > 0.5:
                        short_ancestor = name
                        if short_ancestor.startswith("LCA_Node_of_"):
                            short_ancestor = short_ancestor.replace("LCA_Node_of_", "LCA_")
                        
                        if len(short_ancestor) > 12:
                            short_ancestor = short_ancestor[:10] + ".."
                            
                        ax.text(
                            row["Expected_Log2FC"] + 0.15,
                            row[col_name] - 0.10,
                            short_ancestor,
                            fontsize=7,
                            alpha=0.8,
                            color="darkred",
                            fontweight="semibold",
                            style="italic"
                        )

    plt.suptitle("METAPROTEOMICS LOG2 FOLD CHANGE VALIDATION BENCHMARK MATRIX (V2_Rainbow)\n"
                 "Strict Exclusion of Zero-Recipe Species with Uniform Rainbow Depth Scale",
                 fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    plot_path = outdir / "step2_mixture_model_foldchange_matrix_v2_rainbow.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] 3x3 Log2 FC Rainbow Benchmark plot matrix saved to: {plot_path}")

    # ==========================================================================
    # CONSOLE DIAGNOSTIC BENCHMARK SUMMARY
    # ==========================================================================
    print("\n" + "="*95)
    print("           METAPROTEOMICS LOG2 FOLD CHANGE METHODOLOGY COMPARATIVE BENCHMARK (V2)")
    print("="*95)
    for comp_key in comp_keys:
        sub = eval_df[eval_df["Comparison"] == comp_key].copy()
        
        print(f"\nComparison: {conditions_labels[comp_key]}")
        print("-" * 65)
        
        for col_name, excl_name, title, _ in methods:
            valid = sub[~sub[excl_name] & sub["Expected_Log2FC"].notna() & sub[col_name].notna()]
            
            pears = valid["Expected_Log2FC"].corr(valid[col_name], method="pearson") if len(valid) > 2 else np.nan
            spear = valid["Expected_Log2FC"].corr(valid[col_name], method="spearman") if len(valid) > 2 else np.nan
            
            # Leaves only
            leaves = valid[valid["Node_Type"] == "Leaf"]
            r_leaves = leaves["Expected_Log2FC"].corr(leaves[col_name], method="pearson") if len(leaves) > 2 else np.nan
            
            print(f"  -> {title.split(' ')[0]:<14} | Total Nodes Pearson r: {pears:.4f} | Spearman R: {spear:.4f} | Leaves Pearson r: {r_leaves:.4f}")
    print("="*95 + "\n")

if __name__ == "__main__":
    main()
