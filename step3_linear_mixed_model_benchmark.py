import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

def analyze_lmm_abundance(csv_path: Path, outdir: Path):
    """
    Fits a Linear Mixed Model (LMM) for Abundance Absolute Error:
    Error_log = |log10(Measured_Pct) - log10(Expected_Pct)|
    
    Fixed Effects: Method (Mixture vs Rigid, Mixture vs Unique) + Condition
    Random Effect: (1 | Node) -> Accounts for repeated measures within each taxonomic node.
    """
    print(f"\n=================================================================================")
    print(f"   LINEAR MIXED MODEL (LMM): ABUNDANCE ERROR | Grouping: (1 | Node)")
    print(f"=================================================================================")
    
    if not csv_path.exists():
        print(f"[WARNING] Abundance CSV not found at '{csv_path}'. Skipping abundance LMM.")
        return None

    df = pd.read_csv(csv_path)
    
    req_cols = ["Node", "Condition", "Expected_Pct", "Unique_Pct", "LCA_Pct", "Mix_Pct"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing and "Rigid_Pct" in df.columns:
        df["LCA_Pct"] = df["Rigid_Pct"]
        missing = [c for c in req_cols if c not in df.columns]
            
    if missing:
        print(f"[ERROR] Missing required columns in abundance CSV: {missing}")
        return None

    # Filter positive expected and measured values to avoid log10(0)
    df_valid = df[(df["Expected_Pct"] > 0) & (df["Unique_Pct"] > 0) & (df["LCA_Pct"] > 0) & (df["Mix_Pct"] > 0)].copy()
    
    if df_valid.empty:
        print("[WARNING] No valid positive rows found for log-error calculation.")
        return None

    # Calculate absolute log10 error per node & method
    df_valid["Err_Unique"] = np.abs(np.log10(df_valid["Unique_Pct"]) - np.log10(df_valid["Expected_Pct"]))
    df_valid["Err_Rigid"] = np.abs(np.log10(df_valid["LCA_Pct"]) - np.log10(df_valid["Expected_Pct"]))
    df_valid["Err_Mixture"] = np.abs(np.log10(df_valid["Mix_Pct"]) - np.log10(df_valid["Expected_Pct"]))

    # Reshape to long format for Linear Mixed Model
    long_rows = []
    for _, row in df_valid.iterrows():
        node = str(row["Node"])
        cond = str(row["Condition"])
        node_type = str(row.get("Node_Type", "Unknown"))
        
        long_rows.append({"Node": node, "Condition": cond, "Node_Type": node_type, "Method": "Mixture", "Log_Error": row["Err_Mixture"]})
        long_rows.append({"Node": node, "Condition": cond, "Node_Type": node_type, "Method": "Rigid", "Log_Error": row["Err_Rigid"]})
        long_rows.append({"Node": node, "Condition": cond, "Node_Type": node_type, "Method": "Unique", "Log_Error": row["Err_Unique"]})

    df_long = pd.DataFrame(long_rows)
    results_list = []

    # 1. Overall Model across all conditions
    print("\n--- Fitting Overall Abundance LMM (Log_Error ~ Method + Condition, Random Effect: 1|Node) ---")
    try:
        model_overall = smf.mixedlm(
            "Log_Error ~ C(Method, Treatment(reference='Mixture')) + C(Condition)",
            df_long,
            groups=df_long["Node"]
        )
        fit_overall = model_overall.fit(reml=True)
        print(fit_overall.summary())
        
        params = fit_overall.params
        bse = fit_overall.bse
        pvalues = fit_overall.pvalues
        conf_int = fit_overall.conf_int()

        for term in params.index:
            if "Method" in term:
                method_name = term.split("[T.")[1].replace("]", "") if "[T." in term else term
                results_list.append({
                    "Dataset": "Abundance",
                    "Subset": "Overall",
                    "Comparison": f"{method_name} vs Mixture",
                    "Diff_Estimate": params[term],
                    "Std_Err": bse[term],
                    "z_stat": params[term] / bse[term] if bse[term] > 0 else np.nan,
                    "p_value": pvalues[term],
                    "CI_Lower": conf_int.loc[term, 0],
                    "CI_Upper": conf_int.loc[term, 1]
                })
    except Exception as e:
        print(f"[WARNING] Overall Abundance LMM fit failed: {e}")

    # 2. Per-Condition LMMs
    conditions = df_long["Condition"].unique()
    for cond in conditions:
        df_sub = df_long[df_long["Condition"] == cond].copy()
        print(f"\n--- Fitting LMM for Condition: {cond} (Log_Error ~ Method, Random Effect: 1|Node) ---")
        try:
            model_sub = smf.mixedlm(
                "Log_Error ~ C(Method, Treatment(reference='Mixture'))",
                df_sub,
                groups=df_sub["Node"]
            )
            fit_sub = model_sub.fit(reml=True)
            print(fit_sub.summary())
            
            params = fit_sub.params
            bse = fit_sub.bse
            pvalues = fit_sub.pvalues
            conf_int = fit_sub.conf_int()

            for term in params.index:
                if "Method" in term:
                    method_name = term.split("[T.")[1].replace("]", "") if "[T." in term else term
                    results_list.append({
                        "Dataset": "Abundance",
                        "Subset": cond,
                        "Comparison": f"{method_name} vs Mixture",
                        "Diff_Estimate": params[term],
                        "Std_Err": bse[term],
                        "z_stat": params[term] / bse[term] if bse[term] > 0 else np.nan,
                        "p_value": pvalues[term],
                        "CI_Lower": conf_int.loc[term, 0],
                        "CI_Upper": conf_int.loc[term, 1]
                    })
        except Exception as e:
            print(f"[WARNING] LMM fit for {cond} failed ({e}). Attempting OLS with cluster-robust errors.")
            try:
                ols_fit = smf.ols("Log_Error ~ C(Method, Treatment(reference='Mixture'))", data=df_sub).fit(cov_type='cluster', cov_kwds={'groups': df_sub['Node']})
                for term in ols_fit.params.index:
                    if "Method" in term:
                        method_name = term.split("[T.")[1].replace("]", "") if "[T." in term else term
                        results_list.append({
                            "Dataset": "Abundance",
                            "Subset": cond,
                            "Comparison": f"{method_name} vs Mixture",
                            "Diff_Estimate": ols_fit.params[term],
                            "Std_Err": ols_fit.bse[term],
                            "z_stat": ols_fit.tvalues[term],
                            "p_value": ols_fit.pvalues[term],
                            "CI_Lower": ols_fit.conf_int().loc[term, 0],
                            "CI_Upper": ols_fit.conf_int().loc[term, 1]
                        })
            except Exception as e_ols:
                print(f"[ERROR] Robust OLS fallback also failed: {e_ols}")

    res_df = pd.DataFrame(results_list)
    out_file = outdir / "lmm_abundance_statistical_results.csv"
    res_df.to_csv(out_file, index=False)
    print(f"\n[SUCCESS] Abundance LMM results saved to: {out_file}")
    return res_df


def analyze_lmm_foldchange(csv_path: Path, outdir: Path):
    """
    Fits a Linear Mixed Model (LMM) for Log2 Fold Change Absolute Error:
    FC_Error = |Measured_Log2FC - Expected_Log2FC|
    
    Fixed Effects: Method (Mixture vs Rigid, Mixture vs Unique) + Comparison
    Random Effect: (1 | Node) -> Accounts for repeated measures within each taxonomic node.
    """
    print(f"\n=================================================================================")
    print(f"   LINEAR MIXED MODEL (LMM): LOG2 FOLD CHANGE ERROR | Grouping: (1 | Node)")
    print(f"=================================================================================")
    
    if not csv_path.exists():
        print(f"[WARNING] Fold Change CSV not found at '{csv_path}'. Skipping fold change LMM.")
        return None

    df = pd.read_csv(csv_path)
    
    req_cols = ["Node", "Comparison", "Expected_Log2FC", "Unique_Log2FC", "LCA_Log2FC", "Mix_Log2FC"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns in fold change CSV: {missing}")
        return None

    # Filter out excluded or NaN rows
    excl_u = df.get("Unique_Excluded", pd.Series(False, index=df.index))
    excl_l = df.get("LCA_Excluded", pd.Series(False, index=df.index))
    excl_m = df.get("Mix_Excluded", pd.Series(False, index=df.index))
    
    valid_mask = ~excl_u & ~excl_l & ~excl_m & df["Expected_Log2FC"].notna() & df["Unique_Log2FC"].notna() & df["LCA_Log2FC"].notna() & df["Mix_Log2FC"].notna()
    df_valid = df[valid_mask].copy()

    if df_valid.empty:
        print("[WARNING] No valid unexcluded rows found for Log2 FC error calculation.")
        return None

    # Calculate absolute error |Measured_FC - Expected_FC|
    df_valid["Err_Unique"] = np.abs(df_valid["Unique_Log2FC"] - df_valid["Expected_Log2FC"])
    df_valid["Err_Rigid"] = np.abs(df_valid["LCA_Log2FC"] - df_valid["Expected_Log2FC"])
    df_valid["Err_Mixture"] = np.abs(df_valid["Mix_Log2FC"] - df_valid["Expected_Log2FC"])

    # Reshape to long format
    long_rows = []
    for _, row in df_valid.iterrows():
        node = str(row["Node"])
        comp = str(row["Comparison"])
        node_type = str(row.get("Node_Type", "Unknown"))
        
        long_rows.append({"Node": node, "Comparison": comp, "Node_Type": node_type, "Method": "Mixture", "FC_Error": row["Err_Mixture"]})
        long_rows.append({"Node": node, "Comparison": comp, "Node_Type": node_type, "Method": "Rigid", "FC_Error": row["Err_Rigid"]})
        long_rows.append({"Node": node, "Comparison": comp, "Node_Type": node_type, "Method": "Unique", "FC_Error": row["Err_Unique"]})

    df_long = pd.DataFrame(long_rows)
    results_list = []

    # 1. Overall Model across all FC comparisons
    print("\n--- Fitting Overall Fold Change LMM (FC_Error ~ Method + Comparison, Random Effect: 1|Node) ---")
    try:
        model_overall = smf.mixedlm(
            "FC_Error ~ C(Method, Treatment(reference='Mixture')) + C(Comparison)",
            df_long,
            groups=df_long["Node"]
        )
        fit_overall = model_overall.fit(reml=True)
        print(fit_overall.summary())
        
        params = fit_overall.params
        bse = fit_overall.bse
        pvalues = fit_overall.pvalues
        conf_int = fit_overall.conf_int()

        for term in params.index:
            if "Method" in term:
                method_name = term.split("[T.")[1].replace("]", "") if "[T." in term else term
                results_list.append({
                    "Dataset": "FoldChange",
                    "Subset": "Overall",
                    "Comparison": f"{method_name} vs Mixture",
                    "Diff_Estimate": params[term],
                    "Std_Err": bse[term],
                    "z_stat": params[term] / bse[term] if bse[term] > 0 else np.nan,
                    "p_value": pvalues[term],
                    "CI_Lower": conf_int.loc[term, 0],
                    "CI_Upper": conf_int.loc[term, 1]
                })
    except Exception as e:
        print(f"[WARNING] Overall Log2 FC LMM fit failed: {e}")

    # 2. Per-Comparison LMMs
    comparisons = df_long["Comparison"].unique()
    for comp in comparisons:
        df_sub = df_long[df_long["Comparison"] == comp].copy()
        print(f"\n--- Fitting LMM for Comparison: {comp} (FC_Error ~ Method, Random Effect: 1|Node) ---")
        try:
            model_sub = smf.mixedlm(
                "FC_Error ~ C(Method, Treatment(reference='Mixture'))",
                df_sub,
                groups=df_sub["Node"]
            )
            fit_sub = model_sub.fit(reml=True)
            print(fit_sub.summary())
            
            params = fit_sub.params
            bse = fit_sub.bse
            pvalues = fit_sub.pvalues
            conf_int = fit_sub.conf_int()

            for term in params.index:
                if "Method" in term:
                    method_name = term.split("[T.")[1].replace("]", "") if "[T." in term else term
                    results_list.append({
                        "Dataset": "FoldChange",
                        "Subset": comp,
                        "Comparison": f"{method_name} vs Mixture",
                        "Diff_Estimate": params[term],
                        "Std_Err": bse[term],
                        "z_stat": params[term] / bse[term] if bse[term] > 0 else np.nan,
                        "p_value": pvalues[term],
                        "CI_Lower": conf_int.loc[term, 0],
                        "CI_Upper": conf_int.loc[term, 1]
                    })
        except Exception as e:
            print(f"[WARNING] LMM fit for {comp} failed ({e}). Attempting OLS with cluster-robust errors.")
            try:
                ols_fit = smf.ols("FC_Error ~ C(Method, Treatment(reference='Mixture'))", data=df_sub).fit(cov_type='cluster', cov_kwds={'groups': df_sub['Node']})
                for term in ols_fit.params.index:
                    if "Method" in term:
                        method_name = term.split("[T.")[1].replace("]", "") if "[T." in term else term
                        results_list.append({
                            "Dataset": "FoldChange",
                            "Subset": comp,
                            "Comparison": f"{method_name} vs Mixture",
                            "Diff_Estimate": ols_fit.params[term],
                            "Std_Err": ols_fit.bse[term],
                            "z_stat": ols_fit.tvalues[term],
                            "p_value": ols_fit.pvalues[term],
                            "CI_Lower": ols_fit.conf_int().loc[term, 0],
                            "CI_Upper": ols_fit.conf_int().loc[term, 1]
                        })
            except Exception as e_ols:
                print(f"[ERROR] Robust OLS fallback also failed: {e_ols}")

    res_df = pd.DataFrame(results_list)
    out_file = outdir / "lmm_foldchange_statistical_results.csv"
    res_df.to_csv(out_file, index=False)
    print(f"\n[SUCCESS] Fold Change LMM results saved to: {out_file}")
    return res_df


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Linear Mixed Model (LMM) Benchmark Evaluation (1|Node)"
    )
    parser.add_argument(
        "--abundance_csv",
        default="step2_out/step2_mixture_model_benchmark_results.csv",
        help="Path to the abundance benchmark CSV."
    )
    parser.add_argument(
        "--foldchange_csv",
        default="step2_out/step2_mixture_model_foldchange_results.csv",
        help="Path to the log2 fold change benchmark CSV."
    )
    parser.add_argument(
        "--outdir",
        default="./step3_lmm_out",
        help="Output directory for statistical LMM reports."
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    abd_csv = Path(args.abundance_csv)
    fc_csv = Path(args.foldchange_csv)

    res_abd = analyze_lmm_abundance(abd_csv, outdir)
    res_fc = analyze_lmm_foldchange(fc_csv, outdir)

    print("\n" + "="*85)
    print("         SUMMARY OF LINEAR MIXED MODEL (LMM) STATISTICAL CONTRASTS")
    print("="*85)
    
    all_res = []
    if res_abd is not None and not res_abd.empty:
        all_res.append(res_abd)
    if res_fc is not None and not res_fc.empty:
        all_res.append(res_fc)

    if all_res:
        final_df = pd.concat(all_res, ignore_index=True)
        # Apply Benjamini-Hochberg FDR correction across p-values
        if len(final_df) > 0 and "p_value" in final_df.columns:
            pvals = final_df["p_value"].fillna(1.0).values
            _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
            final_df["q_value_FDR"] = qvals

        summary_file = outdir / "lmm_consolidated_statistical_summary.csv"
        final_df.to_csv(summary_file, index=False)
        
        print(f"\n{final_df[['Dataset', 'Subset', 'Comparison', 'Diff_Estimate', 'Std_Err', 'p_value', 'q_value_FDR', 'CI_Lower', 'CI_Upper']].to_string(index=False)}")
        print(f"\n[SUCCESS] Consolidated LMM summary saved to: {summary_file}")
    else:
        print("[WARNING] No statistical results produced. Check CSV inputs.")

if __name__ == "__main__":
    main()
