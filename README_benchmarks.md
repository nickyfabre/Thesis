# Regenerating Benchmarks

To regenerate the benchmarks for the thesis:

1. Ensure dependencies are installed: `pip install -r requirements.txt` (and install `matplotlib scipy statsmodels plotly seaborn` if plotting features are used).
2. For testing/sandbox environments, you can provide a mock supplementary data file and let the script generate mock quantitative data:

\`\`\`bash
cat << 'EOF2' > suplementarydata1_speciesamounts.csv
Label,Species,Input Protein Amount in C (ug) normalized,Input Protein Amount in P (ug) normalized,Input Protein Amount in U (ug) normalized
CRH,Clostridium sp.,10.0,20.0,5.0
ATN,Actinomyces sp.,10.0,5.0,20.0
K12,Escherichia coli,10.0,10.0,10.0
LT2,Salmonella enterica,10.0,10.0,10.0
EOF2
\`\`\`

3. Run the complete pipeline:
\`\`\`bash
python step2_lca_peptide_aggregation_block1_v2.py --known suplementarydata1_speciesamounts.csv
python step2_lca_peptide_aggregation_block2.py
python step2_mixture_model_foldchange_pipeline.py --known suplementarydata1_speciesamounts.csv
python step2_mixture_model_pipeline_v3.py --known suplementarydata1_speciesamounts.csv
python step3_linear_mixed_model_benchmark.py
python step2_mixture_model_additional_metrics_v2.py
\`\`\`

The output benchmark files and charts will be located in the `step2_out/` and `step3_lmm_out/` directories.
