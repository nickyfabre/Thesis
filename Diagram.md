### **Taxonomic Abundance Pipeline (Mass Estimation)**

```
[ fileSDRF_Run_1_2.sdrf_openms_design_triqler_in.tsv ] (QuantMS/Nextflow Output)
[ suplementarydata1_speciesamounts.csv ]              (Physical Microgram Recipe)
                           |
                           v
         === SCRIPT 1: TAXONOMIC RESOLUTION ===
      (step2_lca_peptide_aggregation_block1_v2.py)
                           |
                           v
         [ step2_peptides_lca_assigned.tsv ]         (Peptides mapped with short labels)
                           |
                           v
       === SCRIPT 2: PEPTIDE DETECTABILITY & PPM ===
       (step2_lca_peptide_aggregation_block2.py)
                           |
                           v
         [ step2_peptides_corrected_ppm.tsv ]         (Corrected & normalized PPMs)
                           |
       +-------------------+-------------------+
       | (Experimental Design)                 | (Ground Truth Recipe)
       | [fileSDRF_Run_1_2.sdrf.tsv]           | [suplementarydata1_speciesamounts.csv]
       v                                       v
     =============================================
         === SCRIPT 3: MIXTURE MODEL BENCHMARK ===
           (step2_mixture_model_pipeline_v3.py)
     =============================================
                           |
       +-------------------+-------------------+
       |                                       |
       v                                       v
[step2_mixture_model_benchmark_matrix.png]   [step2_mixture_model_benchmark_results.csv]
(Definitive 3x3 diagnostic matrix plot)     (Tabular correlations and metrics)
```

**Log2 Fold Change Pipeline**

```
[ fileSDRF_Run_1_2.sdrf_openms_design_triqler_in.tsv ] (QuantMS/Nextflow Output)
[ suplementarydata1_speciesamounts.csv ]              (Physical Microgram Recipe)
                           |
                           v
         === SCRIPT 1: TAXONOMIC RESOLUTION ===
      (step2_lca_peptide_aggregation_block1_v2.py)
                           |
                           v
         [ step2_peptides_lca_assigned.tsv ]         (Peptides mapped with short labels)
                           |
                           v
       === SCRIPT 2: PEPTIDE DETECTABILITY & PPM ===
       (step2_lca_peptide_aggregation_block2.py)
                           |
                           v
         [ step2_peptides_corrected_ppm.tsv ]         (Corrected & normalized PPMs)
                           |
       +-------------------+-------------------+
       | (Experimental Design)                 | (Ground Truth Recipe)
       | [fileSDRF_Run_1_2.sdrf.tsv]           | [suplementarydata1_speciesamounts.csv]
       v                                       v
     =============================================
         === SCRIPT 3: BENCHMARK FOLD CHANGE ===
     (step2_mixture_model_foldchange_pipeline.py)
     =============================================
                           |
       +-------------------+-------------------+
       |                                       |
       v                                       v
[step2_mixture_model_foldchange_matrix.png] [step2_mixture_model_foldchange_results.csv]
(3x3 matrix of Log2 FC magnitudes of change)  (Table of expected vs. measured Log2 FCs)
```
