import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Try importing ete3 for taxonomy. If not present, we will fallback to Simulation Mode
try:
    from ete3 import NCBITaxa
    HAS_TAXONOMY_LIB = True
except ImportError:
    HAS_TAXONOMY_LIB = False

pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

CONTAMINANT_PREFIX = "CRAP"
DECOY_PREFIXES = {"DECOY", "decoy", "REV", "rev", "Decoy"}

def load_species_metadata(known_path: Path) -> pd.DataFrame:
    """
    Loads species metadata from csv or tsv to map Label (prefix) to official Scientific Name.
    """
    if not known_path.exists():
        sys.exit(f"FATAL: Species metadata file not found at '{known_path}'")
    
    sep = "\t" if str(known_path).lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(known_path, sep=sep)
    
    required = {"Label", "Species"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"FATAL: Species metadata missing columns {missing}; found {list(df.columns)}")
        
    df['Label'] = df['Label'].astype(str).str.strip()
    df['Species'] = df['Species'].astype(str).str.strip()
    return df

def setup_taxonomy_mapping(species_df: pd.DataFrame):
    """
    Translates scientific names to NCBI TaxIDs and builds the local ete3 tree topology.
    """
    if not HAS_TAXONOMY_LIB:
        print("\n[WARNING] 'ete3' or 'NCBITaxa' is not installed in this environment.")
        print("          Proceeding in SIMULATION MODE. TaxIDs will be mocked.")
        prefix_to_taxid = {row['Label']: 1000 + idx for idx, row in species_df.iterrows()}
        tree_leaf_names = set(str(tid) for tid in prefix_to_taxid.values())
        return prefix_to_taxid, tree_leaf_names, None

    print("\nLoading NCBI Taxonomy database...")
    ncbi = NCBITaxa()
    
    species_list = species_df['Species'].dropna().unique().tolist()
    print(f"Translating {len(species_list)} unique species to official NCBI TaxIDs...")
    name2taxid = ncbi.get_name_translator(species_list)
    
    prefix_to_taxid = {}
    for _, row in species_df.iterrows():
        prefix = row['Label']
        species_name = row['Species']
        if species_name in name2taxid:
            prefix_to_taxid[prefix] = name2taxid[species_name][0]
        else:
            # Fallback if name is not in the translator
            print(f"  [WARN] Species '{species_name}' not found in NCBI translator. Mocking TaxID.")
            prefix_to_taxid[prefix] = 999999
            
    valid_taxids = [tid for tid in prefix_to_taxid.values() if tid != 999999]
    print(f"Building phylogenetic tree topology with {len(valid_taxids)} TaxIDs...")
    tree = ncbi.get_topology(valid_taxids)
    tree_leaf_names = set(tree.get_leaf_names())
    
    return prefix_to_taxid, tree_leaf_names, tree

def resolve_lca_for_proteins(protein_string: str, prefix_to_taxid: dict, tree_leaf_names: set, tree, species_df: pd.DataFrame) -> str:
    """
    Resolves the Lowest Common Ancestor (LCA) for a given protein string (handling DECOYs natively).
    Returns the three-letter code (Label) for unique species leaves, and taxonomic scientific names for ancestors.
    """
    prot_str = str(protein_string).strip()
    
    # Decoy Handling: Keep them intact to preserve Triqler's FDR calculations
    if "DECOY_" in prot_str:
        return prot_str
        
    prots = prot_str.split(';')
    taxids = set()
    
    # Extract strain prefixes (e.g. "CRH", "ATN") and get their mapped TaxIDs
    for p in prots:
        p = p.strip()
        if "_" in p:
            prefix = p.split('_')[0].strip()
            if prefix in prefix_to_taxid:
                taxids.add(prefix_to_taxid[prefix])
                
    if not taxids:
        return "Unknown"
        
    # Reverse map to convert TaxIDs back to 3-letter Labels
    taxid_to_prefix = {v: k for k, v in prefix_to_taxid.items()}

    # Simulation mode LCA resolution
    if not HAS_TAXONOMY_LIB or tree is None:
        labels = sorted([taxid_to_prefix.get(tid, "Unmapped") for tid in taxids])
        if len(labels) == 1:
            return labels[0]
        else:
            return f"LCA_Node_of_{'_'.join(labels)}"

    # Production mode NCBI Taxonomy LCA resolution
    ncbi = NCBITaxa()
    taxid_strings = [str(tid) for tid in taxids]
    valid_nodes = [t for t in taxid_strings if t in tree_leaf_names]
    
    if not valid_nodes:
        return "Unknown"
        
    # Unique peptide: Resolve directly to leaf label (three-letter code)
    if len(valid_nodes) == 1:
        single_taxid = int(valid_nodes[0])
        if single_taxid in taxid_to_prefix:
            return taxid_to_prefix[single_taxid]
        translated = ncbi.get_taxid_translator([single_taxid])
        return translated.get(single_taxid, "Unknown")
        
    # Shared peptide: Resolve using tree get_common_ancestor
    try:
        ancestor = tree.get_common_ancestor(*valid_nodes)
        ancestor_taxid = int(ancestor.name)
        # Keep scientific name of taxid for ancestors (internal nodes)
        translated = ncbi.get_taxid_translator([ancestor_taxid])
        return translated.get(ancestor_taxid, "Unknown")
    except Exception:
        return "Error_Calculating_LCA"

def create_mock_quantms_table(out_path: Path):
    """
    Generates a small mock QuantMS table so the script can be tested without raw files.
    """
    print(f"  Creating a mock QuantMS peptide table at '{out_path}' for local sandbox testing...")
    rows = []
    # Mix of unique and shared peptides across various runs
    peptides = [
        ("PEPTIDE_UNIQUE_CRH", "CRH_XP_001696798.1"),
        ("PEPTIDE_UNIQUE_ATN", "ATN_protein_1"),
        ("PEPTIDE_SHARED_CRH_ATN", "CRH_XP_001696798.1;ATN_protein_1"),
        ("PEPTIDE_SHARED_ALL_ENT", "K12_protein_1;LT2_protein_1"), # Escherichia (K12) and Salmonella (LT2) -> Enterobacteriaceae
        ("PEPTIDE_DECOY_TEST", "DECOY_CRH_XP_001696798.1")
    ]
    
    # Simulate 24 runs (conditions: Constant, Protein, Uneven)
    conditions = ["Constant"] * 8 + ["Protein"] * 8 + ["Uneven"] * 8
    for run in range(1, 25):
        cond = conditions[run-1]
        for pep, prot in peptides:
            rows.append({
                "run": run,
                "condition": cond,
                "proteins": prot,
                "peptide": pep,
                "intensity": np.random.uniform(1000, 50000),
                "charge": 2,
                "searchScore": 0.99
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)

def main():
    ap = argparse.ArgumentParser(description="Step 2: LCA Peptide Allocation - Block 1 (V2 - 3-Letter Code Labels)")
    ap.add_argument("--quant_table", default="fileSDRF_Run_1_2.sdrf_openms_design_triqler_in.tsv")
    ap.add_argument("--known", default="/workspace/knowledge/suplementarydata1_speciesamounts.csv")
    ap.add_argument("--outdir", default="./step2_out")
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    known_path = Path(args.known)
    quant_path = Path(args.quant_table)
    
    # 1. Load Species Metadata
    print("Loading species metadata...")
    species_df = load_species_metadata(known_path)
    
    # 2. Setup Taxonomy Mapping
    prefix_to_taxid, tree_leaf_names, tree = setup_taxonomy_mapping(species_df)
    
    # 3. Handle QuantMS Table (Auto-create mock if missing for sandbox environment)
    if not quant_path.exists():
        print(f"\n[INFO] Input file '{quant_path}' not found.")
        create_mock_quantms_table(quant_path)
        
    print(f"\nProcessing peptide table: {quant_path}...")
    triqler_df = pd.read_csv(quant_path, sep="\t")
    
    # 4. Apply LCA Resolution
    print("Resolving Lowest Common Ancestor (LCA) for all peptides...")
    triqler_df['tree_node'] = triqler_df['proteins'].apply(
        lambda x: resolve_lca_for_proteins(x, prefix_to_taxid, tree_leaf_names, tree, species_df)
    )
    
    # 5. Calculate Biological Metrics & Data Recovery
    print("\n" + "="*50)
    print("        BLOCK 1: TAXONOMY RESOLUTION QUALITY CONTROL (V2)")
    print("="*50)
    total_peptides = len(triqler_df)
    
    biological_nodes = triqler_df[~triqler_df['tree_node'].astype(str).str.contains("DECOY_")]
    valid_bio_df = biological_nodes[~biological_nodes['tree_node'].isin(['Unknown', 'Error_Calculating_LCA'])].copy()
    
    # Determine unique vs shared
    def is_shared(protein_string):
        if "DECOY_" in str(protein_string):
            return False
        prefixes = set()
        for p in str(protein_string).split(';'):
            p = p.strip()
            if "_" in p:
                prefixes.add(p.split('_')[0].strip())
        return len(prefixes) > 1
        
    valid_bio_df['is_shared'] = valid_bio_df['proteins'].apply(is_shared)
    shared_counts = valid_bio_df['is_shared'].value_counts(normalize=True) * 100
    unique_pct = shared_counts.get(False, 0)
    shared_pct = shared_counts.get(True, 0)
    
    print(f"Total Peptides Processed: {total_peptides}")
    print(f"Unique Peptides (Leaves): {unique_pct:.2f}%")
    print(f"Shared Peptides (Internal Nodes): {shared_pct:.2f}%")
    print(f"Saved from discard by LCA: {shared_pct:.2f}%")
    print("\nTop 5 Most Populated Nodes:")
    print(valid_bio_df['tree_node'].value_counts().head(5))
    print("="*50)
    
    # 6. Save Intermediate LCA Assigned Peptide Table
    output_path = outdir / "step2_peptides_lca_assigned.tsv"
    triqler_df.to_csv(output_path, sep="\t", index=False)
    print(f"\n[SUCCESS] Block 1 finished. LCA mapped peptides saved to: {output_path}")

if __name__ == "__main__":
    main()
