 original idea of integrating abundances over a species tree. I dont find the appropriate project definitions, but I think this text can serve as a better description of what we discussed previously.
*Background
In metaproteomics we measure peptides from a community of organisms. Each peptide is mapped back to the proteins, and thus the organisms, it could originate from. You have already run quantMS, so you have peptide abundances across a set of conditions.
The complication is that a single peptide sequence is often shared between several organisms and cannot be assigned to one species unambiguously. Conventional metaproteomics handles this at the protein level. We are not interested in proteins; we want the abundances of the organisms.
The idea
We place the abundance on a phylogenetic (species) tree rather than on individual species. A peptide specific to one species contributes abundance to that leaf. A peptide that cannot be resolved between two or more species contributes instead to the node where those species join, i.e. their last common ancestor, or more precisely to the corresponding internal branch of the tree.
Every peptide is thereby allocated to the most specific point in the tree its evidence supports. Specific peptides resolve down to leaves; shared peptides stay higher up. No abundance is lost and none is forced onto a single species.
The task
Starting from your peptide abundances per condition:
For each peptide, determine the set of organisms whose proteomes contain it, and find the node (leaf or internal) that is the last common ancestor of that set.
Aggregate peptide abundances onto tree nodes, so each node/branch carries a quantity per condition.
Compare conditions: test for differential abundance at the level of tree nodes rather than proteins. A change at an internal node indicates a shift in a clade that cannot be pinned to one species; a change at a leaf indicates a species-specific shift.
Deliverables
A method that takes the quantMS peptide quantities plus a species tree and returns per-condition abundances at every node, together with a differential test between conditions. An evaluation on the available dataset, and a comparison against naive strategies (discarding shared peptides, or splitting their abundance equally between species).
