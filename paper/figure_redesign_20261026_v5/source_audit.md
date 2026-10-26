# Implementation contract for the figures

The diagrams describe the frozen stable-leaf TreeLoom architecture, with four output classes for the European example. Source pins are in `source_hashes.json`. No architecture or experimental result is changed by this revision.

| Stage | Verified contract |
|---|---|
| Node input | Normalized age, root bit and sampled-leaf bit. Linear 3 to 32 followed by tanh. |
| Site input | Eight donor log frequencies selected by the called allele, eight donor availability bits, panel log frequency, entropy and called bit. Linear 19 to 32, GELU, Linear 32 to 32, tanh. |
| Site pooling | Within a leaf and interval, mean and maximum over called sites. Append normalized called count, empty bit and normalized interval length, giving 67 entries. Linear 67 to 32 and tanh. |
| Leaf metadata | Four reference ancestry entries, unknown bit and normalized sampling age. Concatenate with pooled site encoding, then Linear 38 to 32 and tanh. Add only at sampled-leaf rows. |
| Branch gates | Nine observed entries: child age, parent age, duration, descendant fraction, exact support, nearest support, discrepancy, normalized split count, empty flag. Separate upward and downward Linear 9 to 32 with sigmoid. The toy has eight real branches. The implementation stores a dummy root slot that carries no branch message. |
| Upward pass | Transform and gate each child state, divide by its parent's child count, then sum. Concatenate aggregate with node base, apply Linear 64 to 32 and tanh, add the base and LayerNorm. Leaves have zero child aggregate. |
| Downward pass | Root outside state is zero. For a child, sum parent outside state, parent base and parent aggregate, subtract that child's saved normalized message. Transform, gate, transform with tanh, then LayerNorm. There is no extra downward residual. |
| Structure | Compare descendant sets of proper clades across neighboring intervals. Gain, loss and persistence can overlap. Separate factored maps with rank eight transform the current upward state. Scatter their sum at current clade representatives. There is no transport of a neighboring ancestor's hidden state. |
| Memory | Gather first-round upward plus outside states at the age-cut representatives. Project 32 to 8, copy to member leaves, and follow each leaf in both genomic directions. Decay is exp(-interval length / learned scale), with scale 100 + 99900 sigmoid(theta) bp. Each direction starts at its current forcing. |
| Memory return | Mean of the two directions per leaf, then mean over leaves in each current group. Subtract current group forcing, project 8 to 32, scatter only at representative rows. |
| Second round | Add both corrections to the original base, then repeat a complete upward and downward pass with the same weights and gates. |
| Readout | Select leaf rows from both final states. Concatenate 32 plus 32, Linear 64 to 32 and tanh. Stack the same leaf across intervals. Append three geometry entries, giving trees by 35. |
| Sequence classifier | One bidirectional GRU, width 32 in each direction. Concatenate its 64 features with the original local 32, then Linear 96 to 32, GELU, Linear 32 to 4, softmax. |

Normalization uses log(1+x)/log(10001) for ages and branch duration, log(1+x)/log(100001) for counts, and log(1+x)/log(2000001) for interval lengths and center distances.

The saved toy contains intervals [0,3000), [3000,7000), and [7000,10000). The observed sites are at 1500, 5000 and 8500 bp. All five sampled leaves have age zero. Tree two uses nodes q,r1,r2,r3,r4,a,b,c,root, corresponding to original IDs 0,1,2,3,4,6,8,9,12. Its internal ages are 175,212.5,400,675 generations. At the cut of 350 generations, tree-one groups are {q,r1}, {r2}, {r3,r4}. Trees two and three have {q}, {r1,r2}, {r3,r4}.

Observed toy values come from this saved tree sequence and illustrative reference labels. Hidden matrix entries are schematic. They are not recorded model activations or accuracy results. No simulation or training was performed.
