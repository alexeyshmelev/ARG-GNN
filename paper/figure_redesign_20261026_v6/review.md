# Compact TreeLoom figures, revision 6

Revision 5 passed a detailed reconstruction check but missed the communication goal. It exposed too much routine implementation detail and gave preprocessing the same visual weight as the model. This revision changes that hierarchy.

The overview follows observed trees and reference data through the TreeLoom encoder, same-leaf sequence assembly, a bidirectional GRU and ancestry prediction. The second figure expands the two tree rounds and only the two context mechanisms. Ordinary activations, full feature recipes, normalization constants and rank details remain in the methods and existing revision history.

The new reader receives only two frozen images and assesses the main computation and reading order. It is not asked to recover every implementation operation. Source agreement is checked independently. The complete reports remain available, including any remaining ambiguities.

The source model, saved toy, result tables and stopped experiments are unchanged. All authoring and rendering run in the dedicated server repository.

The first fresh reader recovered the computation but found group selection ambiguous. Three labels were clarified for a second independent reader. That reader correctly described the complete conceptual prediction path and both context mechanisms. Its remaining questions about interval measurements, age cuts and residual summation are addressed in the captions. Both full reports and the explicit scope of acceptance are retained.
