# Independent blind read: v9

Only the rendered schematic and its 2x image were inspected. No manuscript, caption, source, previous review, or model output was read.

## What the image communicates

- **a → b:** a time-oriented ancestral graph spans a 0–10 kb region split into three intervals. Three corresponding local trees, T1–T3, retain the magenta query q1 while reference leaf order changes. The diamonds labelled R are visible but their meaning is not defined here.
- **Observed inputs:** four reference samples carry four-class one-hot ancestry vectors: r1 = Ana, r2 = CHG, r4 = EHG, r3 = WHG. q1 has an unknown ancestry label. Each tree also has five sample-aligned binary values, labelled g_(t,q): 0,1,1,1,1 for T1 and 0,0,0,1,1 for T2/T3 in each displayed leaf order. A [τ,0,0] vector points to an internal node of T2. The image does not define those binary values or the vector components.
- **Visible operation order:** an input bracket leads to MLP → b_v → GNNθ, whose icon explicitly shows Up then Down passes. Its output splits to Δ_str and Sample memory. Sample memory connects each named sample across T1–T3 in both directions, then feeds Linear. A plus combines Δ_str, Linear output, and a b_v skip connection. The sum enters another Up → Down GNNθ. This produces a sample-by-tree embedding grid labelled e_tq. That grid feeds a one-layer BiGRU; interval lengths l_t enter the BiGRU separately. A bypass from e_tq joins the BiGRU output at the ∥ symbol, conventionally concatenation. MLP then produces four-colour ancestry probabilities p_tq. Argmax yields q1 ancestry along the three intervals; this example assigns WHG throughout.
- **Identity:** r1/r2/r3/r4/q1 labels, fixed rows in Sample memory and the embedding grid, and continuous magenta highlighting preserve sample identity despite changing tree leaf order. The figure shows same-sample links across trees rather than linking samples by leaf position. Panel a does not label every leaf, so its complete sample mapping cannot be audited from the image alone.

## Mandatory versus caption-level concerns

**Drawing-level clarity concern:** the only explicit b→c bracket lies immediately beneath the ancestry-vector row. The g_(t,q) rows and [τ,0,0] callout have no explicit route to the MLP or GNN. Consequently I cannot establish from the drawing whether they are actual encoder inputs or explanatory annotations. To make the input vectors understandable from the figure alone, visibly group or route all intended observed inputs. This is a clarity issue, not evidence that the method is incorrect.

**No confirmed rendering/topology blocker:** no literal text/node/arrow overlap or clipped label was found at either size. All three trees keep q1 and r3 as siblings, and the unlabelled GNN thumbnails are compatible with that ladder topology. The long sum→second-GNN return path is traceable, although visually long.

**Caption-level requirements:** define R, g_(t,q), τ and the components of the internal-node vector; state the numeric encoding used for q1's missing ancestry; explain Δ_str and the actual Sample memory operation; confirm ∥ means concatenation and what is concatenated; explain what l_t does inside the BiGRU; clarify whether repeated θ means shared GNN parameters and whether p_tq/e_tq refer only to q1 or all displayed sample rows. The image does not support inferring attention, a particular memory update, a loss, training protocol, or a numerical ancestry inference mechanism beyond the visible layers and argmax.

The no-title constraint is met: only a/b/c and ordinary layer/data labels are present.
