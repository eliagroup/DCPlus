## Basic idea
The DC+ project started with the idea to add voltage magnitudes to the established DC calculation, hence the name DC+. DC+ is now equivalent to one Newton-Raphson AC iteration, but is designed around low rank updates. This enables a large speed advantage when compared to traditional solver. The DC+ solver linearizes around the full AC converged N-0 load-flow case. Each action that is performed, e.g. a line outage, injection outage, bus split or any combination of these actions, is than equivalent to one step AC towards this action. The core idea is to materialize the inverse of the N-0 Jacobian once and utilize the gpu's batch matrix operation capabilities for a high speed action calculation on this particular grid situation.

# DC+ strengths
- Topological searches: bus splits + security analysis
- Injection changes, e.g. time series calculation
- Small tap changes on regulation transformer

# DC+ weakness
- Extreme tap changes on regulation transformer (noticeable by exploding numbers far beyond the per unit system)
- A change where the Newton-Raphson AC iteration is large (e.g. when an outage needs more than 10 iterations to converge, here DC often performs even worse)
- Any kind of controller

A further description can be found here: "[Voltage-sensitive distribution factors for contingency analysis and topology optimization](https://arxiv.org/pdf/2509.19976)"

## Numpy reference implementation

The numpy cpu reference is for testing and readability only. This may change if there is a need for a fast cpu version as well. The jax and numpy are designed to mirror each other, but may take different inputs (e.g. numpy takes the inverse jacobian, the jax the transposed inverse jacobian)

## Jax implementation

The core of the project is the Jax implementation which is designed for a high throughput on gpu. 

## Branch outage

In the NumPy reference, the branch outage update is assembled in [`branch_outage_update_inverse`][dc_plus.numpy.lodf.branch_outage_update_inverse] using [`full_rank_delta_inv_jacobian`][dc_plus.numpy.lodf.full_rank_delta_inv_jacobian].
For each outaged branch, [`_prepare_low_rank_factors_from_admittance`][dc_plus.numpy.low_rank_helper._prepare_low_rank_factors_from_admittance] builds a small Jacobian delta block `D` and the corresponding affected state indices `idx_list`.
The full inverse-Jacobian correction is then computed with a Woodbury-style expression:

$$
\Delta J^{-1} = C\,(I + D\,B)^{-1}\,D\,R
$$

where `R = J^{-1}[idx_list, :]`, `C = J^{-1}[:, idx_list]`, and `B = J^{-1}[idx_list, idx_list]`.
The updated inverse is applied branch-by-branch as:

$$
J^{-1}_{\text{new}} = J^{-1}_{\text{old}} - \Delta J^{-1}
$$

The monitored-element path is implemented in [`branch_outage_monitored_bus_dx`][dc_plus.numpy.lodf.branch_outage_monitored_bus_dx].
Instead of materializing the full `\Delta J^{-1}`, it solves only the small correction system for the outaged branch:

$$
(I + D\,B)\,x = D\,(J^{-1}\,\Delta f)_{idx}
$$

and then evaluates only the monitored bus state components (`\Delta\theta`, `\Delta u`).
The function maps `monitor_bus` to its angle and magnitude equation indices, and returns:

$$
\Delta x_{\text{monitor}} = - (J^{-1}\,\Delta f)_{\text{monitor}} + J^{-1}_{\text{monitor},idx}\,x
$$

This gives a direct monitored-bus update without building the full matrix update.

## Bus Split

The core idea of the current bus split implementation is to break down the bus split into many branch outages on bu A and branch connections at a new bus B.

In the NumPy reference, the bus split update is implemented in [`compute_bsdf_update`][dc_plus.numpy.bsdf_full_rank.compute_bsdf_update].
Current scope is PQ split with the reassignment of branches, shunts and non-voltage-regulating injections.

The function first collects the affected equation indices (`idx_list`) by taking:
- the split bus A components,
- the new bus B components,
- and all angle/magnitude components of branch endpoints for each reassigned branch, both before and after replacing bus A by bus B.

It then builds a local dense delta block `delta_block` on `idx_list`.
For each reassigned branch, it computes two 4x4 admittance-based Jacobian contribution blocks:
- `delta_old` for the original connection,
- `delta_new` for the post-split connection.

These are accumulated into `delta_block` as `+delta_old - delta_new`, projected onto valid equation positions.
After that, the new bus state equations are injected by subtracting 1.0 on the diagonal entries for bus B angle and magnitude components.

Finally, this local delta is lifted to a full inverse-Jacobian update through the same full-rank Woodbury routine ([`_apply_full_rank_update`][dc_plus.numpy.bsdf_full_rank._apply_full_rank_update] -> [`full_rank_delta_inv_jacobian`][dc_plus.numpy.lodf.full_rank_delta_inv_jacobian]):

$$
J^{-1}_{\text{new}} = J^{-1}_{\text{base}} - \Delta J^{-1}(\text{from } delta\_block, idx\_list)
$$

So the bus split matrix is formed by aggregating per-branch old/new local deltas on a targeted index set, then applying one full-rank inverse update.
