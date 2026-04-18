# Injection Outage

`non_voltage_regulating_injection_outage_dx` computes one-step AC state updates for a batch of injection outages around the N-0 operating point.

The Jacobian is kept fixed. Each outage only contributes an active and reactive power mismatch at the affected bus, and the batched update is evaluated with the base inverse Jacobian.

$$
\Delta X = - M J^{-T}
$$

where `J` is the base-case Jacobian and `M` is the batch of outage mismatch vectors in Jacobian ordering.

## Batch API

The function expects a 2D outage array with shape `(n_contingencies, n_outages)`.

- Each row is one contingency.
- A row may contain multiple outage indices.
- Negative entries are ignored and can be used as padding.
- The return value has shape `(n_contingencies, n_eq)`.

## Scope

This update is intended for non-voltage-regulating injections such as loads and other fixed-power injections.

## Generator Behavior

The function can also be applied to voltage-regulating generator outages, but this is only an approximation. A voltage-regulating generator outage should normally trigger a PV to PQ bus-type change. The current implementation does not model that switch, so it does not capture the resulting voltage drop at the regulated bus.

## HVDC

- HVDC converter outages should be outaged in pairs. DC+ models the HVDC line as a pair of generator + load injections, one at each terminal.
