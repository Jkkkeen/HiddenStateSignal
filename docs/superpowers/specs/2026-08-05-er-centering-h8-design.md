# ER Centering and H8 Token Entropy Design

## Scope

Update the Experiment 2E discovery specification and metric draft only. The change covers H5, V8, and a new H8 family. It does not start training, hidden-state extraction, or H200 work.

## H5: Centered State-Trajectory ER Dynamics

For each final-layer chunk-state prefix `Z[:k]`, subtract the prefix row mean before SVD:

\[
Z^c_{1:k}=Z_{1:k}-\mathbf 1\bar z_k^\top,
\qquad
\bar z_k=\frac{1}{k}\sum_{i=1}^k z_i.
\]

The centered prefix ER sequence is the primary sequence used to compute ERV and ERA. The previous uncentered ER, ERV, and ERA remain sensitivity outputs and do not enter the primary registry.

## V8: Vertical Effective Rank

For the layer-state matrix, subtract the row mean across layers before SVD. Centered `layer_state_ER` is the reported layer-state quantity.

For the layer-update matrix, keep uncentered `layer_update_ER` as primary because adjacent-layer differencing already removes the common state offset. Also save `layer_update_ER_centered` as a sensitivity output. This prevents identical same-direction layer updates from collapsing to a zero matrix in the primary metric.

## H8: Token-Level Entropy Dynamics

H8 is computed once from token hidden states and is not duplicated across the `mean` and `last` chunk representations.

The final transformer layer is the preregistered horizontal primary view. All-layer profiles are exploratory outputs only.

Save three clearly named token-level quantities:

1. `token_time_diff_coordinate_entropy`: coordinate-centered energy entropy of `h[t] - h[t-1]`. Its stage median is the H8 primary representative.
2. `token_state_coordinate_entropy`: coordinate-centered energy entropy of the undifferenced token state. This matches the old Experiment04 `token_raw_entropy` operator but uses an unambiguous name.
3. `token_state_raw_energy_entropy`: uncentered energy entropy of the undifferenced token state. This retains persistent high-activation-coordinate effects.

For local trajectory plots, aggregate token scalars within the frozen 128-token windows at stride 32 using the median. For checkpoint-stage analysis, use the corresponding stage medians. Save mean and P90 as diagnostics; they do not replace the frozen primary representative.

The primary registry becomes H1-H8 plus V1-V8, for 16 primary representatives. Only `token_time_diff_coordinate_entropy_median` enters the H8 primary slot; the two state-entropy quantities are paired controls.

## Validation

Add or update numerical toy checks for:

- centered ER invariance to adding a common vector to every row;
- centered H5 prefix ER/ERV/ERA output;
- V8 centered layer-state ER and raw/centered layer-update ER outputs;
- token time-difference entropy on a constant trajectory;
- distinction between coordinate-centered and raw state entropy;
- H8 representation deduplication and minimum token coverage.
