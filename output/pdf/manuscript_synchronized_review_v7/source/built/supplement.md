Supplemental Material: Reliability-aware active learning for FFLO phase spaces
S1. PHYSICAL MODEL AND FREE-ENERGY FUNCTIONAL
We study a one-dimensional altermagnetic superconductor in the Nambu basis
Ψk = (ck+q/2,↑,ck+q/2,↓,c†
−k+q/2,↑,c†
−k+q/2,↓)T
with
HBdG(k) =
(
h(k +q/2)−µ ∆ q
∆∗
q −σyhT (−k +q/2)σy +µ
)
. (S1)
where h(k) = t cosk + (0,λy sink,λz sink +JA cosk)·σ. The production code evaluates the
normal-referenced occupation-weighted spectral objective
Fimpl(∆,q ) = 1
2Nk
∑
k,n
En(k; ∆,q )f
[En(k; ∆,q )
kBT
]
+ ∆ 2
U
−1
2Nk
∑
k,n
En(k; 0,q )f
[En(k; 0,q )
kBT
]
. (S2)
with f(z) = 1/(ez + 1) implemented as a stable sigmoid. The sum includes all four BdG eigenvalue
branches, the factor1/2 accounts for Nambu doubling, and the final term subtracts the∆ = 0
spectrum at the sameq. Thek sum is divided by the configured full grid sizeNk = 800. Temperature
enters only through the occupations: the production implementation contains no separately added
entropy or log-partition term. We therefore useFimpl for the implemented objective rather than
replacing it by a textbook grand-potential expression. All frozen minima and labels are those
obtained from this objective, with energies in units oft.
S2. EXACT BDG ORACLE
Algorithm S1: Thermodynamic and topology evaluation.Input: requested coordinate, numeri-
cal configuration, and enabled oracle mode.
Output: observables, phase label, numerical status, and applicable topology fields.
TABLE S1. Physical and numerical parameters.
Quantity Value Role
t 1 energy unit
λy,λz 0.6, 0.6 spin–orbit couplings
µ 0.55 2D chemical potential
U 1.6 pairing interaction
ωD 10 momentum mask cutoff
Nk 800 Brillouin-zone grid
coarse ∆ grid [0, 0.6], 400 points thermodynamic scan
initial q grid [−1, 0.5], 400 points momentum scan
∆ ϵ,qϵ 10−3, 10−2 phase assignment
2
1. For each coordinate, construct the BdG spectra on the configured(∆,q ) grid and evaluate
the unchanged objective in Sec. S1.
2. Locate candidate minima. In the adaptive-box path, test enclosure; while allowed and
necessary, expand theq window directionally (at most three expansions). Refine selected
global, nearly degenerate, and edge-risk basins using the configured grids (at least 800q and
600 ∆ points in this path).
3. Apply the near-zero-∆ comparison. Assign thermodynamic identity separately from training
eligibility and unresolved/rerun flags; the incremental oracle follows its recorded refinement
configuration.
4. If the topology calculation is enabled and thermodynamic/numerical applicability holds, scan
the full-zone gap. Only a resolved gapped FFLO state receives the Pfaffian index; otherwise
return the relevant applicability or unresolved status.
5. Return records with numerical metadata. Coordinate reconciliation occurs at dataset merge;
it is not a physical classification rule.
S3. RELIABILITY-STATE HIERARCHY
Accepted/reliable points have an enclosed minimum and a resolved normal-state comparison.
Boundary-delicate points lie within the adopted finite energy or order-parameter resolution. A
q-window-unresolved point retains low-energy structure at the scanned edge after allowed expansion;
a near-zero-∆ -unresolved point lacks a resolved condensation-energy sign. Rerun-required records
remain outside the clean physical training view. Topology-inapplicable includes all non-FFLO or
numerically unreliable states, while gap-unresolved states do not receive aZ2 label. Phase identity,
training eligibility, rerun status, and topology applicability are separate fields.
S4. COMPLETE ACTIVE-LEARNING LOOP
Algorithm S2: Stage-resolved active-learning loop.Input: stage configuration and initial or
resumed exact dataset (Sec. S6.1).
Output: updated records, trained models, selected-call tables, and diagnostic history.
1. For a cold start, evaluate the stage-specific initial design; for a continuation, load the recorded
resume dataset without drawing another seed.
2. At each iteration, construct the eligible training view and phase-stratified training/validation
split. Reinitialize and train both ensembles; standardize inputs and regression targets using
training statistics.
3. Generate the stage-specific candidate domain, predict observables/probabilities, and score
the candidates using Secs. S6.2 and S6.3.
4. Select a batch using Algorithm S3; write the point shards. Evaluate the BdG oracle and
preserve failed or unresolved returns as explicit records.
5. Merge the completed shards, reconcile duplicate coordinate keys, and update the eligible
learning view without overwriting reliability flags.
3
Exact BdG evaluation: checks and returned information
1  Minimize Compare Fimpl(Δ, q) minima and normal reference
2  Test numerical resolution Check momentum-window enclosure
and near-zero pairing amplitude
3  Separate status Reliable label: Normal / uniform SC / FFLO
Keep delicate and rerun flags separately
4  Test topology For applicable FFLO states, evaluate bulk gap;
only resolved gapped states receive ν = 0 or 1
The surrogate schedules calls; it does not assign physical labels.
FIG. S1. Exact-oracle reliability-state flow. Numerical status is separate from phase identity, and topology is
restricted to reliable gapped FFLO states.
6. Recompute the recorded monitoring diagnostics. Continue unless the configured operational
stop or budget is reached; save the dataset, configuration, selections, and diagnostics before
returning.
S5. SURROGATE ARCHITECTURE AND TRAINING
The implementation uses separate PyTorch ensembles for regression and phase classification,
not a Gaussian process or tree model. For Stages I–III, each of five members is an MLP with two
64-unit hidden layers and ReLU activations. Inputs are standardized by the training mean and
standard deviation. Regression targets(∆,q,η,I+
c ,I−
c ) are also standardized; the regression loss is
mean-squared error. The classifier uses cross entropy on Normal, uniform-SC, and FFLO labels.
Adam uses learning rate10−3 and weight decay10−5, batch size 512, and 240 epochs for both
tasks. Members use seeds42 + 1000 +i and 42 + 2000 +i for regression and classification. Models
are retrained from scratch at each scheduling step; no early-stopping rule is implemented in these
model classes. Regression uncertainty is the member standard deviation; phase probabilities are
averaged over members and converted to entropy and margin scores.
S6. ACQUISITION FUNCTION AND MICROBATCH SELECTION
Let Ucls = 0.4H + 0.6 exp[−m/0.2], whereH is normalized phase entropy andm is the largest-
probability margin. Let Ureg = (U∆ +Uq)/2, BNS = exp[−|ˆ∆ −∆ ϵ|/0.04] 4PNPSC, BUF =
4
TABLE S2. Two-dimensional surrogate and training hyperparameters (Stages I–III).
Item Value
ensemble size 5 per task
hidden layers 2 ×64, ReLU
validation fraction 0.15, phase stratified
optimizer Adam
learning rate / weight decay 10−3/10−5
epochs / batch size 240 / 512
regression loss MSE
classification loss cross entropy
uncertainty member dispersion / averaged probabilities
Regression ensemble and classifier: stage-specific inputs
Stages I–III Inputs: kBT/t, JA/t
Two hidden layers: 64 + 64 ReLU units
Stage IV Inputs: kBT/t, JA/t, μ/t
Two hidden layers: 96 + 96 ReLU units
Both tasks Five independent members per task; retrain each round
Regression: MSE; classification: cross entropy
Outputs Regression: Δ, q, η, I +
c , I −
c  and dispersion
Classification: phase probabilities
Acquisition is a scheduler only; exact BdG results supply the labels.
FIG. S2. Stage-specific surrogate architecture. Stages I–III have two inputs and 64-unit hidden layers; Stage
IV has three inputs and 96-unit hidden layers. The regression and classification ensembles supply scheduling
scores, not physical labels.
PSC exp[−||ˆq|−qϵ|/0.04], andG = (G∆ +Gq)/2. The Stage-I/II full profile is
Aphase =Ucls + 0.6Ureg +BNS + 0.9BUF + 0.5G,
A0 =Aphase + 0.4Eq +wextEext. (S3)
where Eq flags predicted q-edge risk and Eext combines extrapolation with uncertainty. wext
decreases piecewise from 0.15 to 0.08 and 0.03 at iterations 10 and 30. High-confidence interiors
receive a soft penalty; the active pool is set by the larger of a scheduled quantile and0.7p95(A0).
Observation repulsion uses length 0.02 and floor 0.5. During greedy stochastic construction, batch
repulsion uses length 0.03 and floor 0.01, yieldingAselect =ApoolRobsRbatch and sampling probability
5
TABLE S3. Production acquisition components.
Component Definition Normalization or weight Role
phase entropy/margin Ucls 0.4/0.6 mixture classification ambiguity
regression spread Ureg min–max normalized; 0.6 phase-facing uncertainty
Normal–SC BNS competition gated; weight 1 order-parameter boundary
uniform–FFLO BUF SC gated; weight 0.9 finite- q boundary
gradients G grid derivative; weight 0.5 rapid variation
numerical risk Eq weight 0.4 q-window risk
exploration Eext annealed 0.15/0.08/0.03 domain coverage
repulsion RobsRbatch length/floor rules avoid redundant calls
proportional toAγ
select, withγ= 1.5, 2.5, 4.0 across the same schedule. Existing exact coordinates
are excluded before selection.
Algorithm S3: Stage-specific repelled selection. Input: candidate coordinates, component
scores, existing observations, and stage configuration.
Output: ordered selected batch, probabilities, and selection metadata.
1. In 2D, apply exact-coordinate and cooldown masks, interior penalties, and the active-pool
threshold. Initialize batch repulsion to one and include observation repulsion.
2. In 3D, use the proposal pool and channel quotas; use the weighted combined score for
exhausted-channel backfill. Apply the recorded duplicate handling.
3. While the requested batch is incomplete and positive finite weights exist: form the appropriate
score times repulsion, raise to the stage-specific power, normalize, and sample one index.
4. Record its probability, remove the selected index, and multiply the remaining batch factors by
K(d;ℓ,r). Return early if no eligible weight remains; do not invent extra oracle evaluations.
S6.1. Stage-specific settings and auxiliary responses
The run configurations distinguish initialization from continuation. Stage I uses a 512-location
random-grid design, seed 42, on the241×321 temperature–exchange grid. Stage II continues
that dataset rather than drawing another independent seed. Stage III starts a separate 512-point
scrambled Sobol design, also with seed 42, on the same bounded two-coordinate domain. Its final
saved configuration is a resume at iteration 17; the initialization field describes the original run,
not 512 additional observations at resume. Stage IV starts 1,024 scrambled Sobol points with seed
20260623 in the three-coordinate domain. The nominal subsequent batch size is 256 in all four
stages; actual selected and retained counts need not equal this request.
All stages train separate regression and classification ensembles. The archived Stage-IV adapter
overrides the input dimension, hidden width, epoch count, ensemble size, and seed; its optimization
batch size, Adam settings, and validation fraction come from the archived constructor, not from
the current development defaults. The target vector remains(∆ opt,q opt,η,I+
c ,I−
c ). No additional
learned spectral network is used in the Stage-III/IV acquisition implementation examined here:
predicted ∆,q are passed to analytic Pfaffian expressions to construct scheduling proxies. They do
not replace exact spectral labels.
6
Production acquisition: two-dimensional full profile
1  Score candidates A0 = Aphase + 0.4Eq + wextEext
Phase ambiguity, numerical risk, exploration
2  Restrict the pool Apool = A0Pinterior
Exact-coordinate masks and scheduled quantiles
3  Select a microbatch p(x) ∝ [ApoolRobsRbatch]γ
Update batch repulsion after every selected point
4  Return to BdG Evaluate selected coordinates; retain numerical status
Merge observations and retrain
Stage III uses topology scores; Stage IV uses channel quotas (see text).
FIG. S3. Acquisition decomposition and repelled microbatch selection.
TABLE S4. Stage-specific training settings extracted from saved run configurations and the corresponding
archived model constructors. Optimization batches are distinct from 256-point oracle batches.
Setting Stages I–II Stage III Stage IV
Input coordinates T,JA T,JA T,JA,µ
Hidden layers (ReLU) 64, 64 64 , 64 96 , 96
Members per task 5 5 5
Epochs (regression/classification) 240/240 240 /240 180 /180
Optimization batch size 512 512 512
Learning rate 10−3 10−3 10−3
Weight decay 10−5 10−5 10−5
Validation fraction 0.15 0.15 0.15
Base seed 42 42 20260623
Here I+
c ≥0 and I−
c ≤0 are the signed extrema located on the positive and negative current
branches outward from the optimized momentum. The implementation records
η=|I+
c|−|I−
c|
|I+c|+|I−c|, (S4)
with zero returned if both currents vanish (and−1 or +1 if only the positive or negative branch
vanishes). These are auxiliary response targets, not independently validated diode-response claims.
In the saved Stage-I/IIfull profile, response scores are exported but excluded from the main
acquisition. Stage III’stopo_trivial profile and Stage IV’s explicit three-channel selector likewise
use the ∆,q uncertainty channels rather thanη,I±
c to schedule calls. The presence of an optional
7
response-weighted cleanup branch in the archived code is not evidence that it was used in these
endpoint configurations.
S6.2. Definitions of the two-dimensional acquisition components
The following definitions apply to the saved two-dimensional implementation. WriteN [z] =
(z−zmin)/(zmax−zmin) over finite values on the candidate array; constant arrays and nonfinite
entries map to zero, and the result is clipped to[0, 1]. Each uncertainty channel is normalized
separately. Grid gradients use first-order edge differences and coordinate spacings, not pixel indices:
G = 1
2N
[√
(∂T/t ˆ∆) 2 + (∂JA/t ˆ∆) 2
]
+ 1
2PSCN
[√
(∂T/t|ˆq|)2 + (∂JA/t|ˆq|)2
]
. (S5)
In this equation the temperature coordinate denoteskBT/t. Let [z]+ = max(z, 0), dq = min(|ˆq +
1|,|0.5−ˆq|), andτ=kBT/t, j =JA/t. Then
Eq =PSC max{exp[−dq/0.02],N ([j−1.2]+)}, (S6)
Eext = max{N([τ−0.5]+),N ([j−1.2]+)}max{Ucls,U reg}. (S7)
The thresholds above are acquisition parameters, not physical phase criteria. The interior indicator
requiresPmax > 0.98 and each ofUNS,UUF,G,Eq, andEext below 0.05, withUNS = 4PNPSC and
UUF = 4PuniformPFFLO. Its multiplicative penalty is 0.5 before iteration 10 and 0.1 afterward; all
other candidates retain factor one. ThusApool =A0Pinterior, not the unpenalized score.
The pool quantile is 0.90, 0.95, and 0.98 before iteration 10, from 10 to 29, and from 30
onward. Among eligible hard-unseen candidates, the threshold is the maximum of this quantile
and 0.7Q0.95(Apool). The target admitted fraction decreases linearly from 0.20 at iteration zero to
0.05 at iteration 30. If exceeded, the quantile increases in steps of 0.005, up to 0.995, subject to
the minimum-batch guard; ties and this quantile ceiling mean this is not an unconditional exact
cardinality cap. Before iteration 5, quantile relaxation in steps of 0.02 down to 0.70 may fill the
configured minimum batch (64); afterward underfilled batches are allowed and no information-free
filling is required.
Distances below use coordinates divided by their domain ranges,(0.56, 2.12), with dobs the
nearest retained-coordinate distance. The repulsion kernel and within-batch update are
K(d;ℓ,r) =r + (1−r){1−exp[−(d/ℓ)2]}, (S8)
Robs(x) =K(dobs(x); 0.02, 0.5), R batch(x;B) =
∏
s∈B
K(d(x,s ); 0.03, 0.01), (S9)
p(x|B) = [Apool(x)Robs(x)Rbatch(x;B)]γ
∑
z∈P\B[Apool(z)Robs(z)Rbatch(z;B)]γ. (S10)
HereP is the eligible active pool andγ= 1.5, 2.5, 4 on the same iteration intervals. A selected index
is removed immediately, and the product is updated after every pick. Existing exact coordinates are
excluded using four-decimal coordinate keys; configured recent-selection and normal-boundary-band
cooldowns also apply. Repeated requests in historical accounting must not be equated with successful
unique returns.
S6.3. Topology-aware and three-dimensional selectors
Stage III replaces thefull score bywpN (Aphase) +wsAs +wcAc, with weights(0.45, 0.45, 0.10)
before iteration 30 and(0.25, 0.65, 0.10) thereafter. It does not simply add a new spectral head to
8
the old full score. The spectral proxy is the maximum of a Pfaffian-margin score, an opposite-index
proximity score, and a gapless-support proximity score. With normalized distancesd0,d 1,dg,dt to
trivial, topological, gapless, and trusted-topology observations,
As =PSC max{e−mP/0.02,e−min(d0,d1)/0.06−|d0−d1|/0.06,e−dg/0.06}, (S11)
Ac =PSC{1−e−(dt/0.08)2
}. (S12)
Absent support sets disable their proximity term; absent trusted-topology support givesAc =PSC.
The dimensionless margin ismP = min(|P0|,|Pπ|)/E2
∗, withE∗= max(1,|t|,|λy|,|λz|,|µ|,|JA|,|∆|)
in the code’s energy units and a10−300 denominator floor. Stage III uses the fixed chemical
potential; Stage IV uses the candidate chemical potential. These scores guide selection and confer
no topological label.
Stage IV constructs a nominal 65,536-point proposal pool: 70% global scrambled Sobol, 10%
jittered local thermodynamic-bracket midpoints, 15% jittered topology-bracket midpoints, and 5%
chemical-potential-edge proposals. Missing bracket proposals are backfilled. This is a different
proposal strategy from Stage III’s diagnostic-only brackets. Coordinates are clipped to the recorded
domain and proposal duplicates are removed at eight decimal places. Existing observations are
identified at six decimal places. The saved selection tables, not a nominal pool-size assumption,
determine actual call counts.
The Stage-IV phase score isclip[0.65H + 0.35 clip(¯σ/Q0.95(¯σ), 0, 1), 0, 1], where ¯σ= (σ∆ +σq)/2
and the denominator has a10−12 floor. Its spectral score takes the larger of the Pfaffian-margin
term and the opposite-index term in Eq.(S11), now with distance scale 0.08 and no gapless term.
Coverage is 1−exp[−(dobs/0.04)2], withoutPSC. Distances normalize all three domain ranges
(0.56, 2.12, 2.0).
Channel weights (0.45, 0.40, 0.15) change to (0.25, 0.60, 0.15) at iteration 12. The selector
allocates integer phase/spectral/coverage quotas from these weights; within each channel it samples
the channel score times batch repulsion to power 3. Exhausted channels fall back to the weighted
combined score. Batch repulsion uses ℓ= 0.045, floor 0.02, and the same product update as
Eq. (S9). Thus Stage IV must not be described as the two-dimensional quantile-pool selector
applied unchanged in three dimensions.
S7. SAMPLING EFFICIENCY
The Stage-I candidate universe is the complete241×321 Cartesian grid overkBT/t∈[0, 0.56] and
JA/t∈[0, 2.12], containing 77,361 locations. Its endpoint contains 6,880 unique exact coordinates,
so fexact = 6880/77361 = 0.08893 and Rcall = 77361/6880 = 11.24. Stage II continues on the same
candidate domain and contains every Stage-I endpoint coordinate plus 554 new exact coordinates,
producing the 7,434-coordinate reliability endpoint displayed in main-text Fig. 2. Shared coordinates
have identical phase labels. Selection records exceed endpoint coordinates because repeated requests
and selected coordinates without a retained endpoint return are separate accounting layers. A
boundary-enrichment factor cannot be reconstructed because complete candidate-pool boundary
labels are absent. Per-iteration timings exist, but no matched exhaustive-grid baseline supports a
wall-time speedup.
S8. NUMERICAL DIAGNOSTICS AND OPERATIONAL STOPPING
These criteria are operational controls for the numerical workflow. They are not thermodynamic
observables or universal physical scales. Phase-map change is the fraction of fixed monitor-grid
9
TABLE S5. Stage-resolved selection and retained-coordinate accounting.
Quantity Stage I Stage II Meaning
prescribed candidate locations 77,361 77,361 common Cartesian domain
selected call records 8,192 9,216 requests, including repeats
unique selected coordinates 7,305 7,885 deduplicated requested locations
unique exact endpoint coordinates 6,880 7,434 retained exact observations
training-eligible records 6,880 7,434 clean learning view
trusted exact records 6,104 6,598 reliable subset
selected without endpoint record 425 451 no retained endpoint return
sampling fraction 8.893% – Stage-I discovery headline
exact-location reduction 11.244 × – grid locations, not wall time
Stage-I sampling: locations are not executed-attempt counts
Candidate universe
77,361
Final exact dataset
6,880
Selected records: 8,192; unique selected coordinates: 7,305.
Retained-location fraction: 8.89%; grid/retained ratio: 11.24×.
Location counts do not establish a runtime gain or matched-baseline result.
FIG. S4. Stage-I candidate-universe and exact-location accounting. The reduction is a location-count ratio,
not a wall-time acceleration.
cells whose predicted phase changes between iterations. Boundary displacement is a bidirectional
nearest-neighbor p95 distance between successive extracted boundary point sets after normalization
by the parameter-domain ranges. Coverage is thep95 distance from the current monitor-grid
boundaries to exact samples. Surprise is the fraction of returned labels that differ from the prior
prediction, with trusted and flagged denominators stored separately. Four of five main conditions
must hold for four successive iterations after a five-iteration minimum. The adopted thresholds
reflect the monitor-grid resolution and workflow tolerance, not a physical scale.
10
TABLE S6. Operational diagnostics.
Diagnostic Definition Domain Operational
setting
Interpretation
phase-map
change
changed-cell fraction fixed 241 ×321
monitor
< 0.002 map stability
Normal–SC shift bidirectional p95 normalized domain < 0.004167 boundary
stability
SC–FFLO shift bidirectional p95 normalized domain < 0.004167 boundary
stability
coverage boundary-to-exact
p95
normalized domain < 0.00625 sampling
support
label surprise prediction/return
mismatch
selected calls < 0.05 sequential
stability
patience successive eligible
iterations
iteration history 4 operational
termination
S9. TOPOLOGY DIAGNOSIS AND RECONSTRUCTION SENSITIVITY
Topology is applicable only to thermodynamically accepted, numerically reliable, gapped FFLO
states. The minimum absolute BdG eigenvalue is scanned over the full Brillouin zone. Atk = 0,π,
the analytic Pfaffians are
P0 = (µ−t cos(q/2))2 + ∆ 2−(λy sin(q/2))2
−(JA cos(q/2) +λz sin(q/2))2, (S13)
Pπ= (µ+t cos(q/2))2 + ∆ 2−(λy sin(q/2))2
−(JA cos(q/2) +λz sin(q/2))2. (S14)
Away from the relative Pfaffian tolerance10−8, P0Pπ< 0 givesν= 1 and P0Pπ> 0 givesν= 0.
Analytic values were checked against direct Majorana-basis Pfaffians, and CPU/GPU full-zone
gaps were compared in the archived topology validation. The displayed transition contour is a
deterministic support-restricted reconstruction on a401×401 monitor grid with support radius
0.045. For neighborhoodsk = 6, 8, 12, its arc lengths are 0.8270, 0.8215, and 0.8038, while coverage
p95 is 0.006269, 0.006185, and 0.005913. This sensitivity and the local opposite-Z2 brackets make
the contour a guide to the gap-closing region rather than an exact boundary; nodal alternatives are
not exhaustively excluded.
S10. THREE-DIMENSIONAL EXTENSION
Stage IV samples (kBT/t,J A/t,µ/t) with µ/t∈[−0.5, 1.5]. The main figure shows the ex-
act thermodynamic point cloud, reliable FFLO topology support, and bins centered atµ/t=
−0.45, 0.15, 0.85, and 1.45 with half-width 0.025. No interpolation is performed between these
bins. Archived map-change, proxy-surface displacement, coverage, surprise, and component-count
histories are retained as numerical diagnostics only. The discrete support reveals systematic three-
dimensional structure but does not establish a converged continuous topology surface or complete
atlas.
Main-text Fig. 3(a) contains 3,213 trusted gapped FFLO observations (2,290 cFFLO and 923
tFFLO). The main-text reconstruction retains thek = 8 operational-audit pass and thek = 6
11
TABLE S7. Main-text Fig. 4(c) bin accounting. Each window has half-width 0.025. Trusted here is the
numerical-reliability subset, not the gapped-FFLO topology subset.
µ/tcenter Retained Normal Uniform SC FFLO Trusted
−0.45 272 95 1 176 255
0.15 154 53 4 97 149
0.85 223 57 18 148 208
1.45 142 104 3 35 139
0 5 10 15 20 25
iteration
0
1000
2000
3000
4000
5000
6000
7000
count
all exact
trusted exact
trusted topology
a Sample and trust-count growth
0 5 10 15 20 25
iteration
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
fraction of exact samples
Normal
uniform SC
FFLO
b Thermodynamic composition
0 5 10 15 20 25
iteration
0.2
0.4
0.6
0.8
fraction of applicable SC
ν = 0 ν = 1
c Topology-label composition
23 24 25
iteration
0.0014
0.0016
0.0018
0.0020
0.0022
0.0024
0.0026
map-change fraction
d Topology-map change
23 24 25
iteration
0.0056
0.0057
0.0058
0.0059
0.0060
normalized p95 displacement
 e Proxy-surface p95 displacement
23 24 25
iteration
0.08
0.10
0.12
0.14
trusted-surprise fraction
f Trusted-topology surprise
Stage IV archived numerical stability diagnostics
Raw histories only — diagnostic only; not a physical convergence claim
FIG. S5. Stage-IV numerical histories, without threshold lines. (a) Retained and trusted sample growth.
(b) Thermodynamic composition. (c) Topology composition. (d) Available topology-map changes. (e)
Proxy-surface displacement. (f) Trusted-label surprise. These are iteration diagnostics, not a formally
converged three-dimensional topology surface.
sensitivity deviation; neither is a universal physical convergence criterion. The four three-dimensional
bin totals in Table S7 count retained observations, not estimates of phase-space volume.
12
S11. REPRODUCIBILITY, CODE, DATA, AND ENVIRONMENT
The scientific entry points and the manuscript build serve different purposes. The historical two-
dimensional run configuration identifiesscripts/slurm_exact_oracle_array.sh for pointwise
jobs; active-learning preparation is implemented in ml_phase/active_refine.py. The corre-
sponding archived modules include ml_phase/exact_oracle.py, models.py, acquisition.py,
and topology_oracle.py. The three-dimensional launch package contains scripts/submit_
stageiv_3d_full_loop.sh, scripts/stageiv_3d_select.py, and scripts/slurm_stageiv_
exact_array.sh; selection callsml_phase/stageiv_3d.py with the saved production configura-
tion. These identify scientific entry points, not a claim that running a LaTeX script repeats the
physical calculations.
The export contains archived source snapshots and run configurations, not a public executable
service. Stages I–III record Python 3.10.20 and PyTorch 2.11.0+cu130, with eight round-robin
oracle shards. These environment and deployment entries should not be generalized to all runs: the
Stage-IV launch script defaults to eight cost-aware shards and CPU selection, but shell overrides
and the complete deployment environment are not established by the production JSON alone.
Reproduction additionally requires the launch package, actual job overrides, and platform setup.
Bitwise GPU reproducibility is not established by a fixed pseudorandom seed alone.
The retained datasets, selection tables, and numerical histories underlie the main figures. In
particular, main-text Fig. 2 uses the Stage-II retained-coordinate dataset, Fig. 3 uses the reliable
gapped FFLO subset and support-restricted reconstruction, and Fig. 4 uses the Stage-IV point
cloud with the bin convention in Sec. S10. Table S5 separates requests, unique requested locations,
and retained observations. Total executed evaluations, including failed attempts and repeat work,
cannot be inferred from these three counters alone; no complete executed-attempt ledger is claimed
here.
The accompanying document-source package contains a separatebuild.py that regenerates the
double-column main text, single-column review text, and this Supplement from a clean temporary
directory. This script only builds documents from unchanged figure assets. Input hashes identify
the source versions because the local export contains no Git repository metadata. Neither the
archived scientific package nor these manuscript sources have a confirmed public repository URL or
persistent identifier; public access and long-term deposition remain to be arranged by the authors.
Saved run configurations establish the stage-specific parameter choices, while the export provides
shared and stage-specific source snapshots. A per-run commit or executable hash linking every
Stage-I/II invocation to the shared source snapshot is not available in this package. Accordingly,
the method descriptions document the recovered implementation and recorded configurations; they
are not a demonstrated end-to-end rerun or a complete executed-attempt ledger.