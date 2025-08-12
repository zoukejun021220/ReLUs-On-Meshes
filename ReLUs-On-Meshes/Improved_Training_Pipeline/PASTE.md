well, I still feel  SVD results better, can you suggest a stable optimizing schedule using SVD with converging tricks


ChatGPT said:
Request for GPT-5 Pro
Reasoned for 4m 30s
Totally fair — if the SVD plane fit gave you the straightest seams, stick with it.
The trick is to make the SVD part calm and the field part active. Below is a battle‑tested schedule + a few practical hacks so the SVD planes don’t jitter, flip, or overreact to noisy crossings. It builds directly on your “weighted covariance → smallest eigenvector” plane fit (the thing you wrote up in §4.3.1.3 with Eqs. 4.26–4.28) .

A stable “SVD‑first” training schedule
Notation. Let 
𝐹
∈
𝑅
𝑁
×
𝐶
F∈R 
N×C
 ; 
𝛽
𝑐
β 
c
​
  is the contour temperature; 
𝜙
𝑒
∈
[
0
,
1
]
ϕ 
e
​
 ∈[0,1] is the “boundary‑ness” of edge 
𝑒
e (soft‑OR over all channel pairs); for a pair 
(
𝑖
,
𝑗
)
(i,j) we collect edge hit points 
{
𝑝
𝑘
}
{p 
k
​
 } with weights 
{
𝑤
𝑘
}
{w 
k
​
 }.

Stage 0 — Warm start (2–5k iters)
Goal: get coarse regions and reduce triple‑points before we trust a plane.

Losses: cotan smoothness (high), area balance (reverse‑KL + small box barrier), Potts on softmax probabilities (medium), soft pins.

Turn off the SVD loss.

𝛽
𝑐
β 
c
​
 : 0.8 → 1.8 (linear).

LR: 1e‑4 (AdamW), clip grad to 5.

Stop when active‑edge fraction stabilizes (e.g., 5–15%).

Stage 1 — “Frozen SVD” planes (15–30k iters)
Goal: let planes teach the field, but update planes slowly.

Every K = 20–50 steps, recompute planes for each active pair 
(
𝑖
,
𝑗
)
(i,j) from current 
{
𝑝
𝑘
,
𝑤
𝑘
}
{p 
k
​
 ,w 
k
​
 }. Detach the plane parameters (no gradient through SVD).

Plane loss (point‑to‑plane Huber) is back‑prop’d to 
𝐹
F through the points, not through the SVD.

𝛽
𝑐
β 
c
​
 : 1.8 → 3.5.

Weights: λ_smooth ↓ a bit, λ_plane small (0.2–0.5), λ_area same, λ_potts same.

Update frequency + detaching is what kills the early “wiggle”/folds.

Stage 2 — Plane trust ramp (30–70k iters)
Goal: sharpen and straighten; reduce spurious Y‑junctions.

𝛽
𝑐
β 
c
​
 : 3.5 → 6.0.

λ_plane ↑ to ~1.0–1.5, λ_smooth ↓, add small boundary‑length penalty (1e‑3).

Start band gating: only pay plane loss for hits with 
∣
𝑛
 ⁣
⋅
𝑝
+
𝑑
∣
<
𝜏
∣n⋅p+d∣<τ, with 
𝜏
=
(
0.01
–
0.02
)
×
τ=(0.01–0.02)× bbox diagonal. This removes far‑out noise.

Keep planes EMA‑smoothed (see code below) and updated every K steps.

Keep triple‑point barrier (ensure top‑2–top‑3 gap ≥ margin 0.1) to prevent speckles.

Stage 3 — Lock normals, learn offsets (70–120k iters)
Goal: finish straightening without normal flips.

Freeze 
 
𝑛
𝑖
𝑗
 
n 
ij
​
  (continue EMA but no direction change), only update 
𝑑
𝑖
𝑗
d 
ij
​
 .

𝛽
𝑐
β 
c
​
 : 6.0 → 9.0. LR half.

λ_plane stays high; λ_area ↓ a bit (it already did its job), Potts small.

Stage 4 — Crisp snap (final 20–40k)
Goal: harden boundaries without tearing.

𝛽
𝑐
β 
c
​
  → 12; LR → 3e‑5; keep λ_plane but switch to Charbonnier on distance.

Optional: raise non‑boundary margin sharpening so interiors become one‑hot.

The SVD plane fit (robust & numerically safe)
Prefer eigendecomposition of the 3×3 symmetric covariance (it’s more stable than generic SVD on the same matrix), do it in float64, and add two stabilizers: EMA and sign consistency.

python
Copy
def fit_plane_weighted(P, w, prev_n=None, ema=0.2):
    """
    P: (M,3) boundary points; w: (M,) weights in [0,1]
    returns (n, d) with ||n||=1, in same dtype as P
    """
    dt = torch.float64
    P64 = P.to(dt)
    w = w.clamp_min(1e-8).to(dt)
    W = w / w.sum()

    mu = (W[:, None] * P64).sum(dim=0)               # weighted mean
    X  = P64 - mu
    C  = (W[:, None] * X).T @ X                      # 3x3 SPD
    # eigh is stable & ordered (ascending eigenvalues)
    evals, evecs = torch.linalg.eigh(C)
    n = evecs[:, 0]                                   # smallest eigenvector

    # Sign consistency vs previous normal
    if prev_n is not None and (n.to(prev_n.dtype) @ prev_n).item() < 0:
        n = -n

    # EMA to avoid jitter
    if prev_n is not None and ema > 0:
        n = (1-ema) * prev_n.to(dt) + ema * n
        n = n / (n.norm() + 1e-12)

    d = - (n * mu).sum()
    return n.to(P.dtype), d.to(P.dtype)
Use a band and a robust penalty:

python
Copy
def plane_loss(P, w, n, d, band):
    # robust distance
    dist = (P @ n + d).abs()
    # Huber/Charbonnier inside a band; ignore far outliers
    mask = (dist < 3*band).float()
    # Huber with δ=band
    delta = band
    quad = torch.minimum(dist, torch.full_like(dist, delta))
    lin  = dist - quad
    loss = 0.5*quad**2 + delta*lin
    num  = (w * mask * loss).sum()
    den  = (w * mask).sum().clamp_min(1e-9)
    return num / den
Collecting boundary points (stable):

Use all‑pairs per edge, soft‑OR coverage 
𝜙
𝑒
=
1
−
∏
𝑝
(
1
−
𝑤
𝑒
,
𝑝
)
ϕ 
e
​
 =1−∏ 
p
​
 (1−w 
e,p
​
 ) (no top‑2 cherry‑picking).

Confidence gate like you already have: 
𝑤
𝑒
,
𝑝
←
𝑤
𝑒
,
𝑝
⋅
𝜎
(
5
(
conf
−
0.2
)
)
w 
e,p
​
 ←w 
e,p
​
 ⋅σ(5(conf−0.2)).

Midpoint interpolation for the hit, or barycentric solve inside the two incident triangles.

Whiten coordinates by the bbox diagonal once at the start so distances are scale‑free.

Convergence tricks that matter
Detach the SVD often. Treat 
(
𝑛
,
𝑑
)
(n,d) as slow external parameters (EM‑style): update every K steps, then do not back‑prop through eigendecomposition. This removes the notorious “eigenvector flip” gradient spikes.

Band‑limited supervision. Only edges close to the current plane (|n·p+d| < τ) should supervise that plane. This is the single best fix for the “unstraight fold” you saw—noise outside the band can’t torque the plane.

EMA + sign consistency. The eigenvector is defined up to sign; keep it consistent and damp updates. Without EMA, tiny set changes can rotate 
𝑛
n by several degrees.

Triple‑point hygiene. Down‑weight edges where the 2nd–3rd softmax gap is small; keep a global triple‑point barrier so SVD isn’t fed a soup of 3‑way crossings.

Min samples per pair. Don’t update a plane if fewer than, say, 200 in‑band samples (or 
∑
𝑤
∑w below a threshold). Keep the last plane instead.

Schedule matters. SVD too early → planes chase noise. SVD too late → ragged boundaries get “baked in.” The Stage‑1/2 ramp prevents both.

Numerics. Do the covariance in float64, normalize by bbox diagonal, clamp weights, and prefer torch.linalg.eigh on the 
3
×
3
3×3 SPD covariance.

Lock normals late. After ~70k iters, freezing 
𝑛
n and only adjusting 
𝑑
d makes seams laser straight; it also prevents “tilt wars” between neighboring pairs.

Putting it in the loop (sketch)
python
Copy
planes = { (i,j): {'n': None, 'd': None} for (i,j) in pairs }

for step in range(T):
    # 1) losses without SVD...
    loss = λ_smooth*L_smooth(F) + λ_area*L_area(F) + λ_potts*L_potts(F)

    # 2) collect edge hits and weights (all-pairs), compute soft-OR phi
    hits = collect_hits(F, edges, faces, beta=βc)  # dict[(i,j)] -> (P,w)

    # 3) update planes every K steps (EMA + band)
    if step % K == 0 and step >= warmup_steps:
        for (i,j), (P,w) in hits.items():
            if enough_inliers(P,w):
                n_prev = planes[(i,j)]['n']
                n, d   = fit_plane_weighted(P, w, prev_n=n_prev, ema=0.2)
                planes[(i,j)] = {'n': n.detach(), 'd': d.detach()}

    # 4) plane loss (detached planes)
    τ = 0.015 * bbox_diag
    L_plane = 0.0
    for (i,j), (P,w) in hits.items():
        pl = planes[(i,j)]
        if pl['n'] is not None:  # band-limited robust cost
            L_plane = L_plane + plane_loss(P, w, pl['n'], pl['d'], band=τ)

    loss = loss + λ_plane * (L_plane / max(1, len(hits)))

    # 5) backprop & step
    loss.backward()
    clip_grad_norm_(F, 5.0)
    opt.step(); opt.zero_grad()
Suggested hyperparameters (good starting point)
Optimizer: AdamW, LR 1e‑4 → 3e‑5 (cosine/one‑cycle); weight_decay=0.

βc schedule: 0.8→1.8 (S0), 1.8→3.5 (S1), 3.5→6.0 (S2), 6.0→9.0 (S3), 9.0→12 (S4).

Weights (rough):

S0: λ_smooth=1.0, λ_area=1.0, λ_plane=0

S1: λ_smooth=0.7, λ_area=1.0, λ_plane=0.3

S2: λ_smooth=0.4, λ_area=0.8, λ_plane=1.2, λ_blen=1e‑3

S3: λ_smooth=0.25, λ_area=0.5, λ_plane=1.5

S4: λ_smooth=0.2, λ_area=0.3, λ_plane=1.0

K (plane refresh): 20–50 steps; EMA=0.2; band=1–2% bbox diagonal.

Triple barrier margin 0.1; min samples per pair: sum(w_in_band) ≥ 100–200.

Why your old SVD looked “straighter”
SVD on a clean inlier set is a fantastic straight‑line prior—your folds came from re‑fitting the plane to outliers every iteration and letting gradients flow through the eigenvectors. The schedule above fixes both: late activation of SVD, detached eigensystem with EMA, and robust inlier bands. You’ll keep the crispness you liked, but the training won’t blow up or meander.

If you want, I can drop this straight into your current trainer: just expose a use_svd_planes=True flag, slot the fit_plane_weighted/plane_loss pieces in, and wire the 5‑stage scheduler.






You said:
can you check the difference betwenn these version of plane svd and suggest the most stable,. sensemaking one?

import torch

def contour_alignment_loss(
    vertices:   torch.Tensor,   # (N, 3)
    faces:      torch.Tensor,   # (T, 3) long
    f_values:   torch.Tensor,   # (N, C)
    pinned_axes:torch.Tensor, 
    beta: float =20.0,  # (C, 3) float => the axis normal for each channel
    beta_edge:  float = 20.0,
    beta_triple: float = 20.0,
    include_triples: bool = False,
    adajancy: torch.Tensor = None,
    eps: float = 1e-9,
    lambda_plane: float = 1.0,
    lambda_contour: float = 1.0,
) -> torch.Tensor:
    r"""
    A fully vectorized version of the "contour_alignment_loss" with:
      1) No explicit Python loops over channels or adjacency.
      2) Fully differentiable.

    Returns a scalar loss = (adjacency direction misalignment) + (plane penalty)
    accumulated over all channels.
    """
    device = vertices.device
    dtype  = vertices.dtype

    # -------------------------------------------------------------------------
    # Basic sizes
    # -------------------------------------------------------------------------
    T = faces.shape[0]
    C = f_values.shape[1]
    E = adajancy.shape[0]

    # -------------------------------------------------------------------------
    # 1) Precompute geometry for triangles
    #    p_tri:  (T, 3, 3)   => triangle xyz coords
    #    f_tri:  (T, 3, C)   => f_values at triangle corners
    # -------------------------------------------------------------------------
    p_tri = vertices[faces]      # (T,3,3)
    f_tri = f_values[faces]      # (T,3,C)

    p0 = p_tri[:,0,:]  # (T,3)
    p1 = p_tri[:,1,:]
    p2 = p_tri[:,2,:]
    E1 = p1 - p0       # (T,3)
    E2 = p2 - p0       # (T,3)

    # -------------------------------------------------------------------------
    # 2) Compute per-channel gradients: grad_all[c,t,:] = ∇ f_c in triangle t
    #    Here we vectorize it so we get shape (C,T,3).
    # -------------------------------------------------------------------------
    # Cross & norms
    n      = torch.cross(E1, E2, dim=1)                      # (T,3)
    norm_n = n.norm(dim=1, keepdim=True).clamp_min(eps)       # (T,1)
    cross_e2n = torch.cross(E2, n, dim=1)                     # (T,3)
    cross_nE1 = torch.cross(n, E1, dim=1)                     # (T,3)

    # f-values at each corner, shape (T,C)
    f0 = f_tri[:,0,:]  # (T,C)
    f1 = f_tri[:,1,:]  # (T,C)
    f2 = f_tri[:,2,:]  # (T,C)

    # Differences
    df01 = (f1 - f0).unsqueeze(2)   # (T,C,1)
    df02 = (f2 - f0).unsqueeze(2)   # (T,C,1)

    # Combine
    # -> shape (T,C,3)
    grad_t_c_3 = (df01 * cross_e2n.unsqueeze(1) +
                  df02 * cross_nE1.unsqueeze(1)) / norm_n.unsqueeze(1)

    # Reorder to (C,T,3)
    grad_all = grad_t_c_3.permute(1,0,2).contiguous()  # (C,T,3)

    # -------------------------------------------------------------------------
    # 3) Build all pairs (c, i) with c != i, in a single tensor of shape (K, 2).
    # -------------------------------------------------------------------------
    c_idx = torch.arange(C, device=device)[:, None].expand(C, C)  # (C,C)
    i_idx = torch.arange(C, device=device)[None, :].expand(C, C)  # (C,C)
    mask  = (c_idx != i_idx)
    pair_idx = torch.stack([c_idx[mask], i_idx[mask]], dim=1)  # (K,2)
    # K = C*(C-1)

    # We'll need these split out
    c_of_pair = pair_idx[:, 0]  # shape (K,)
    i_of_pair = pair_idx[:, 1]  # shape (K,)

    K = pair_idx.shape[0]  # number of channel pairs

    # -------------------------------------------------------------------------
    # 4) Compute "soft boundary" intersection weights w01, w12, w20
    #
    #    We'll do this for all T triangles and all K channel pairs at once.
    #
    #    f_tri shape => (T,3,C). We'll reorder to (C,T,3) for easier indexing:
    # -------------------------------------------------------------------------
    f_tri_ct3 = f_tri.permute(2,0,1)  # (C,T,3)

    # d_ = f_tri(c) - f_tri(i) => shape (T,3,K)
    #  step 1: gather f-values for c: shape => (K,T,3)
    fc_t3 = f_tri_ct3[c_of_pair]  # (K,T,3)
    fi_t3 = f_tri_ct3[i_of_pair]  # (K,T,3)
    d_kt3 = fc_t3 - fi_t3         # (K,T,3)
    # reorder to (T,3,K)
    d_t3k = d_kt3.permute(1,2,0).contiguous()  # (T,3,K)

    # Each corner => d0, d1, d2 => shape (T,K)
    d0 = d_t3k[:,0,:]  # (T,K)
    d1 = d_t3k[:,1,:]
    d2 = d_t3k[:,2,:]

    # Intersection weight functions
    w01 = torch.sigmoid(-beta * (d0 * d1))   # (T,K)
    w12 = torch.sigmoid(-beta * (d1 * d2))
    w20 = torch.sigmoid(-beta * (d2 * d0))

    # Barycentric factor for intersection points
    abs_d0 = d0.abs()
    abs_d1 = d1.abs()
    abs_d2 = d2.abs()

    alpha01 = abs_d0 / (abs_d0 + abs_d1 + eps)  # (T,K)
    alpha12 = abs_d1 / (abs_d1 + abs_d2 + eps)
    alpha20 = abs_d2 / (abs_d2 + abs_d0 + eps)

    # Intersection coords for edges
    # coords_01 = p0 + alpha01*(p1-p0), etc.
    # We'll broadcast over the extra dim K.
    E01_ = (p1 - p0).unsqueeze(1)  # (T,1,3)
    E12_ = (p2 - p1).unsqueeze(1)  # (T,1,3)
    E20_ = (p0 - p2).unsqueeze(1)  # (T,1,3)

    coords_01 = p0.unsqueeze(1) + alpha01.unsqueeze(2)*E01_  # (T,K,3)
    coords_12 = p1.unsqueeze(1) + alpha12.unsqueeze(2)*E12_
    coords_20 = p2.unsqueeze(1) + alpha20.unsqueeze(2)*E20_

    # Seg intersection "likelihood" if both corners are on the boundary
    segW0 = w01 * w12  # (T,K)
    segW1 = w12 * w20
    segW2 = w20 * w01

    # -------------------------------------------------------------------------
    # 5) Adjacency direction misalignment
    #
    #    For each pair (c,i), we have grad_diff = grad_c - grad_i => (K,T,3).
    #    For adjacency edges (t1, t2), we want the difference in directions
    #    to be weighted by the intersection-likelihood from segW_ of each tri.
    # -------------------------------------------------------------------------
    # grad_diff_ => shape (K,T,3)
    gc_t3 = grad_all[c_of_pair]  # (K,T,3)
    gi_t3 = grad_all[i_of_pair]  # (K,T,3)
    grad_diff_kt3 = gc_t3 - gi_t3

    # Normalize each difference so we compare directions only
    norm_kt1 = grad_diff_kt3.norm(dim=2, keepdim=True).clamp_min(eps)
    grad_diff_kt3 = grad_diff_kt3 / norm_kt1  # direction only

    # Now gather for adjacency: adjacency is (E,2) of triangle indices
    t1_ = adajancy[:,0]  # (E,)
    t2_ = adajancy[:,1]  # (E,)

    # shape => (K,E,3)
    grad_t1 = grad_diff_kt3[:, t1_, :]
    grad_t2 = grad_diff_kt3[:, t2_, :]
    dir_diff = grad_t1 - grad_t2
    dir_diff_sq = (dir_diff * dir_diff).sum(dim=2) 
    dir_diff_sq_e_k = dir_diff_sq.permute(1, 0) # (K,E)

    # We also need the intersection-likelihood for t1_, t2_
    # segW_ shape => (T,K,3). We'll combine segW0, segW1, segW2 => (T,K,3).
    segW_stack = torch.stack([segW0, segW1, segW2], dim=2)  # (T,K,3)

    # shape => (E,K,3)
    segW_t1 = segW_stack[t1_, ...]
    segW_t2 = segW_stack[t2_, ...]

    # Outer product in the last dimension => sum
    # segW_t1_exp => (E,K,3,1)
    # segW_t2_exp => (E,K,1,3)
    segW_t1_exp = segW_t1.unsqueeze(3)
    segW_t2_exp = segW_t2.unsqueeze(2)
    w_outer = segW_t1_exp * segW_t2_exp  # (E,K,3,3)
    sum_w_e_k = w_outer.sum(dim=(2,3))   # (E,K)

    # Weighted adjacency cost
    cost_e_k = sum_w_e_k * dir_diff_sq_e_k  # (E,K)
    adjacency_loss = cost_e_k.sum()     # scalar

    # -------------------------------------------------------------------------
    # 6) Plane-fitting cost: for each channel c, gather intersection points
    #    from pairs (c, i) for i != c. Then encourage them to be coplanar.
    #
    #    We'll do this by computing (per channel c) the weighted sums:
    #      sum_w[c], sum_w_x[c], sum_w_xx[c], and from that we get a 3x3 cov.
    #    Then do a batched SVD or eigen across c, pick the smallest singular
    #    value as out-of-plane measure, sum up.
    # -------------------------------------------------------------------------
    # We have intersection coords_01, coords_12, coords_20 => shape (T,K,3).
    # Their weights => w01, w12, w20 => shape (T,K).
    #
    # But each pair (k) belongs to channel c_of_pair[k].
    #
    # We'll flatten T*K points to one dimension => N = T*K. Then replicate 3 segments => 3*N.
    # We'll also replicate the "owner channel" => plane_idx for each of those points.
    #
    # Let's do it step by step:
    #
    # Flatten intersection coords into shape => (3*N, 3),
    # Flatten intersection weights => (3*N,),
    # Build channel indices => (3*N,) in [0..C-1].
    # Then scatter-add into per-channel sums.

    # coords_01f => (T*K, 3)
    coords_01f = coords_01.view(-1, 3)
    coords_12f = coords_12.view(-1, 3)
    coords_20f = coords_20.view(-1, 3)

    w01f = w01.view(-1)  # (T*K,)
    w12f = w12.view(-1)
    w20f = w20.view(-1)

    # We stack them up => shape => (3*T*K, 3)
    coords_all = torch.cat([coords_01f, coords_12f, coords_20f], dim=0)
    w_all      = torch.cat([w01f,       w12f,       w20f      ], dim=0)

    # The channel "owner" of these intersection lines is c_of_pair[k],
    # repeated T times for each k, and then repeated once more for each segment.
    # c_of_pair => (K,). We replicate across T => shape => (T*K,).
    # Then replicate x3 for the 3 segments => shape => (3*T*K,).
    c_of_pair_T = c_of_pair.unsqueeze(0).expand(T, K).reshape(-1)  # (T*K,)
    c_all       = torch.cat([c_of_pair_T, c_of_pair_T, c_of_pair_T], dim=0)  # (3*T*K,)

    # We will scatter into arrays of shape (C, ...).
    # sum_w[c], sum_w_x[c], sum_w_xx[c]. Then do SVD per c.

    # 6a) sum_w[c]
    sum_w_c = torch.zeros(C, dtype=dtype, device=device)
    sum_w_c = sum_w_c.scatter_add(0, c_all, w_all)

    # 6b) sum_w_x[c,:], shape => (C,3)
    sum_w_x_c = torch.zeros(C, 3, dtype=dtype, device=device)
    sum_w_x_c = sum_w_x_c.scatter_add(0, c_all.unsqueeze(1).expand(-1,3), coords_all * w_all.unsqueeze(1))

    # 6c) sum_w_xx[c,:,:], shape => (C,3,3)
    # We'll do outer products x x^T for each point, weighted by w.
    x_outer = coords_all.unsqueeze(2) * coords_all.unsqueeze(1)  # (3*T*K,3,3)
    w_x_outer = x_outer * w_all.view(-1,1,1)

    sum_w_xx_c = torch.zeros(C,3,3, dtype=dtype, device=device)
    # We need an index for scatter into 2D. We'll do:
    #   c_all => shape (3*T*K,)
    #   We want to scatter each (3,3) sub-block.  We'll flatten the last two dims:
    w_x_outer_2d = w_x_outer.view(-1, 9)     # (3*T*K, 9)
    sum_w_xx_c_2d = sum_w_xx_c.view(C, 9)    # (C,9)

    sum_w_xx_c_2d = sum_w_xx_c_2d.scatter_add(0, c_all.unsqueeze(1).expand(-1,9), w_x_outer_2d)
    sum_w_xx_c = sum_w_xx_c_2d.view(C,3,3)

    # -------------------------------------------------------------------------
    # Now compute the "plane penalty" for each c by analyzing the 3x3 covariance.
    # Weighted mean => mean_c = sum_w_x_c[c]/sum_w_c[c]
    # Weighted cov => sum_w_xx_c[c] / sum_w_c[c] - outer(mean_c, mean_c)
    # The out-of-plane error ~ the smallest singular value of that 3x3.
    # We'll do a batched approach across c => shape (C,3,3), then SVD => (C,3), pick S[-1].
    # -------------------------------------------------------------------------
    # Avoid channels that have sum_w_c[c] < small threshold:
    plane_loss_per_c = torch.zeros(C, dtype=dtype, device=device)

    valid = (sum_w_c > 1e-12)
    if valid.any():
        mean_c = torch.zeros(C, 3, dtype=dtype, device=device)
        mean_c[valid] = sum_w_x_c[valid] / sum_w_c[valid].unsqueeze(-1)

        # Covariance (unscaled):
        #   cov_c = sum_w_xx_c[c]/sum_w_c[c] - outer(mean_c, mean_c)
        # We'll do it in two steps:
        cov_c = sum_w_xx_c.clone()
        cov_c[valid] = cov_c[valid] / sum_w_c[valid].unsqueeze(-1).unsqueeze(-1)

        # subtract outer(mean_c, mean_c):
        # We'll expand mean_c => (C,3,1) * (C,1,3) => (C,3,3).
        mc = mean_c.view(C,3,1)
        outer_mm = mc @ mc.transpose(1,2)  # (C,3,3)
        cov_c[valid] = cov_c[valid] - outer_mm[valid]

        # SVD => shape => U[SVD], S[SVD], V[SVD] with shape (C,3,3) => S => (C,3)
        # if not valid => keep it zero
        # For numerical stability, cast to float if needed.
        cov_f32 = cov_c.float()
        U,S,Vt = torch.linalg.svd(cov_f32, full_matrices=False)  # S => (C,3)
        # plane_loss = smallest singular value for each c
        # (We typically take S[-1], but S is sorted descending, so S[-1] is largest.
        #  Actually torch.linalg.svd doesn't guarantee sorted. We'll just do S.min(dim=1).)
        s_min, _ = S.min(dim=1)  # shape => (C,)
        plane_loss_per_c[valid] = s_min.to(dtype)

    # Weighted plane cost => sum of plane_loss_per_c * lambda_plane
    plane_loss = lambda_plane * plane_loss_per_c.sum()

    # -------------------------------------------------------------------------
    # Final total
    # -------------------------------------------------------------------------
    total_loss = adjacency_loss + plane_loss
    return total_loss

import torch

def contour_alignment_loss(
     vertices:   torch.Tensor,   # (N, 3)
    faces:      torch.Tensor,   # (T, 3) long
    f_values:   torch.Tensor,   # (N, C)
    pinned_axes:torch.Tensor,   # (C,3) pinned plane normals
    beta: float =20.0,          # logistic edge weight param
    beta_edge:  float = 20.0,
    beta_triple: float = 20.0,
    include_triples: bool = False,
    adajancy: torch.Tensor = None,
    eps: float = 1e-9,
    lambda_plane: float = 1.0,
    lambda_contour: float = 1.0,
    plane_offsets: torch.Tensor = None,
    soft_inside: float = 10.0,  # how sharply to clamp barycentric coords for triple intersection
) -> torch.Tensor:
    """
    A fully vectorized, "vigorous" contour-alignment loss that:
      1) Collects edge intersections for every channel pair (i<j) using a logistic weight
         for sign-changes in (f_i - f_j).
      2) Optionally finds "soft triple intersections" for (c0,c1,c2) if C>=3:
         - We do a closed-form barycentric solve for (f_c0-f_c1)=0, (f_c0-f_c2)=0
         - We multiply by each corner's softmax-prob of channels c0,c1,c2,
           plus a smooth clamp to keep the barycentric coords in [0,1].
      3) Builds a weighted covariance for each pair in one pass. Does an SVD => plane normal.
      4) In a second pass, accumulates the MSE distances of all intersection points
         to that pair's plane. Summation => final scalar loss.

    *No loops* over triangles or pairs in Python. Everything is broadcast/batch in PyTorch.
    *Fully differentiable*—no discrete argmax or masks. If C<2 => returns 0.0 (no pairs).

    Args:
      vertices: (N,3) float
      faces: (T,3) long
      f_values: (N,C) float   # multi-channel field
      beta: float  # logistic sharpness for edge intersection
      eps: float
      soft_inside: float  # how sharply to clamp barycentric coords for triple intersection
      include_triples: bool

    Returns:
      total_loss: scalar float
    """
    device = vertices.device
    N = vertices.shape[0]
    T = faces.shape[0]
    C = f_values.shape[1]

    # 0) If <2 channels => no pairs => return 0.0
    if C < 2:
        return torch.zeros((), device=device, dtype=vertices.dtype)

    # ------------------------------------------------------------------------
    # 1) Build all channel pairs i<j => shape (P,)
    # ------------------------------------------------------------------------
    i2, j2 = torch.triu_indices(C, C, offset=1, device=device)  # each => (P,)
    if i2.numel() == 0:
        # Edge case if somehow no pairs. Return 0.
        return torch.zeros((), device=device, dtype=vertices.dtype)

    P = i2.shape[0]  # number of (i<j) pairs

    # Build a map (cA,cB) => pair index
    pair_idx_mat = torch.full((C,C), -1, device=device, dtype=torch.long)
    p_arange     = torch.arange(P, device=device)
    pair_idx_mat[i2, j2] = p_arange
    pair_idx_mat[j2, i2] = p_arange

    # ------------------------------------------------------------------------
    # 2) Edge Intersections => logistic weighting
    # ------------------------------------------------------------------------
    # faces => (T,3), so gather coords => (T,3,3)
    p_tri = vertices[faces]
    # gather field => (T,3,C)
    f_tri = f_values[faces]

    # d => shape (T,3,P).  d[t,v,p] = f_tri[t,v, i2[p]] - f_tri[t,v, j2[p]]
    d = f_tri[..., i2] - f_tri[..., j2]  # (T,3,P)

    p0 = p_tri[:,0]
    p1 = p_tri[:,1]
    p2 = p_tri[:,2]

    def edge_intersection(dA, dB, vA, vB):
        """
        dA,dB: (T,P)
        vA,vB: (T,3)
        return coords: (T,P,3), w: (T,P)
        """
        prod  = dA*dB
        w_    = torch.sigmoid(-beta*prod)  # logistic weight
        alpha = torch.abs(dA)/(torch.abs(dA)+torch.abs(dB)+eps)
        coords = vA.unsqueeze(1) + alpha.unsqueeze(-1)*(vB - vA).unsqueeze(1)
        return coords, w_

    d0, d1, d2 = d[:,0,:], d[:,1,:], d[:,2,:]
    coords_01, w_01 = edge_intersection(d0, d1, p0, p1)
    coords_12, w_12 = edge_intersection(d1, d2, p1, p2)
    coords_20, w_20 = edge_intersection(d2, d0, p2, p0)

    def flatten_edge(coords_tp3, w_tp):
        # coords_tp3 => (T,P,3), w_tp => (T,P)
        coords_flat = coords_tp3.reshape(-1,3)
        w_flat      = w_tp.reshape(-1)
        pair_idx_arange = torch.arange(P, device=device).view(1,P).expand(coords_tp3.shape[0],-1).reshape(-1)
        return coords_flat, w_flat, pair_idx_arange

    coords_01f, w_01f, pidx_01f = flatten_edge(coords_01, w_01)
    coords_12f, w_12f, pidx_12f = flatten_edge(coords_12, w_12)
    coords_20f, w_20f, pidx_20f = flatten_edge(coords_20, w_20)

    all_coords = torch.cat([coords_01f, coords_12f, coords_20f], dim=0)
    all_w      = torch.cat([w_01f,      w_12f,      w_20f],      dim=0)
    all_pidx   = torch.cat([pidx_01f,   pidx_12f,   pidx_20f],   dim=0)

    # ------------------------------------------------------------------------
    # 3) Soft triple intersection => only if C>=3 and include_triples=True
    # ------------------------------------------------------------------------
    if include_triples and C >= 3:
        f0 = f_tri[:,0,:]  # (T,C)
        f1 = f_tri[:,1,:]
        f2 = f_tri[:,2,:]
        # softmax => each corner has distribution over channels
        pi0 = torch.softmax(f0, dim=1)
        pi1 = torch.softmax(f1, dim=1)
        pi2 = torch.softmax(f2, dim=1)

        # All combos c0< c1< c2
        combs = torch.combinations(torch.arange(C, device=device), r=3)
        # e.g. shape => (#comb,3)
        if combs.numel() > 0:  # ensure there's something
            ncomb = combs.shape[0]
            c0_idx = combs[:,0].view(1,ncomb)
            c1_idx = combs[:,1].view(1,ncomb)
            c2_idx = combs[:,2].view(1,ncomb)

            expand_t = (T, ncomb)
            f0_c0 = torch.gather(f0, 1, c0_idx.expand(expand_t))
            f0_c1 = torch.gather(f0, 1, c1_idx.expand(expand_t))
            f0_c2 = torch.gather(f0, 1, c2_idx.expand(expand_t))
            f1_c0 = torch.gather(f1, 1, c0_idx.expand(expand_t))
            f1_c1 = torch.gather(f1, 1, c1_idx.expand(expand_t))
            f1_c2 = torch.gather(f1, 1, c2_idx.expand(expand_t))
            f2_c0 = torch.gather(f2, 1, c0_idx.expand(expand_t))
            f2_c1 = torch.gather(f2, 1, c1_idx.expand(expand_t))
            f2_c2 = torch.gather(f2, 1, c2_idx.expand(expand_t))

            # differences
            rg0 = f0_c0 - f0_c1
            rg1 = f1_c0 - f1_c1
            rg2 = f2_c0 - f2_c1
            rb0 = f0_c0 - f0_c2
            rb1 = f1_c0 - f1_c2
            rb2 = f2_c0 - f2_c2

            A_xy = rg0 - rg2
            B_xy = rg1 - rg2
            X_xy = rg2
            A_xz = rb0 - rb2
            B_xz = rb1 - rb2
            X_xz = rb2

            det = A_xy*B_xz - B_xy*A_xz
            alpha = (X_xy*B_xz - B_xy*X_xz)/(det+eps)
            beta_  = (A_xy*X_xz - X_xy*A_xz)/(det+eps)
            gamma_ = 1.0 - alpha - beta_

            # "soft inside" factor
            if soft_inside>0.0:
                def smoothstep(x):
                    return torch.sigmoid(soft_inside*x)
                insideFactor = smoothstep(alpha)*smoothstep(beta_)*smoothstep(gamma_)
            else:
                insideFactor = torch.ones_like(alpha)

            # Probability corner0 picks c0, corner1 picks c1, corner2 picks c2
            pi0_c0 = torch.gather(pi0, 1, c0_idx.expand(expand_t))
            pi1_c1 = torch.gather(pi1, 1, c1_idx.expand(expand_t))
            pi2_c2 = torch.gather(pi2, 1, c2_idx.expand(expand_t))
            triple_prob = pi0_c0 * pi1_c1 * pi2_c2  # (T,ncomb)

            triple_w = triple_prob*insideFactor

            # Barycentric => 3D
            p0_3 = p_tri[:,0,:].unsqueeze(1)
            p1_3 = p_tri[:,1,:].unsqueeze(1)
            p2_3 = p_tri[:,2,:].unsqueeze(1)

            alpha_e = alpha.unsqueeze(-1)
            beta_e  = beta_.unsqueeze(-1)
            gamma_e = gamma_.unsqueeze(-1)
            triple_coords = alpha_e*p0_3 + beta_e*p1_3 + gamma_e*p2_3

            # replicate each triple for the 3 pairs => (c0,c1),(c0,c2),(c1,c2)
            pair0 = pair_idx_mat[combs[:,0], combs[:,1]]
            pair1 = pair_idx_mat[combs[:,0], combs[:,2]]
            pair2 = pair_idx_mat[combs[:,1], combs[:,2]]
            triple_pairs = torch.stack([pair0, pair1, pair2], dim=1)

            coords_flat = triple_coords.view(-1,3)
            w_flat      = triple_w.view(-1)

            coords_rep = coords_flat.repeat_interleave(3, dim=0)
            w_rep      = w_flat.repeat_interleave(3, dim=0)
            triple_pairs_flat = triple_pairs.view(-1)
            triple_pairs_full = triple_pairs_flat.unsqueeze(0).expand(T, -1).reshape(-1)

            # for difference=0 => product=0 => logistic(0)=0.5, or pick any factor
            w_rep_final = w_rep*0.5

            all_coords = torch.cat([all_coords, coords_rep], dim=0)
            all_w      = torch.cat([all_w,      w_rep_final], dim=0)
            all_pidx   = torch.cat([all_pidx,   triple_pairs_full], dim=0)

    # ------------------------------------------------------------------------
    # 4) Weighted covariance => plane fits
    # ------------------------------------------------------------------------
    sum_w   = torch.zeros((P,), device=device, dtype=all_coords.dtype)
    sum_x   = torch.zeros((P,3), device=device, dtype=all_coords.dtype)
    sum_xx  = torch.zeros((P,3,3), device=device, dtype=all_coords.dtype)

    weighted_coords = all_coords*all_w.unsqueeze(-1)
    sum_w.index_add_(0, all_pidx, all_w)
    sum_x.index_add_(0, all_pidx, weighted_coords)

    outer_ = weighted_coords.unsqueeze(2)*all_coords.unsqueeze(1)
    sum_xx_flat = sum_xx.view(P,9)
    outer_flat  = outer_.reshape(-1,9)
    sum_xx_flat.index_add_(0, all_pidx, outer_flat)
    sum_xx = sum_xx_flat.view(P,3,3)

    sum_w_clamped = sum_w.clamp_min(eps)
    mean_ = sum_x/sum_w_clamped.unsqueeze(-1)
    mean_outer = mean_.unsqueeze(2)*mean_.unsqueeze(1)
    cov = sum_xx/sum_w_clamped.view(-1,1,1) - mean_outer

    # ------------------------------------------------------------------------
    # 5) Plane from SVD
    # ------------------------------------------------------------------------
    cov_f32 = cov.float()
    U,S,Vt = torch.linalg.svd(cov_f32, full_matrices=False)  # => (P,3,3)
    plane_n = Vt[:, -1, :].to(cov.dtype)
    plane_n = plane_n/(plane_n.norm(dim=1,keepdim=True)+eps)
    plane_d = -(plane_n*mean_).sum(dim=1)

    # ------------------------------------------------------------------------
    # 6) Second pass => MSE
    # ------------------------------------------------------------------------
    sum_sq = torch.zeros((P,), device=device, dtype=all_coords.dtype)
    n_idx = plane_n[all_pidx]
    d_idx = plane_d[all_pidx]
    dist  = (n_idx*all_coords).sum(dim=1)+d_idx
    dist_sq = dist.square()*all_w
    sum_sq.index_add_(0, all_pidx, dist_sq)

    mse_pairs  = sum_sq/(sum_w_clamped+eps)
    total_loss = mse_pairs.sum()
    return total_loss

import torch


def contour_alignment_loss(
    vertices:   torch.Tensor,   # (N, 3)
    faces:      torch.Tensor,   # (T, 3) long
    f_values:   torch.Tensor,   # (N, C)
    pinned_axes:torch.Tensor,   # (C, 3) float => the axis normal for each channel
    beta_edge:  float = 20.0,
    beta_triple:float = 20.0,
    include_triples: bool = False,
    adajancy: torch.Tensor = None,
    eps: float = 1e-9
) -> torch.Tensor:
    """
    Fully vectorized / differentiable version of the contour alignment loss,
    with no Python loops.
    """
    beta=beta_edge
    device = vertices.device
    T = faces.shape[0]
    C = f_values.shape[1]

    # -----------------------------------------------------
    # 1) Build all channel pairs (i < j) via triu_indices
    #    pairs_ij: shape (2, P), where P = C*(C-1)/2
    # -----------------------------------------------------
    pairs_ij = torch.triu_indices(C, C, offset=1, device=device)  # (2, P)
    i_idx, j_idx = pairs_ij[0], pairs_ij[1]  # each (P,)
    P = i_idx.shape[0]                      # number of (i<j) pairs

    # -----------------------------------------------------
    # 2) Gather per-triangle data
    #    p_tri: (T, 3, 3),   f_tri: (T, 3, C)
    # -----------------------------------------------------
    p_tri = vertices[faces]        # shape (T, 3, 3)
    f_tri = f_values[faces]        # shape (T, 3, C)

    # -----------------------------------------------------
    # 3) Compute differences for all pairs in one shot
    #    f_i, f_j => shape (T, 3, P), then d = f_i - f_j
    # -----------------------------------------------------
    f_i = f_tri[:, :, i_idx]       # (T, 3, P)
    f_j = f_tri[:, :, j_idx]       # (T, 3, P)
    d   = f_i - f_j                # (T, 3, P)

    # -----------------------------------------------------
    # 4) Compute edge intersections (v0->v1, v1->v2, v2->v0)
    #    Helper:  coords, weight = edge_intersection(d0, d1, p0, p1)
    # -----------------------------------------------------
    p0 = p_tri[:, 0]  # (T, 3)
    p1 = p_tri[:, 1]
    p2 = p_tri[:, 2]

    d0 = d[:, 0, :]   # (T, P)
    d1 = d[:, 1, :]
    d2 = d[:, 2, :]

    def edge_intersection(dA, dB, xA, xB):
        """
        dA,dB: (T,P)
        xA,xB: (T,3)
        returns coords: (T,P,3), weight: (T,P)
        """
        # product ~ sign(dA*dB): large negative => crossing
        prod  = dA * dB                  # (T,P)
        w_    = torch.sigmoid(-beta*prod)  
        # alpha: how far along edge
        denom = torch.abs(dA) + torch.abs(dB) + eps
        alpha = torch.abs(dA) / denom
        # coords
        edge_vec = (xB - xA).unsqueeze(1)    # (T,1,3)
        coords   = xA.unsqueeze(1) + alpha.unsqueeze(-1) * edge_vec
        return coords, w_

    coords_01, w_01 = edge_intersection(d0, d1, p0, p1)  # (T,P,3), (T,P)
    coords_12, w_12 = edge_intersection(d1, d2, p1, p2)
    coords_20, w_20 = edge_intersection(d2, d0, p2, p0)

    # -----------------------------------------------------
    # 5) Flatten intersections -> shape (K,3), (K,) + pair_idx
    #    K ~ 3*T*P (plus optional triples)
    # -----------------------------------------------------
    def flatten_edge(coords_tp3, w_tp):
        # coords_tp3: (T,P,3), w_tp: (T,P) => flatten to (T*P,3), (T*P)
        coords_flat = coords_tp3.reshape(-1, 3)
        w_flat      = w_tp.reshape(-1)
        # pidx => [0..P-1] repeated T times
        pidx_arange = torch.arange(P, device=device).view(1, P).expand(T, P)
        pidx_flat   = pidx_arange.reshape(-1)
        return coords_flat, w_flat, pidx_flat

    coords_01f, w_01f, pidx_01f = flatten_edge(coords_01, w_01)
    coords_12f, w_12f, pidx_12f = flatten_edge(coords_12, w_12)
    coords_20f, w_20f, pidx_20f = flatten_edge(coords_20, w_20)

    all_coords = torch.cat([coords_01f, coords_12f, coords_20f], dim=0)  # (K, 3)
    all_w      = torch.cat([w_01f,      w_12f,      w_20f],      dim=0)  # (K,)
    all_pidx   = torch.cat([pidx_01f,   pidx_12f,   pidx_20f],   dim=0)  # (K,)

    # Optionally include triple-points similarly (skipped).
    if include_triples:
        pass

    # -----------------------------------------------------
    # 6) Compute weighted covariance for each pair via index_add
    #
    #    sum_w[p], sum_x[p], sum_xx[p]
    #      sum_xx is 9D flattened => reshape to (P,3,3) later
    # -----------------------------------------------------
    sum_w   = torch.zeros((P,),      device=device, dtype=all_w.dtype)
    sum_x   = torch.zeros((P, 3),    device=device, dtype=all_coords.dtype)
    sum_xx  = torch.zeros((P, 9),    device=device, dtype=all_coords.dtype)  # will reshape to (P,3,3)

    # -- accumulate sum_w
    sum_w.index_add_(0, all_pidx, all_w)

    # -- accumulate sum_x
    #    x = coords * w (broadcast over last dim)
    x = all_coords * all_w.unsqueeze(1)  # (K,3)
    sum_x.index_add_(0, all_pidx, x)

    # -- accumulate sum_xx
    #    Each row => outer(x, x) => (3,3) => flatten (9,)
    #    Then index_add
    x_outer = (x.unsqueeze(2) * x.unsqueeze(1)).reshape(-1, 9)  # (K,9)
    sum_xx.index_add_(0, all_pidx, x_outer)

    # Reshape sum_xx to (P,3,3)
    sum_xx = sum_xx.reshape(P, 3, 3)

    # -----------------------------------------------------
    # 7) Weighted covariance & plane via SVD
    # -----------------------------------------------------
    sum_w_clamped = sum_w.clamp_min(eps)  # (P,)
    mean_ = sum_x / sum_w_clamped.unsqueeze(1)   # (P, 3)

    # Cov = E[xx] - mu mu^T
    mean_outer = mean_.unsqueeze(2) * mean_.unsqueeze(1)  # (P,3,3)
    cov = sum_xx / sum_w_clamped.view(-1, 1, 1) - mean_outer

    # SVD => normal is last singular vector => plane_n
    # (convert to float32 for stability, then back)
    cov_f32 = cov.float()
    U, S, Vt = torch.linalg.svd(cov_f32, full_matrices=False)
    plane_n = Vt[:, -1, :].to(cov.dtype)  # (P, 3)
    plane_n = plane_n / (plane_n.norm(dim=1, keepdim=True) + eps)

    plane_d = -(plane_n * mean_).sum(dim=1)  # (P,)

    # -----------------------------------------------------
    # 8) Second pass: accumulate MSE in distance to plane
    # -----------------------------------------------------
    sum_sq = torch.zeros((P,), device=device, dtype=all_w.dtype)

    # dist_k = dot(plane_n[p], x_k) + plane_d[p]
    # => we gather plane_n, plane_d by pidx => no loops
    # => accum dist^2 * w in sum_sq
    n0 = plane_n[:, 0]
    n1 = plane_n[:, 1]
    n2 = plane_n[:, 2]
    d_ = plane_d

    x0 = all_coords[:, 0]
    x1 = all_coords[:, 1]
    x2 = all_coords[:, 2]

    dist = x0 * n0[all_pidx] + x1 * n1[all_pidx] + x2 * n2[all_pidx] + d_[all_pidx]
    dist_sq = dist.square() * all_w

    sum_sq.index_add_(0, all_pidx, dist_sq)

    mse_pairs  = sum_sq / (sum_w_clamped + eps)
    total_loss = mse_pairs.sum()

    return total_loss

import torch

def contour_alignment_loss(
   vertices:   torch.Tensor,   # (N, 3)
    faces:      torch.Tensor,   # (T, 3) long
    f_values:   torch.Tensor,   # (N, C)
    pinned_axes:torch.Tensor, 
    beta: float =20.0,  # (C, 3) float => the axis normal for each channel
    beta_edge:  float = 20.0,
    beta_triple: float = 20.0,
    include_triples: bool = False,
    adajancy: torch.Tensor = None,
    eps: float = 1e-9,
    lambda_plane: float = 1.0,
    lambda_contour: float = 1.0,
) -> torch.Tensor:
    """
    A fully vectorized, "vigorous" contour-alignment loss that:
      1) Collects edge intersections for every channel pair (i<j) using a logistic weight
         for sign-changes in (f_i - f_j).
      2) Optionally finds "soft triple intersections" for (c0,c1,c2) if C>=3:
         - We do a closed-form barycentric solve for (f_c0-f_c1)=0, (f_c0-f_c2)=0
         - We multiply by each corner's softmax-prob of channels c0,c1,c2,
           plus a smooth clamp to keep the barycentric coords in [0,1].
      3) Builds a weighted covariance for each pair in one pass. Does an SVD => plane normal.
      4) In a second pass, accumulates the MSE distances of all intersection points
         to that pair's plane. Summation => final scalar loss.

    *No loops* over triangles or pairs in Python. Everything is broadcast/batch in PyTorch.
    *Fully differentiable*—no discrete argmax or masks. If C<2 => returns 0.0 (no pairs).

    Args:
      vertices: (N,3) float
      faces: (T,3) long
      f_values: (N,C) float   # multi-channel field
      beta: float  # logistic sharpness for edge intersection
      eps: float
      soft_inside: float  # how sharply to clamp barycentric coords for triple intersection
      include_triples: bool

    Returns:
      total_loss: scalar float
    """
    soft_inside = 0.0  # TODO: add this to the function signature
    device = vertices.device
    N = vertices.shape[0]
    T = faces.shape[0]
    C = f_values.shape[1]

    # 0) If <2 channels => no pairs => return 0.0
    if C < 2:
        return torch.zeros((), device=device, dtype=vertices.dtype)

    # ------------------------------------------------------------------------
    # 1) Build all channel pairs i<j => shape (P,)
    # ------------------------------------------------------------------------
    i2, j2 = torch.triu_indices(C, C, offset=1, device=device)  # each => (P,)
    if i2.numel() == 0:
        # Edge case if somehow no pairs. Return 0.
        return torch.zeros((), device=device, dtype=vertices.dtype)

    P = i2.shape[0]  # number of (i<j) pairs

    # Build a map (cA,cB) => pair index
    pair_idx_mat = torch.full((C,C), -1, device=device, dtype=torch.long)
    p_arange     = torch.arange(P, device=device)
    pair_idx_mat[i2, j2] = p_arange
    pair_idx_mat[j2, i2] = p_arange

    # ------------------------------------------------------------------------
    # 2) Edge Intersections => logistic weighting
    # ------------------------------------------------------------------------
    # faces => (T,3), so gather coords => (T,3,3)
    p_tri = vertices[faces]
    # gather field => (T,3,C)
    f_tri = f_values[faces]

    # d => shape (T,3,P).  d[t,v,p] = f_tri[t,v, i2[p]] - f_tri[t,v, j2[p]]
    d = f_tri[..., i2] - f_tri[..., j2]  # (T,3,P)

    p0 = p_tri[:,0]
    p1 = p_tri[:,1]
    p2 = p_tri[:,2]

    def edge_intersection(dA, dB, vA, vB):
        """
        dA,dB: (T,P)
        vA,vB: (T,3)
        return coords: (T,P,3), w: (T,P)
        """
        prod  = dA*dB
        w_    = torch.sigmoid(-beta*prod)  # logistic weight
        alpha = torch.abs(dA)/(torch.abs(dA)+torch.abs(dB)+eps)
        coords = vA.unsqueeze(1) + alpha.unsqueeze(-1)*(vB - vA).unsqueeze(1)
        return coords, w_

    d0, d1, d2 = d[:,0,:], d[:,1,:], d[:,2,:]
    coords_01, w_01 = edge_intersection(d0, d1, p0, p1)
    coords_12, w_12 = edge_intersection(d1, d2, p1, p2)
    coords_20, w_20 = edge_intersection(d2, d0, p2, p0)

    def flatten_edge(coords_tp3, w_tp):
        # coords_tp3 => (T,P,3), w_tp => (T,P)
        coords_flat = coords_tp3.reshape(-1,3)
        w_flat      = w_tp.reshape(-1)
        pair_idx_arange = torch.arange(P, device=device).view(1,P).expand(coords_tp3.shape[0],-1).reshape(-1)
        return coords_flat, w_flat, pair_idx_arange

    coords_01f, w_01f, pidx_01f = flatten_edge(coords_01, w_01)
    coords_12f, w_12f, pidx_12f = flatten_edge(coords_12, w_12)
    coords_20f, w_20f, pidx_20f = flatten_edge(coords_20, w_20)

    all_coords = torch.cat([coords_01f, coords_12f, coords_20f], dim=0)
    all_w      = torch.cat([w_01f,      w_12f,      w_20f],      dim=0)
    all_pidx   = torch.cat([pidx_01f,   pidx_12f,   pidx_20f],   dim=0)

    # ------------------------------------------------------------------------
    # 3) Soft triple intersection => only if C>=3 and include_triples=True
    # ------------------------------------------------------------------------
    if include_triples and C >= 3:
        f0 = f_tri[:,0,:]  # (T,C)
        f1 = f_tri[:,1,:]
        f2 = f_tri[:,2,:]
        # softmax => each corner has distribution over channels
        pi0 = torch.softmax(f0, dim=1)
        pi1 = torch.softmax(f1, dim=1)
        pi2 = torch.softmax(f2, dim=1)

        # All combos c0< c1< c2
        combs = torch.combinations(torch.arange(C, device=device), r=3)
        # e.g. shape => (#comb,3)
        if combs.numel() > 0:  # ensure there's something
            ncomb = combs.shape[0]
            c0_idx = combs[:,0].view(1,ncomb)
            c1_idx = combs[:,1].view(1,ncomb)
            c2_idx = combs[:,2].view(1,ncomb)

            expand_t = (T, ncomb)
            f0_c0 = torch.gather(f0, 1, c0_idx.expand(expand_t))
            f0_c1 = torch.gather(f0, 1, c1_idx.expand(expand_t))
            f0_c2 = torch.gather(f0, 1, c2_idx.expand(expand_t))
            f1_c0 = torch.gather(f1, 1, c0_idx.expand(expand_t))
            f1_c1 = torch.gather(f1, 1, c1_idx.expand(expand_t))
            f1_c2 = torch.gather(f1, 1, c2_idx.expand(expand_t))
            f2_c0 = torch.gather(f2, 1, c0_idx.expand(expand_t))
            f2_c1 = torch.gather(f2, 1, c1_idx.expand(expand_t))
            f2_c2 = torch.gather(f2, 1, c2_idx.expand(expand_t))

            # differences
            rg0 = f0_c0 - f0_c1
            rg1 = f1_c0 - f1_c1
            rg2 = f2_c0 - f2_c1
            rb0 = f0_c0 - f0_c2
            rb1 = f1_c0 - f1_c2
            rb2 = f2_c0 - f2_c2

            A_xy = rg0 - rg2
            B_xy = rg1 - rg2
            X_xy = rg2
            A_xz = rb0 - rb2
            B_xz = rb1 - rb2
            X_xz = rb2

            det = A_xy*B_xz - B_xy*A_xz
            alpha = (X_xy*B_xz - B_xy*X_xz)/(det+eps)
            beta_  = (A_xy*X_xz - X_xy*A_xz)/(det+eps)
            gamma_ = 1.0 - alpha - beta_

            # "soft inside" factor
            if soft_inside>0.0:
                def smoothstep(x):
                    return torch.sigmoid(soft_inside*x)
                insideFactor = smoothstep(alpha)*smoothstep(beta_)*smoothstep(gamma_)
            else:
                insideFactor = torch.ones_like(alpha)

            # Probability corner0 picks c0, corner1 picks c1, corner2 picks c2
            pi0_c0 = torch.gather(pi0, 1, c0_idx.expand(expand_t))
            pi1_c1 = torch.gather(pi1, 1, c1_idx.expand(expand_t))
            pi2_c2 = torch.gather(pi2, 1, c2_idx.expand(expand_t))
            triple_prob = pi0_c0 * pi1_c1 * pi2_c2  # (T,ncomb)

            triple_w = triple_prob*insideFactor

            # Barycentric => 3D
            p0_3 = p_tri[:,0,:].unsqueeze(1)
            p1_3 = p_tri[:,1,:].unsqueeze(1)
            p2_3 = p_tri[:,2,:].unsqueeze(1)

            alpha_e = alpha.unsqueeze(-1)
            beta_e  = beta_.unsqueeze(-1)
            gamma_e = gamma_.unsqueeze(-1)
            triple_coords = alpha_e*p0_3 + beta_e*p1_3 + gamma_e*p2_3

            # replicate each triple for the 3 pairs => (c0,c1),(c0,c2),(c1,c2)
            pair0 = pair_idx_mat[combs[:,0], combs[:,1]]
            pair1 = pair_idx_mat[combs[:,0], combs[:,2]]
            pair2 = pair_idx_mat[combs[:,1], combs[:,2]]
            triple_pairs = torch.stack([pair0, pair1, pair2], dim=1)

            coords_flat = triple_coords.view(-1,3)
            w_flat      = triple_w.view(-1)

            coords_rep = coords_flat.repeat_interleave(3, dim=0)
            w_rep      = w_flat.repeat_interleave(3, dim=0)
            triple_pairs_flat = triple_pairs.view(-1)
            triple_pairs_full = triple_pairs_flat.unsqueeze(0).expand(T, -1).reshape(-1)

            # for difference=0 => product=0 => logistic(0)=0.5, or pick any factor
            w_rep_final = w_rep*0.5

            all_coords = torch.cat([all_coords, coords_rep], dim=0)
            all_w      = torch.cat([all_w,      w_rep_final], dim=0)
            all_pidx   = torch.cat([all_pidx,   triple_pairs_full], dim=0)

    # ------------------------------------------------------------------------
    # 4) Weighted covariance => plane fits
    # ------------------------------------------------------------------------
    sum_w   = torch.zeros((P,), device=device, dtype=all_coords.dtype)
    sum_x   = torch.zeros((P,3), device=device, dtype=all_coords.dtype)
    sum_xx  = torch.zeros((P,3,3), device=device, dtype=all_coords.dtype)

    weighted_coords = all_coords*all_w.unsqueeze(-1)
    sum_w.index_add_(0, all_pidx, all_w)
    sum_x.index_add_(0, all_pidx, weighted_coords)

    outer_ = weighted_coords.unsqueeze(2)*all_coords.unsqueeze(1)
    sum_xx_flat = sum_xx.view(P,9)
    outer_flat  = outer_.reshape(-1,9)
    sum_xx_flat.index_add_(0, all_pidx, outer_flat)
    sum_xx = sum_xx_flat.view(P,3,3)

    sum_w_clamped = sum_w.clamp_min(eps)
    mean_ = sum_x/sum_w_clamped.unsqueeze(-1)
    mean_outer = mean_.unsqueeze(2)*mean_.unsqueeze(1)
    cov = sum_xx/sum_w_clamped.view(-1,1,1) - mean_outer

    # ------------------------------------------------------------------------
    # 5) Plane from SVD
    # ------------------------------------------------------------------------
    cov_f32 = cov.float()
    U,S,Vt = torch.linalg.svd(cov_f32, full_matrices=False)  # => (P,3,3)
    plane_n = Vt[:, -1, :].to(cov.dtype)
    plane_n = plane_n/(plane_n.norm(dim=1,keepdim=True)+eps)
    plane_d = -(plane_n*mean_).sum(dim=1)

    # ------------------------------------------------------------------------
    # 6) Second pass => MSE
    # ------------------------------------------------------------------------
    sum_sq = torch.zeros((P,), device=device, dtype=all_coords.dtype)
    n_idx = plane_n[all_pidx]
    d_idx = plane_d[all_pidx]
    dist  = (n_idx*all_coords).sum(dim=1)+d_idx
    dist_sq = dist.square()*all_w
    sum_sq.index_add_(0, all_pidx, dist_sq)

    mse_pairs  = sum_sq/(sum_w_clamped+eps)
    total_loss = mse_pairs.sum()
    return total_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict
import math
import time
import pyvista as pv
###############################################################################
# 1) Generate an icosphere mesh with vectorized operations
###############################################################################

def create_icosphere_mesh(target_points=None, subdivisions=2, radius=1.0):
    """
    Creates an icosahedron-based sphere mesh by subdividing faces.
    Vectorized where possible.
    
    Args:
        target_points: Desired number of vertices (approximate) - if provided, 
                      overrides the subdivisions parameter
        subdivisions: Number of times to subdivide the initial icosahedron
        radius: Radius of the final sphere
        
    Returns:
        vertices_np: Array of shape (N, 3) containing vertex positions
        faces_np: Array of shape (T, 3) containing triangle indices
    """
    # If target_points is provided, calculate appropriate subdivision level
    if target_points is not None:
        # Initial icosahedron has 12 vertices
        # Each subdivision approximately quadruples the number of vertices
        # So we can estimate: num_vertices ≈ 12 * 4^subdivisions
        
        # Solve for subdivisions: log_4(target_points/12)
        if target_points < 12:
            print(f"Warning: Target points {target_points} is less than minimum (12). Using 0 subdivisions.")
            subdivisions = 0
        else:
            # Calculate subdivision level needed to reach target_points
            subdivisions = int(np.log(target_points / 12) / np.log(4))
            
            # Calculate actual number of points with this subdivision
            estimated_points = 12 * (4 ** subdivisions)
            print(f"Using {subdivisions} subdivisions to create approximately {estimated_points} points " +
                  f"(target was {target_points}).")
    # Start with icosahedron vertices
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base_vertices = np.array([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1],
    ], dtype=np.float32)
    
    # Normalize vertices to lie on a unit sphere (vectorized)
    norms = np.linalg.norm(base_vertices, axis=1, keepdims=True)
    base_vertices /= norms
    
    # Initial faces of the icosahedron
    base_faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ], dtype=np.int32)
    
    # Subdivide the mesh
    vertices = base_vertices
    faces = base_faces
    
    for _ in range(subdivisions):
        vertices, faces = subdivide_mesh(vertices, faces)
    
    # Scale by radius (vectorized)
    vertices *= radius
    
    return vertices, faces

def subdivide_mesh(vertices, faces):
    """
    Subdivide each triangle into four smaller triangles.
    Optimized version with more efficient data structures.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex positions
        faces: Array of shape (T, 3) containing triangle indices
        
    Returns:
        new_vertices: Array of shape (N', 3) containing new vertex positions
        new_faces: Array of shape (4T, 3) containing new triangle indices
    """
    vertices = list(vertices)
    new_faces = []
    edge_to_mid = {}
    
    # Extract all edges from faces for batch processing
    all_edges = []
    for face in faces:
        v1, v2, v3 = face
        all_edges.extend([(min(v1, v2), max(v1, v2)), 
                         (min(v2, v3), max(v2, v3)), 
                         (min(v3, v1), max(v3, v1))])
    
    # Find unique edges
    unique_edges = list(set(all_edges))
    
    # Compute midpoints in one batch
    for edge in unique_edges:
        a, b = edge
        mid = 0.5 * (np.array(vertices[a]) + np.array(vertices[b]))
        mid = mid / np.linalg.norm(mid)
        edge_to_mid[edge] = len(vertices)
        vertices.append(mid)
    
    # Create new faces
    for face in faces:
        v1, v2, v3 = face
        e1 = (min(v1, v2), max(v1, v2))
        e2 = (min(v2, v3), max(v2, v3))
        e3 = (min(v3, v1), max(v3, v1))
        
        a = edge_to_mid[e1]
        b = edge_to_mid[e2]
        c = edge_to_mid[e3]
        
        new_faces.extend([
            [v1, a, c],
            [v2, b, a],
            [v3, c, b],
            [a, b, c]
        ])
    
    return np.array(vertices), np.array(new_faces, dtype=np.int32)



def load_volume_tet_mesh_and_extract_surface(file_path):
    """
    Loads a VTK (or VTU) file containing a volumetric tetrahedral mesh,
    extracts its boundary surface, and returns a (vertices, faces) pair
    with all boundary faces triangulated.

    Args:
        file_path (str): Path to the VTK/VTU file.

    Returns:
        vertices_np (np.ndarray): Array of shape (N, 3) containing surface vertex coordinates.
        faces_np (np.ndarray): Array of shape (F, 3) containing triangulated surface faces (vertex indices).
    """
    # 1) Read mesh from file
    mesh = pv.read(file_path)  # PyVista automatically guesses file type (VTK, VTU, etc.)

    # 2) Extract the boundary surface
    surface_mesh = mesh.extract_surface()

    # 3) Triangulate (ensures only triangular cells)
    surface_mesh = surface_mesh.triangulate()

    # surface_mesh.faces is a "face array" of the form [3, i0, i1, i2, 3, i0, i1, i2, ...]
    # which we can reshape into a matrix of shape (num_faces, 4), and drop the first column (the "3")
    faces_array = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # shape: (num_faces, 3)

    # Extract points
    vertices_np = surface_mesh.points  # shape: (N, 3)

    return vertices_np, faces_array
###############################################################################
# 2) Mesh data structures with vectorized operations
###############################################################################

def compute_face_areas(vertices_np, faces_np):
    """
    Compute the area of each face in the mesh using vectorized operations.
    
    Args:
        vertices_np: Array of shape (N, 3) containing vertex positions
        faces_np: Array of shape (T, 3) containing triangle indices
        
    Returns:
        areas: Array of shape (T,) containing face areas
    """
    # Get all vertices of all triangles at once
    v0 = vertices_np[faces_np[:, 0]]  # (T, 3)
    v1 = vertices_np[faces_np[:, 1]]  # (T, 3)
    v2 = vertices_np[faces_np[:, 2]]  # (T, 3)
    
    # Compute edge vectors
    e1 = v1 - v0  # (T, 3)
    e2 = v2 - v0  # (T, 3)
    
    # Compute areas using cross product
    cross = np.cross(e1, e2)  # (T, 3)
    areas = 0.5 * np.linalg.norm(cross, axis=1)  # (T,)
    
    return areas

def build_triangle_adjacency(faces_np):
    """
    Find pairs of triangles that share an edge.
    More optimized implementation with vectorized operations where possible.
    
    Args:
        faces_np: Array of shape (T, 3) containing triangle indices
        
    Returns:
        adjacency: Array of shape (E, 2) containing pairs of adjacent triangle indices
    """
    # Use numpy operations to create edges
    T = faces_np.shape[0]
    
    # Create an array of all edges (3 edges per triangle)
    all_edges = np.zeros((T * 3, 3), dtype=np.int64)
    
    # For each triangle, create its 3 edges as (min_idx, max_idx, tri_idx)
    for t_idx in range(T):
        tri = faces_np[t_idx]
        i1, i2, i3 = tri
        
        # Sort vertex indices for each edge
        edges = np.array([
            [min(i1, i2), max(i1, i2), t_idx],
            [min(i2, i3), max(i2, i3), t_idx],
            [min(i3, i1), max(i3, i1), t_idx]
        ])
        
        all_edges[t_idx*3:t_idx*3+3] = edges
    
    # Sort by edge (first by min_idx, then by max_idx)
    sorted_edges = all_edges[np.lexsort((all_edges[:, 1], all_edges[:, 0]))]
    
    # Find edges that appear exactly twice (shared by 2 triangles)
    adjacency = []
    i = 0
    while i < len(sorted_edges) - 1:
        if (sorted_edges[i, 0] == sorted_edges[i+1, 0] and 
            sorted_edges[i, 1] == sorted_edges[i+1, 1]):
            # Found a shared edge
            t1 = sorted_edges[i, 2]
            t2 = sorted_edges[i+1, 2]
            adjacency.append((min(t1, t2), max(t1, t2)))
            i += 2
        else:
            i += 1
    
    return np.array(adjacency, dtype=np.int64)

def build_vertex_edges(faces_np):
    """
    Find all unique edges in the mesh with vectorized operations.
    
    Args:
        faces_np: Array of shape (T, 3) containing triangle indices
        
    Returns:
        edges: Array of shape (E, 2) containing vertex edge indices
    """
    # Extract all edges from triangles
    T = faces_np.shape[0]
    all_edges = np.zeros((T * 3, 2), dtype=np.int64)
    
    # For each triangle, extract sorted edges
    for t_idx in range(T):
        i1, i2, i3 = faces_np[t_idx]
        
        # Sort vertex indices for each edge
        edges = np.array([
            [min(i1, i2), max(i1, i2)],
            [min(i2, i3), max(i2, i3)],
            [min(i3, i1), max(i3, i1)]
        ])
        
        all_edges[t_idx*3:t_idx*3+3] = edges
    
    # Use numpy's unique function on structured arrays to find unique edges
    dtype = [('v1', np.int64), ('v2', np.int64)]
    structured_edges = np.array([(e[0], e[1]) for e in all_edges], dtype=dtype)
    unique_edges = np.unique(structured_edges)
    
    # Convert back to regular array
    edges = np.array([(e[0], e[1]) for e in unique_edges], dtype=np.int64)
    
    return edges

###############################################################################
# 3) Choose pinned vertices for 6 regions
###############################################################################

def find_axis_vertices(vertices_np):
    """
    Find vertices to pin for 6 regions using vectorized operations.
    
    Args:
        vertices_np: Array of shape (N, 3) containing vertex positions
        
    Returns:
        pinned_indices: List of 6 vertex indices
    """
    # Find max/min along each axis in one operation
    max_indices = np.argmax(vertices_np, axis=0)  # [x_max_idx, y_max_idx, z_max_idx]
    min_indices = np.argmin(vertices_np, axis=0)  # [x_min_idx, y_min_idx, z_min_idx]
    
    # Order as [top, bottom, front, back, right, left]
    pinned_indices = [
        max_indices[2],  # top (max z)
        min_indices[2],  # bottom (min z)
        max_indices[1],  # front (max y)
        min_indices[1],  # back (min y)
        max_indices[0],  # right (max x)
        min_indices[0]   # left (min x)
    ]
    
    return pinned_indices

###############################################################################
# 4) Initialize 6-channel scalar field with pinned vertices
###############################################################################

def init_6channels_with_pins(num_vertices, pinned_indices, device):
    """
    Initialize a 6-channel scalar field with pinned vertices.
    Vectorized implementation.
    
    Args:
        num_vertices: Number of vertices in the mesh
        pinned_indices: List of 6 vertex indices to pin
        device: PyTorch device
        
    Returns:
        f_param: PyTorch parameter of shape (N, 6)
    """
    # Initialize with small random values
    f_init = 0.01 * np.random.randn(num_vertices, 6).astype(np.float32)
    
    # Create pin mask tensor
    pin_mask = np.ones((len(pinned_indices), 6)) * -1.0
    np.fill_diagonal(pin_mask, 1.0)
    
    # Set pinned values: channel c=+1, all others=-1
    for i, v_idx in enumerate(pinned_indices):
        f_init[v_idx] = pin_mask[i]
    
    # Convert to PyTorch parameter
    f_param = nn.Parameter(torch.tensor(f_init, device=device))
    
    return f_param

###############################################################################
# 5) Highly vectorized loss functions for the 6-channel field
###############################################################################import torch
import torch

def contour_alignment_loss(
    vertices:        torch.Tensor,  # (N,3) float
    faces:           torch.Tensor,  # (T,3) long
    f_values:        torch.Tensor,  # (N,C) float
    beta:            float = 20.0,
    eps:             float = 1e-9,
    soft_inside:     float = 10.0,
    include_triples: bool = True
) -> torch.Tensor:
    """
    A fully vectorized, "vigorous" contour-alignment loss that:
      1) Collects edge intersections for every channel pair (i<j) using a logistic weight
         for sign-changes in (f_i - f_j).
      2) Optionally finds "soft triple intersections" for (c0,c1,c2) if C>=3:
         - We do a closed-form barycentric solve for (f_c0-f_c1)=0, (f_c0-f_c2)=0
         - We multiply by each corner's softmax-prob of channels c0,c1,c2,
           plus a smooth clamp to keep the barycentric coords in [0,1].
      3) Builds a weighted covariance for each pair in one pass. Does an SVD => plane normal.
      4) In a second pass, accumulates the MSE distances of all intersection points
         to that pair's plane. Summation => final scalar loss.

    *No loops* over triangles or pairs in Python. Everything is broadcast/batch in PyTorch.
    *Fully differentiable*—no discrete argmax or masks. If C<2 => returns 0.0 (no pairs).

    Args:
      vertices: (N,3) float
      faces: (T,3) long
      f_values: (N,C) float   # multi-channel field
      beta: float  # logistic sharpness for edge intersection
      eps: float
      soft_inside: float  # how sharply to clamp barycentric coords for triple intersection
      include_triples: bool

    Returns:
      total_loss: scalar float
    """
    device = vertices.device
    N = vertices.shape[0]
    T = faces.shape[0]
    C = f_values.shape[1]

    # 0) If <2 channels => no pairs => return 0.0
    if C < 2:
        return torch.zeros((), device=device, dtype=vertices.dtype)

    # ------------------------------------------------------------------------
    # 1) Build all channel pairs i<j => shape (P,)
    # ------------------------------------------------------------------------
    i2, j2 = torch.triu_indices(C, C, offset=1, device=device)  # each => (P,)
    if i2.numel() == 0:
        # Edge case if somehow no pairs. Return 0.
        return torch.zeros((), device=device, dtype=vertices.dtype)

    P = i2.shape[0]  # number of (i<j) pairs

    # Build a map (cA,cB) => pair index
    pair_idx_mat = torch.full((C,C), -1, device=device, dtype=torch.long)
    p_arange     = torch.arange(P, device=device)
    pair_idx_mat[i2, j2] = p_arange
    pair_idx_mat[j2, i2] = p_arange

    # ------------------------------------------------------------------------
    # 2) Edge Intersections => logistic weighting
    # ------------------------------------------------------------------------
    # faces => (T,3), so gather coords => (T,3,3)
    p_tri = vertices[faces]
    # gather field => (T,3,C)
    f_tri = f_values[faces]

    # d => shape (T,3,P).  d[t,v,p] = f_tri[t,v, i2[p]] - f_tri[t,v, j2[p]]
    d = f_tri[..., i2] - f_tri[..., j2]  # (T,3,P)

    p0 = p_tri[:,0]
    p1 = p_tri[:,1]
    p2 = p_tri[:,2]

    def edge_intersection(dA, dB, vA, vB):
        """
        dA,dB: (T,P)
        vA,vB: (T,3)
        return coords: (T,P,3), w: (T,P)
        """
        prod  = dA*dB
        w_    = torch.sigmoid(-beta*prod)  # logistic weight
        alpha = torch.abs(dA)/(torch.abs(dA)+torch.abs(dB)+eps)
        coords = vA.unsqueeze(1) + alpha.unsqueeze(-1)*(vB - vA).unsqueeze(1)
        return coords, w_

    d0, d1, d2 = d[:,0,:], d[:,1,:], d[:,2,:]
    coords_01, w_01 = edge_intersection(d0, d1, p0, p1)
    coords_12, w_12 = edge_intersection(d1, d2, p1, p2)
    coords_20, w_20 = edge_intersection(d2, d0, p2, p0)

    def flatten_edge(coords_tp3, w_tp):
        # coords_tp3 => (T,P,3), w_tp => (T,P)
        coords_flat = coords_tp3.reshape(-1,3)
        w_flat      = w_tp.reshape(-1)
        pair_idx_arange = torch.arange(P, device=device).view(1,P).expand(coords_tp3.shape[0],-1).reshape(-1)
        return coords_flat, w_flat, pair_idx_arange

    coords_01f, w_01f, pidx_01f = flatten_edge(coords_01, w_01)
    coords_12f, w_12f, pidx_12f = flatten_edge(coords_12, w_12)
    coords_20f, w_20f, pidx_20f = flatten_edge(coords_20, w_20)

    all_coords = torch.cat([coords_01f, coords_12f, coords_20f], dim=0)
    all_w      = torch.cat([w_01f,      w_12f,      w_20f],      dim=0)
    all_pidx   = torch.cat([pidx_01f,   pidx_12f,   pidx_20f],   dim=0)

    # ------------------------------------------------------------------------
    # 3) Soft triple intersection => only if C>=3 and include_triples=True
    # ------------------------------------------------------------------------
    if include_triples and C >= 3:
        f0 = f_tri[:,0,:]  # (T,C)
        f1 = f_tri[:,1,:]
        f2 = f_tri[:,2,:]
        # softmax => each corner has distribution over channels
        pi0 = torch.softmax(f0, dim=1)
        pi1 = torch.softmax(f1, dim=1)
        pi2 = torch.softmax(f2, dim=1)

        # All combos c0< c1< c2
        combs = torch.combinations(torch.arange(C, device=device), r=3)
        # e.g. shape => (#comb,3)
        if combs.numel() > 0:  # ensure there's something
            ncomb = combs.shape[0]
            c0_idx = combs[:,0].view(1,ncomb)
            c1_idx = combs[:,1].view(1,ncomb)
            c2_idx = combs[:,2].view(1,ncomb)

            expand_t = (T, ncomb)
            f0_c0 = torch.gather(f0, 1, c0_idx.expand(expand_t))
            f0_c1 = torch.gather(f0, 1, c1_idx.expand(expand_t))
            f0_c2 = torch.gather(f0, 1, c2_idx.expand(expand_t))
            f1_c0 = torch.gather(f1, 1, c0_idx.expand(expand_t))
            f1_c1 = torch.gather(f1, 1, c1_idx.expand(expand_t))
            f1_c2 = torch.gather(f1, 1, c2_idx.expand(expand_t))
            f2_c0 = torch.gather(f2, 1, c0_idx.expand(expand_t))
            f2_c1 = torch.gather(f2, 1, c1_idx.expand(expand_t))
            f2_c2 = torch.gather(f2, 1, c2_idx.expand(expand_t))

            # differences
            rg0 = f0_c0 - f0_c1
            rg1 = f1_c0 - f1_c1
            rg2 = f2_c0 - f2_c1
            rb0 = f0_c0 - f0_c2
            rb1 = f1_c0 - f1_c2
            rb2 = f2_c0 - f2_c2

            A_xy = rg0 - rg2
            B_xy = rg1 - rg2
            X_xy = rg2
            A_xz = rb0 - rb2
            B_xz = rb1 - rb2
            X_xz = rb2

            det = A_xy*B_xz - B_xy*A_xz
            alpha = (X_xy*B_xz - B_xy*X_xz)/(det+eps)
            beta_  = (A_xy*X_xz - X_xy*A_xz)/(det+eps)
            gamma_ = 1.0 - alpha - beta_

            # "soft inside" factor
            if soft_inside>0.0:
                def smoothstep(x):
                    return torch.sigmoid(soft_inside*x)
                insideFactor = smoothstep(alpha)*smoothstep(beta_)*smoothstep(gamma_)
            else:
                insideFactor = torch.ones_like(alpha)

            # Probability corner0 picks c0, corner1 picks c1, corner2 picks c2
            pi0_c0 = torch.gather(pi0, 1, c0_idx.expand(expand_t))
            pi1_c1 = torch.gather(pi1, 1, c1_idx.expand(expand_t))
            pi2_c2 = torch.gather(pi2, 1, c2_idx.expand(expand_t))
            triple_prob = pi0_c0 * pi1_c1 * pi2_c2  # (T,ncomb)

            triple_w = triple_prob*insideFactor

            # Barycentric => 3D
            p0_3 = p_tri[:,0,:].unsqueeze(1)
            p1_3 = p_tri[:,1,:].unsqueeze(1)
            p2_3 = p_tri[:,2,:].unsqueeze(1)

            alpha_e = alpha.unsqueeze(-1)
            beta_e  = beta_.unsqueeze(-1)
            gamma_e = gamma_.unsqueeze(-1)
            triple_coords = alpha_e*p0_3 + beta_e*p1_3 + gamma_e*p2_3

            # replicate each triple for the 3 pairs => (c0,c1),(c0,c2),(c1,c2)
            pair0 = pair_idx_mat[combs[:,0], combs[:,1]]
            pair1 = pair_idx_mat[combs[:,0], combs[:,2]]
            pair2 = pair_idx_mat[combs[:,1], combs[:,2]]
            triple_pairs = torch.stack([pair0, pair1, pair2], dim=1)

            coords_flat = triple_coords.view(-1,3)
            w_flat      = triple_w.view(-1)

            coords_rep = coords_flat.repeat_interleave(3, dim=0)
            w_rep      = w_flat.repeat_interleave(3, dim=0)
            triple_pairs_flat = triple_pairs.view(-1)
            triple_pairs_full = triple_pairs_flat.unsqueeze(0).expand(T, -1).reshape(-1)

            # for difference=0 => product=0 => logistic(0)=0.5, or pick any factor
            w_rep_final = w_rep*0.5

            all_coords = torch.cat([all_coords, coords_rep], dim=0)
            all_w      = torch.cat([all_w,      w_rep_final], dim=0)
            all_pidx   = torch.cat([all_pidx,   triple_pairs_full], dim=0)

    # ------------------------------------------------------------------------
    # 4) Weighted covariance => plane fits
    # ------------------------------------------------------------------------
    sum_w   = torch.zeros((P,), device=device, dtype=all_coords.dtype)
    sum_x   = torch.zeros((P,3), device=device, dtype=all_coords.dtype)
    sum_xx  = torch.zeros((P,3,3), device=device, dtype=all_coords.dtype)

    weighted_coords = all_coords*all_w.unsqueeze(-1)
    sum_w.index_add_(0, all_pidx, all_w)
    sum_x.index_add_(0, all_pidx, weighted_coords)

    outer_ = weighted_coords.unsqueeze(2)*all_coords.unsqueeze(1)
    sum_xx_flat = sum_xx.view(P,9)
    outer_flat  = outer_.reshape(-1,9)
    sum_xx_flat.index_add_(0, all_pidx, outer_flat)
    sum_xx = sum_xx_flat.view(P,3,3)

    sum_w_clamped = sum_w.clamp_min(eps)
    mean_ = sum_x/sum_w_clamped.unsqueeze(-1)
    mean_outer = mean_.unsqueeze(2)*mean_.unsqueeze(1)
    cov = sum_xx/sum_w_clamped.view(-1,1,1) - mean_outer

    # ------------------------------------------------------------------------
    # 5) Plane from SVD
    # ------------------------------------------------------------------------
    cov_f32 = cov.float()
    U,S,Vt = torch.linalg.svd(cov_f32, full_matrices=False)  # => (P,3,3)
    plane_n = Vt[:, -1, :].to(cov.dtype)
    plane_n = plane_n/(plane_n.norm(dim=1,keepdim=True)+eps)
    plane_d = -(plane_n*mean_).sum(dim=1)

    # ------------------------------------------------------------------------
    # 6) Second pass => MSE
    # ------------------------------------------------------------------------
    sum_sq = torch.zeros((P,), device=device, dtype=all_coords.dtype)
    n_idx = plane_n[all_pidx]
    d_idx = plane_d[all_pidx]
    dist  = (n_idx*all_coords).sum(dim=1)+d_idx
    dist_sq = dist.square()*all_w
    sum_sq.index_add_(0, all_pidx, dist_sq)

    mse_pairs  = sum_sq/(sum_w_clamped+eps)
    total_loss = mse_pairs.sum()
    return total_loss




def contour_alignment_loss_6channels(points, triangles, f_values, adjacency, beta=20.0):
    """
    Compute a differentiable contour alignment loss that avoids the "stuck at constant value" problem.
    This is a wrapper around contour_alignment_loss_6channels_label_subdivide.
    
    Args:
        points: Tensor of shape (N, 3) containing vertex positions
        triangles: Tensor of shape (T, 3) containing triangle vertex indices
        f_values: Tensor of shape (N, 6) containing scalar field values
        adjacency: Tensor of shape (E, 2) containing adjacent triangle pairs
        beta: Temperature parameter for softmax (unused in the subdivide version)
        
    Returns:
        loss: Scalar loss for contour alignment
    """
    # Call the implementation with appropriate parameters
    return contour_alignment_loss(
        points=points, 
        triangles=triangles, 
        f_values=f_values, 
        adjacency=adjacency,
        check_triple=True
    )

def area_balance_loss_optimized(points, triangles, f_values, beta, mesh_area):
    """
    Highly vectorized implementation of area balance loss using softmax instead of sigmoid.
    
    Args:
        points: Tensor of shape (N, 3) containing vertex positions
        triangles: Tensor of shape (T, 3) containing triangle vertex indices
        f_values: Tensor of shape (N, 6) containing scalar field values
        beta: Softmax sharpness parameter
        mesh_area: Total mesh area
        
    Returns:
        loss: Area balance loss value
        fractions: Tensor of shape (6,) with fraction of area for each channel
    """
    device = points.device
    num_channels = f_values.shape[1]
    num_triangles = triangles.shape[0]
    
    # Define barycentric sampling points
    bary_points = torch.tensor([
        [1/3, 1/3, 1/3],  # Center
        [0.5, 0.5, 0.0],  # Edge midpoints
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ], device=device)
    
    num_samples = bary_points.shape[0]
    bary_weights = torch.ones(num_samples, device=device) / num_samples
    
    # Compute triangle areas in one batch
    v0 = triangles[:, 0]  # (T,)
    v1 = triangles[:, 1]  # (T,)
    v2 = triangles[:, 2]  # (T,)
    
    p0 = points[v0]  # (T, 3)
    p1 = points[v1]  # (T, 3)
    p2 = points[v2]  # (T, 3)
    
    e1 = p1 - p0  # (T, 3)
    e2 = p2 - p0  # (T, 3)
    
    normals = torch.cross(e1, e2, dim=1)  # (T, 3)
    areas = 0.5 * torch.norm(normals, dim=1)  # (T,)
    
    # Convert to softmax probabilities
    raw_scores = beta * f_values  # (N, 6)
    
    # Get scalar field values at triangle vertices
    f0 = f_values[v0]  # (T, 6)
    f1 = f_values[v1]  # (T, 6)
    f2 = f_values[v2]  # (T, 6)
    
    # Initialize tensor to accumulate channel areas
    channel_areas = torch.zeros(num_channels, device=device)
    
    # Process all triangles and all barycentric samples at once
    # Reshape for broadcasting:
    # f0: (T, 6) -> (T, 1, 6)
    # f1: (T, 6) -> (T, 1, 6)
    # f2: (T, 6) -> (T, 1, 6)
    # bary_points: (S, 3) -> (1, S, 3)
    f0 = f0.unsqueeze(1)  # (T, 1, 6)
    f1 = f1.unsqueeze(1)  # (T, 1, 6)
    f2 = f2.unsqueeze(1)  # (T, 1, 6)
    
    bary_expanded = bary_points.unsqueeze(0)  # (1, S, 3)
    
    # Extract barycentric coordinates
    b0 = bary_expanded[:, :, 0].unsqueeze(2)  # (1, S, 1)
    b1 = bary_expanded[:, :, 1].unsqueeze(2)  # (1, S, 1)
    b2 = bary_expanded[:, :, 2].unsqueeze(2)  # (1, S, 1)
    
    # Interpolate field at all barycentric points for all triangles
    f_interp = (f0 * b0) + (f1 * b1) + (f2 * b2)  # (T, S, 6)
    
    # Apply softmax to all interpolated values
    f_interp_beta = beta * f_interp  # (T, S, 6)
    exp_vals = torch.exp(f_interp_beta)  # (T, S, 6)
    sum_exp = torch.sum(exp_vals, dim=2, keepdim=True)  # (T, S, 1)
    s = torch.softmax(f_interp_beta, dim=2)
    
    # Apply sample weights and reshape
    s_weighted = s * bary_weights.view(1, num_samples, 1)  # (T, S, 6)
    
    # Sum over samples
    s_mean = torch.sum(s_weighted, dim=1)  # (T, 6)
    
    # Weight by triangle areas and sum
    weighted_areas = s_mean * areas.unsqueeze(1)  # (T, 6)
    channel_areas = torch.sum(weighted_areas, dim=0)  # (6,)
    
    # Compute fractions and loss
    fractions = channel_areas / mesh_area
    target = torch.ones_like(fractions) / num_channels
    loss = torch.sum(torch.abs(fractions - target))
    
    return loss, fractions

def smoothness_loss_optimized(f_values, vertex_edges):
    """
    Vectorized implementation of smoothness loss.
    
    Args:
        f_values: Tensor of shape (N, 6) containing scalar field values
        vertex_edges: Tensor of shape (E, 2) containing vertex edge indices
        
    Returns:
        loss: Smoothness loss value
    """
    # Get field values at all edge endpoints at once
    v1_idx = vertex_edges[:, 0]  # (E,)
    v2_idx = vertex_edges[:, 1]  # (E,)
    
    f1 = f_values[v1_idx]  # (E, 6)
    f2 = f_values[v2_idx]  # (E, 6)
    
    # Compute squared difference for all edges and all channels
    diff = f1 - f2  # (E, 6)
    loss = torch.sum(diff**2)
    
    return loss

def compute_combined_loss_optimized(f_values, points, triangles, adjacency, vertex_edges,
                                  mesh_area, beta=20.0, lambda_contour=1.0, lambda_smooth=1.0,
                                  lambda_area=1.0, use_label_subdivide=True):
    """
    Compute the combined loss with optimized vectorized operations.
    Now using contour_alignment_loss_6channels_label_subdivide for improved boundary detection.
    
    Args:
        f_values: Tensor of shape (N, 6) containing scalar field values
        points: Tensor of shape (N, 3) containing vertex positions
        triangles: Tensor of shape (T, 3) containing triangle vertex indices
        adjacency: Tensor of shape (E, 2) containing adjacent triangle pairs
        vertex_edges: Tensor of shape (E', 2) containing vertex edge indices
        mesh_area: Total mesh area
        beta: Softmax temperature parameter
        lambda_contour: Weight for contour alignment loss
        lambda_smooth: Weight for smoothness loss
        lambda_area: Weight for area balance loss
        use_label_subdivide: If True, use the label subdivision approach, otherwise use the simpler diff approach
        
    Returns:
        total_loss: Combined loss value
        loss_dict: Dictionary containing individual loss components
    """
    # Compute individual losses
    if use_label_subdivide:
        # Use the improved label subdivision approach for contour alignment
        contour_loss = contour_alignment_loss(
            vertices=points,
            faces=triangles,
            f_values=f_values,
            include_triples=True,
            beta=beta,
            eps=1e-9
            )
    else:
        # Use the simpler probability difference approach
        contour_loss = contour_alignment_loss_6channels(points, triangles, f_values, adjacency, beta)
        
    smooth_loss = smoothness_loss_optimized(f_values, vertex_edges)
    area_loss, area_fracs = area_balance_loss_optimized(points, triangles, f_values, beta, mesh_area)
    
    # Combine losses (note: no overlap loss since softmax handles that)
    total_loss = (lambda_contour * contour_loss +
                 lambda_smooth * smooth_loss +
                 lambda_area * area_loss)
    
    # Store individual losses for monitoring
    loss_dict = {
        'contour': contour_loss.item(),
        'smoothness': smooth_loss.item(),
        'area_balance': area_loss.item(),
        'total': total_loss.item(),
        'area_fractions': area_fracs.detach().cpu().numpy()
    }
    
    return total_loss, loss_dict

def visualize_segmentation_hardmax(vertices_np, faces_np, f_values, vis_resolution=30):
    """
    Visualize the segmentation using true hardmax boundaries with barycentric interpolation.
    Each pixel in a triangle is colored based on the argmax of interpolated values,
    allowing for sharp boundaries that cut through triangles.
    
    Args:
        vertices_np: Array of shape (N, 3) containing vertex positions
        faces_np: Array of shape (T, 3) containing triangle indices
        f_values: Array of shape (N, 6) containing scalar field values  
        vis_resolution: Resolution for the visualization sphere
    """
    try:
        import pyvista as pv
        import numpy as np
        import time
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        
        print("Creating hardmax visualization with PyVista...")
        start_time = time.time()
        
        # Set up PyVista theme
        pv.set_plot_theme("document")
        
        # Define region colors
        region_colors = np.array([
            [1.0, 0.0, 0.0, 1.0],  # Red - Region 1
            [0.0, 0.0, 1.0, 1.0],  # Blue - Region 2
            [0.0, 1.0, 0.0, 1.0],  # Green - Region 3
            [1.0, 1.0, 0.0, 1.0],  # Yellow - Region 4
            [1.0, 0.0, 1.0, 1.0],  # Magenta - Region 5
            [0.0, 1.0, 1.0, 1.0]   # Cyan - Region 6
        ])
        
        # Create a proper matplotlib colormap
        region_cmap = ListedColormap(region_colors)
        
        # Create high-resolution sphere for barycentric rendering
        print(f"Creating high-resolution sphere with {vis_resolution*2} resolution...")
        high_res_sphere = pv.Sphere(radius=1.0, 
                                   theta_resolution=vis_resolution*2, 
                                   phi_resolution=vis_resolution*2)
        
        # Get the low-res sphere points (original mesh)
        # Create a triangular mesh using the original vertices and faces
        low_res_mesh = pv.PolyData(vertices_np, 
                                  np.column_stack((np.full(len(faces_np), 3), faces_np)).flatten())
        
        # For interpolation, we need to find which triangle each high-res point belongs to
        # and compute its barycentric coordinates
        print("Finding closest cells for all high-res points...")
        
        # Find closest cell (triangle) for each high-res point
        # This returns the cell index directly as a NumPy array
        triangle_indices = low_res_mesh.find_closest_cell(high_res_sphere.points)
        
        # Initialize arrays to store interpolated values and labels
        interpolated_labels = np.zeros(len(high_res_sphere.points), dtype=np.int32)
        
        print("Interpolating values and applying hardmax...")
        # For each high-res point, interpolate using barycentric coordinates
        for i, point in enumerate(high_res_sphere.points):
            if i % 10000 == 0:
                print(f"Processing point {i} of {len(high_res_sphere.points)}...")
                
            # Get the triangle this point maps to
            tri_idx = triangle_indices[i]
            
            # Get the triangle vertices
            triangle = faces_np[tri_idx]
            v0, v1, v2 = triangle
            
            # Get vertex positions
            p0 = vertices_np[v0]
            p1 = vertices_np[v1]
            p2 = vertices_np[v2]
            
            # Compute barycentric coordinates
            # Method: solve the linear system:
            # point = b0*p0 + b1*p1 + b2*p2
            # with b0 + b1 + b2 = 1
            
            # Create vectors
            v0v1 = p1 - p0
            v0v2 = p2 - p0
            v0p = point - p0
            
            # Create matrix
            d00 = np.dot(v0v1, v0v1)
            d01 = np.dot(v0v1, v0v2)
            d11 = np.dot(v0v2, v0v2)
            d20 = np.dot(v0p, v0v1)
            d21 = np.dot(v0p, v0v2)
            
            # Compute barycentric coordinates
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-10:
                # Degenerate triangle, just use closest vertex
                dist0 = np.sum((point - p0)**2)
                dist1 = np.sum((point - p1)**2)
                dist2 = np.sum((point - p2)**2)
                
                if dist0 <= dist1 and dist0 <= dist2:
                    interpolated_labels[i] = np.argmax(f_values[v0])
                elif dist1 <= dist0 and dist1 <= dist2:
                    interpolated_labels[i] = np.argmax(f_values[v1])
                else:
                    interpolated_labels[i] = np.argmax(f_values[v2])
                continue
                
            b1 = (d11 * d20 - d01 * d21) / denom
            b2 = (d00 * d21 - d01 * d20) / denom
            b0 = 1.0 - b1 - b2
            
            # Clamp barycentric coordinates (if point is slightly outside triangle)
            b0 = max(0.0, min(1.0, b0))
            b1 = max(0.0, min(1.0, b1))
            b2 = max(0.0, min(1.0, b2))
            
            # Normalize to ensure they sum to 1
            total = b0 + b1 + b2
            if total > 0:
                b0 /= total
                b1 /= total
                b2 /= total
            
            # Get the field values at the triangle vertices
            f0 = f_values[v0]
            f1 = f_values[v1]
            f2 = f_values[v2]
            
            # Interpolate the field values using barycentric coordinates
            f_interp = b0 * f0 + b1 * f1 + b2 * f2
            
            # Apply hardmax (argmax) to get the dominant label
            dominant_label = np.argmax(f_interp)
            interpolated_labels[i] = dominant_label
        
        # Map to 1-indexed for plotting
        label_field = interpolated_labels + 1
        high_res_sphere.point_data["Labels"] = label_field
        
        print("Creating visualization...")
        # Create a plotter
        plotter = pv.Plotter(window_size=[1200, 1200])
        
        # Add title
        plotter.add_text("Hardmax Segmentation with Barycentric Interpolation", 
                       font_size=14, position='upper_edge')
        
        # Add the high-res sphere with hardmax labels
        plotter.add_mesh(
            high_res_sphere,
            scalars="Labels",
            show_edges=False,
            cmap=region_cmap,
            interpolate_before_map=False,  # No interpolation for crisp boundaries
            show_scalar_bar=True,
            clim=[1, 6]
        )
        
        # Add scalar bar
        plotter.add_scalar_bar(
            title="Region Labels (1-6)",
            n_labels=6,
            italic=False,
            fmt="%d",  # Integer format
            font_family="arial",
            shadow=True,
            position_x=0.05,
            position_y=0.05,
            width=0.4
        )
        
        # Find and mark pinned vertices
        max_indices = np.argmax(vertices_np, axis=0)
        min_indices = np.argmin(vertices_np, axis=0)
        pinned_indices = [
            max_indices[2],  # top (max z)
            min_indices[2],  # bottom (min z)
            max_indices[1],  # front (max y)
            min_indices[1],  # back (min y)
            max_indices[0],  # right (max x)
            min_indices[0]   # left (min x)
        ]
        
        # Colors for markers
        marker_colors = [
            [1.0, 0.0, 0.0],  # Red - Top (0)
            [0.0, 0.0, 1.0],  # Blue - Bottom (1)
            [0.0, 1.0, 0.0],  # Green - Front (2)
            [1.0, 1.0, 0.0],  # Yellow - Back (3)
            [1.0, 0.0, 1.0],  # Magenta - Right (4)
            [0.0, 1.0, 1.0]   # Cyan - Left (5)
        ]
        
        # Add annotation markers for each pinned vertex
        region_names = ["Top (1)", "Bottom (2)", "Front (3)", "Back (4)", "Right (5)", "Left (6)"]
        for i, (name, idx) in enumerate(zip(region_names, pinned_indices)):
            # Get vertex position
            pos = vertices_np[idx]
            # Add a point marker at the pinned vertex
            plotter.add_points(pos.reshape(1, 3), color=marker_colors[i], point_size=15)
            # Add text label near the point
            offset = pos * 1.1  # Slightly offset from the point
            plotter.add_point_labels([offset], [name], font_size=10, 
                                   shadow=True, shape=None, text_color=marker_colors[i])
        
        # Set view angle
        plotter.view_isometric()
        plotter.camera.zoom(1.5)
        
        # Save a static screenshot
        try:
            plotter.screenshot('hardmax_segmentation.png', transparent_background=False)
            print("Screenshot saved to hardmax_segmentation.png")
        except Exception as e:
            print(f"Warning: Could not save screenshot: {e}")
            
        print("Displaying interactive visualization. Close the window to continue.")
        plotter.show()
        
        # Also create a direct comparison visualization (original vs hardmax)
        print("Creating comparison visualization...")
        
        # Create original mesh with argmax colors for comparison
        mesh = pv.PolyData(vertices_np, 
                          np.column_stack((np.full(len(faces_np), 3), faces_np)).flatten())
        
        # Apply argmax to get the labels
        labels = np.argmax(f_values, axis=1)
        mesh.point_data["Labels"] = labels + 1  # 1-indexed
        
        # Create a two-panel comparison
        plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
        
        # First panel: Original mesh with vertex labels
        plotter.subplot(0, 0)
        plotter.add_text("Original Mesh (Labels at Vertices)", font_size=14, position="upper_edge")
        
        # Add the original mesh
        plotter.add_mesh(
            mesh,
            scalars="Labels",
            show_edges=True,
            edge_color='black',
            line_width=0.5,
            cmap=region_cmap,
            interpolate_before_map=False,
            show_scalar_bar=False,
            clim=[1, 6]
        )
        
        # Add markers to the first view
        for i, (name, idx) in enumerate(zip(region_names, pinned_indices)):
            pos = vertices_np[idx]
            plotter.add_points(pos.reshape(1, 3), color=marker_colors[i], point_size=15)
            offset = pos * 1.1
            plotter.add_point_labels([offset], [name], font_size=10, 
                                   shadow=True, shape=None, text_color=marker_colors[i])
        
        # Second panel: High-res hardmax visualization
        plotter.subplot(0, 1)
        plotter.add_text("Hardmax with Barycentric Interpolation", font_size=14, position="upper_edge")
        
        # Add the high-res sphere
        plotter.add_mesh(
            high_res_sphere,
            scalars="Labels",
            show_edges=False,
            cmap=region_cmap,
            interpolate_before_map=False,
            show_scalar_bar=True,
            clim=[1, 6]
        )
        
        # Add scalar bar to the second view
        plotter.add_scalar_bar(
            title="Region Labels (1-6)",
            n_labels=6,
            italic=False,
            fmt="%d",
            font_family="arial",
            shadow=True,
            position_x=0.05,
            position_y=0.05,
            width=0.4
        )
        
        # Add markers to the second view too
        for i, (name, idx) in enumerate(zip(region_names, pinned_indices)):
            pos = vertices_np[idx]
            plotter.add_points(pos.reshape(1, 3), color=marker_colors[i], point_size=15)
            offset = pos * 1.1
            plotter.add_point_labels([offset], [name], font_size=10, 
                                   shadow=True, shape=None, text_color=marker_colors[i])
        
        # Link the views
        plotter.link_views()
        plotter.view_isometric()
        
        # Save the comparison
        try:
            plotter.screenshot('hardmax_comparison.png', transparent_background=False)
            print("Comparison saved to hardmax_comparison.png")
        except Exception as e:
            print(f"Warning: Could not save comparison: {e}")
            
        plotter.show()
        
    except ImportError as e:
        print(f"PyVista not available: {e}. Using matplotlib for basic visualization...")
        
        # Fallback to matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Get the discrete labels for the original vertices
        labels = np.argmax(f_values, axis=1)
        
        # Create a 3D figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the mesh points with their labels
        colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan']
        ax.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2],
                  c=[colors[label] for label in labels], s=30)
        
        # Title and axis settings
        ax.set_title('Original Mesh with Vertex Labels')
        ax.set_box_aspect([1, 1, 1])
        
        # Save and show
        plt.savefig('hardmax_basic.png', dpi=300)
        plt.show()
###############################################################################
# 7) Optimized 6-patch segmentation
###############################################################################
from typing import Optional
# --------------------------------------------------------------------------- #
# fast single‑phase optimiser                                                 #
# --------------------------------------------------------------------------- #
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import numpy as np

def optimization(
    vertices_np, faces_np, pinned_indices,
    *,
    n_iters: int = 80_000,           # fewer iters than before
    lr: float = 2e-3,
    beta: float = 1.0,
    target_beta: Optional[float] = 40.0,
    beta_schedule: bool = True,
    lambda_contour_initial: float = 0.0,
    lambda_contour_final: float = 2.0,
    lambda_smooth: float = 0.2,      # slightly higher default
    lambda_area_initial: float = 0.2,
    lambda_area_final: float = 2.0,
    # OneCycleLR parameters:
    pct_start: float = 0.3,
    anneal_strategy: str = 'cos',
    div_factor: float = 25.0,
    # Early stopping:
    enable_early_stopping: bool = True,
    patience: int = 2000,
    print_every: int = 100,
    save_path: str = "final_mesh_and_values.npz"  # Path where we will save the mesh and field values
):
    """
    A faster-converging optimizer using:
      - OneCycleLR (with optional early stopping).
      - Slightly higher smoothness by default.
      - Shorter ramp for contour/area.

    Returns
    -------
    f_final : (N, 6) final field values on CPU
    final_mesh: (N, 3) vertices of the final mesh
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert data to torch
    v  = torch.from_numpy(vertices_np).float().to(device)
    f  = torch.from_numpy(faces_np).long().to(device)

    from collections import defaultdict
    from time import time

    # Build adjacency etc. 
    tri_adj    = torch.from_numpy(build_triangle_adjacency(faces_np)).long().to(device)
    vert_edges = torch.from_numpy(build_vertex_edges(faces_np)).long().to(device)
    mesh_area  = compute_face_areas(vertices_np, faces_np).sum()

    # Initialize the 6-channel field
    f_param = init_6channels_with_pins(len(vertices_np), pinned_indices, device)
    pin_mask = torch.full((6,6), -1.0, device=device)
    torch.diagonal(pin_mask).fill_(1.0)

    # Set up the optimizer (AdamW usually works well)
    opt = optim.AdamW([f_param], lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=n_iters,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,
        final_div_factor=1e+4,
        three_phase=False,
    )

    if beta_schedule and target_beta is not None:
        beta_vals = np.linspace(beta, target_beta, n_iters+1)
    else:
        beta_vals = np.full(n_iters+1, beta)

    ramp_iters = max(int(0.2 * n_iters), 1)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
    history = []
    t0 = time()

    best_loss = float('inf')
    best_iter = 0

    for it in range(1, n_iters+1):
        frac = min(it / ramp_iters, 1.0)

        beta_now = float(beta_vals[it])
        lambda_c_now = lambda_contour_initial + (lambda_contour_final - lambda_contour_initial)*frac
        lambda_a_now = lambda_area_initial + (lambda_area_final - lambda_area_initial)*frac

        lr_now = scheduler.get_last_lr()[0]

        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')): 
            total, comp = compute_combined_loss_optimized(
                f_param, v, f, tri_adj, vert_edges, mesh_area,
                beta=beta_now,
                lambda_contour = lambda_c_now,
                lambda_smooth  = lambda_smooth,
                lambda_area    = lambda_a_now,
                use_label_subdivide = True,
            )

        scaler.scale(total).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(f_param, 5.0)

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        scheduler.step()

        with torch.no_grad():
            for k, idx in enumerate(pinned_indices):
                f_param[idx] = pin_mask[k]

        if total.item() < best_loss:
            best_loss = total.item()
            best_iter = it

        if (it % print_every == 0) or (it == 1) or (it == n_iters):
            print(
                f"iter {it:6d}/{n_iters}  total={total.item():.3e} "
                f"contour={comp['contour']:.3e}  smooth={comp['smoothness']:.3e}  "
                f"area={comp['area_balance']:.3e}  β={beta_now:.1f}  λc={lambda_c_now:.2f}  "
                f"λa={lambda_a_now:.2f}  lr={lr_now:.2e}"
            )
            history.append({
                'iter': it,
                'total': total.item(),
                'contour': comp['contour'],
                'smoothness': comp['smoothness'],
                'area_balance': comp['area_balance'],
                'beta': beta_now,
                'lambda_c': lambda_c_now,
                'lambda_a': lambda_a_now,
                'lr': lr_now
            })

        if enable_early_stopping:
            if it - best_iter > patience:
                print(f"Early stopping at iteration {it} (no improvement for {patience} steps).")
                break

    print(f"Finished in {(time()-t0)/60:.1f} min. Best loss={best_loss:.3e} at iter={best_iter}.")

    # After optimization, save the final mesh and its scalar field values
    final_mesh = vertices_np  # The final mesh
    final_field_values = f_param.detach().cpu().numpy()  # The final scalar field values

    # Save the final mesh and scalar field values to a .npz file
    np.savez_compressed(save_path, mesh=final_mesh, field_values=final_field_values)

    print(f"Final mesh and field values saved to {save_path}")

    return final_field_values, final_mesh, history, save_path







###############################################################################
# 8) Main function
###############################################################################

def main():
    total_start_time = time.time()
    
    # Step 1: Create the sphere mesh
    # pecify either target_points s
    target_points = 5000  # Specify desired number of points (approximate)

    
    print(f"Creating icosphere mesh with target of {target_points} points...")
    start_time = time.time()
    vertices_np, faces_np = load_volume_tet_mesh_and_extract_surface("multipatch\l1-poly-dat\hex\canewt\orig.tet.vtk")
    #vertices_np, faces_np = create_icosphere_mesh(target_points=target_points, subdivisions=2)
    elapsed = time.time() - start_time
    print(f"Created sphere mesh in {elapsed:.2f}s with {len(vertices_np)} vertices and {len(faces_np)} faces")
    
    # Step 2: Choose vertices to pin for 6 regions
    print("Finding vertices to pin...")
    start_time = time.time()
    pinned_indices = find_axis_vertices(vertices_np)
    region_names = ["Top", "Bottom", "Front", "Back", "Right", "Left"]
    elapsed = time.time() - start_time
    print(f"Found pin vertices in {elapsed:.2f}s")
    
    print("Pinning vertices for 6 regions:")
    for i, (name, idx) in enumerate(zip(region_names, pinned_indices)):
        print(f"  {name}: vertex {idx} at position {vertices_np[idx]}")
    
    # Step 3: Optimize the 6-channel scalar field with contour alignment loss
    print("\nStarting optimization...")
    n_iters = 1000000  
    f_optimized, final_mesh, history, save_path= optimization(
        vertices_np=vertices_np,
        faces_np=faces_np,
        pinned_indices=pinned_indices,
        n_iters=n_iters,
        lr=1e-3,
        beta=20.0,
        lambda_contour_initial=0.0,
        lambda_contour_final=10.0,
        lambda_smooth=0.1,
        lambda_area_initial=0.2,
        lambda_area_final=30.0,
        print_every=100,
        target_beta=80.0,  # Adjust if needed
        beta_schedule=True,
        patience=1000000

    # Use the new label subdivision approach
    )
    
    # Step 4: Visualize the result with softmax instead of sigmoid
    print("\nVisualizing result...")
    visualize_segmentation_hardmax(
        vertices_np=vertices_np,
        faces_np=faces_np,
        f_values=f_optimized,
       
        vis_resolution=600,  # Higher resolution for better visualization
       
    )
    
    total_elapsed = time.time() - total_start_time
    print(f"Total execution time: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 



ChatGPT said:
Request for GPT-5 Pro
Reasoned for 4m 14s
Short answer: the “pairwise, weighted‑covariance → SVD → distance‑to‑plane MSE” version is the one that makes the most sense geometrically and tends to be the most stable—with two small fixes: (1) don’t square your weights in sum_xx (one of your variants does), and (2) use an eigendecomposition of the symmetrized covariance (or SVD on the symmetrized matrix) with a tiny diagonal jitter. That’s exactly the “weighted PCA” plane fit you described in your write‑up (Section 4.3.1.3: “Plane fitting via weighted covariance”) and later as “Variant 3” (compact vectorized) .

Here’s what’s different across your versions and why one wins:

“Adjacency + per‑channel plane” version (the first big block).
• You compute gradient differences across triangle adjacencies and then fit one plane per channel using all (c, i) intersections that “belong” to that channel.
• Two issues: (a) boundaries are pairwise objects (between i and j), so aggregating all pairs that touch a channel can mix multiple distinct planes; (b) the adjacency term can dominate and make the plane fit chase gradient noise rather than true boundary geometry. It can converge, but it’s easy to tip into weird minima.

“Pairwise plane fit, with (optional) triple points” (your 2nd & 4th big blocks).
• This is the clean geometry: build all (i<j) channel pairs, collect their edge intersections (plus triple intersections if enabled), compute weighted mean and weighted covariance per pair, SVD/EIGH → smallest singular vector = plane normal, then penalize weighted point‑to‑plane distance MSE.
• This exactly matches the weighted‑covariance approach in your report (4.3.1.3), and the “compact vectorized” variant (4.7.3) .
• It’s simple, interpretable, and differentiable end‑to‑end.

The “compact” 3rd variant (with x = coords * w and x_outer = x @ x^T).
• Bug: you compute sum_xx from x that already contains w, so the outer product contributes w² instead of w. That over‑weights confident edges non‑linearly and destabilizes the covariance. Use weighted_coords * all_coords (one side weighted) or compute outer = (coords.unsqueeze(2) * coords.unsqueeze(1)) * w.unsqueeze(-1).unsqueeze(-1).
• Because of the w², this version tends to be much less stable.

What I recommend (small, surgical upgrades)
Keep your pairwise loss (2nd/4th versions) but make these changes for numerical stability and “sense‑making”:

Use EIGH on a symmetrized covariance (or SVD on the symmetrized matrix).
Covariances are symmetric PSD; tiny FP errors can make them slightly asymmetric. Symmetrize, add a tiny diagonal jitter, and use torch.linalg.eigh to get the eigenvector with the smallest eigenvalue.

python
Copy
# after you compute cov = sum_xx/sum_w - mean_outer
cov = 0.5 * (cov + cov.transpose(1, 2))                # enforce symmetry
trace = cov.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)
jitter = (1e-7 * trace).unsqueeze(-1) / 3.0             # scale-aware tiny jitter
I = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)
cov = cov + jitter * I

# do eigendecomp in float32/64 for stability
evals, evecs = torch.linalg.eigh(cov.float())           # ascending eigenvalues
plane_n = evecs[:, :, 0].to(cov.dtype)                  # smallest eigenvector
plane_n = plane_n / (plane_n.norm(dim=1, keepdim=True) + eps)
plane_d = - (plane_n * mean_).sum(dim=1)
(You can stick to SVD if you prefer; just SVD the symmetrized cov and still pick the last singular vector. EIGH is a touch cleaner for SPD matrices.)

Make sure your covariance uses linear weights, not squared.
In your good versions you did:

python
Copy
weighted_coords = all_coords * all_w.unsqueeze(-1)    # w * x
outer_ = weighted_coords.unsqueeze(2) * all_coords.unsqueeze(1)  # w * (x x^T)
sum_x.index_add_(0, all_pidx, weighted_coords)        # Σ w x
sum_xx_flat.index_add_(0, all_pidx, outer_.reshape(-1, 9))  # Σ w (x x^T)
That’s correct. Avoid the x = coords * w then x_outer = x @ x^T pattern.

Mask degenerate pairs.
Some (i, j) pairs may have almost no weighted intersections. Mask them out or soft‑freeze them:

python
Copy
valid = (sum_w > 1e-8)
# Option A: skip them in loss by zeroing their contribution
# Option B (optional): softly pull them toward a prior normal (e.g., from pinned_axes)
(Optional) robust distances.
If you see outliers, switch from pure MSE to Charbonnier/Huber:

python
Copy
delta = 1e-3
dist = (plane_n[all_pidx] * all_coords).sum(dim=1) + plane_d[all_pidx]
robust = torch.sqrt(dist*dist + delta*delta) - delta
sum_sq.index_add_(0, all_pidx, all_w * robust)
Accumulate moments in higher precision.
If you’re in fp16 mixed precision, cast moment accumulators to fp32 (or even fp64 on CPU) before EIGH/SVD, then cast back.

Triple points: keep them soft and low‑weight early.
Your triple‑point math is fine. Guard it with a modest factor (you already use *0.5) and maybe ramp its contribution as β increases.

Minimal patch inside your 2nd/4th function
Replace your SVD block with this:

python
Copy
# ----- existing up to cov = ... -----
sum_w_clamped = sum_w.clamp_min(eps)
mean_ = sum_x / sum_w_clamped.unsqueeze(-1)
mean_outer = mean_.unsqueeze(2) * mean_.unsqueeze(1)
cov = sum_xx / sum_w_clamped.view(-1,1,1) - mean_outer

# --- stability upgrades ---
cov = 0.5 * (cov + cov.transpose(1,2))
trace = cov.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)
cov = cov + (1e-7 * trace).unsqueeze(-1) / 3.0 * torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0)

evals, evecs = torch.linalg.eigh(cov.float())     # SPD => ascending
plane_n = evecs[:, :, 0].to(cov.dtype)
plane_n = plane_n / (plane_n.norm(dim=1, keepdim=True) + eps)
plane_d = -(plane_n * mean_).sum(dim=1)
…and keep your distance accumulation exactly as you had it (or swap in the robust variant).

Why this one?
It matches the theory and implementation you described in your report: weighted covariance → smallest mode gives the plane normal; then measure distances to that plane (Sec. 4.3.1.3; “Variant 3” in Sec. 4.7) .

It keeps the semantics correct (planes are boundaries between channel pairs, not “per channel”).

Symmetrizing + EIGH + tiny jitter cures the common “covariance not quite PSD” / SVD wiggle you mentioned under “Technical Challenges and Solutions” (4.5.11) and “Convergence Challenges” (4.7.4) .

Fixing the weight squaring bug removes a big source of instability.

If you want, I can fold those exact changes into your preferred file and return a cleaned, single canonical contour_alignment_loss for you to drop in.






You said:
CAN YOUR REWRITE THE LOSS IN YOUR WAY? WITH ALL THE STABLIZING CONVERGING TRICKS ? """
Improved loss functions addressing convergence issues from the report.
Key improvements:
- Intrinsic contour alignment (replaces 3D SVD-based plane fitting)
- Cotangent Laplacian smoothness (replaces unnormalized edge differences)  
- KL divergence area balance (replaces L1 loss)
- Soft pin penalty (replaces hard projection)
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    Safe normalization that avoids NaN gradients when x≈0.
    
    Args:
        x: Input tensor to normalize
        dim: Dimension along which to normalize
        eps: Small value to clamp the norm (default 1e-6)
        
    Returns:
        Normalized tensor with stable gradients
    """
    # Clamp the denominator so backward never sees a divide-by-zero
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def grad3d_intrinsic(h_vals: torch.Tensor, v0: torch.Tensor,
                     v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Compute intrinsic gradient on triangles using Gram matrix (batched).
    
    Args:
        h_vals: (B,3) scalar values at triangle vertices
        v0,v1,v2: (B,3) triangle vertex positions
        
    Returns:
        g: (B,3) gradient in R^3 lying in the triangle plane
    """
    e0 = v1 - v0                      # (B,3)
    e1 = v2 - v0                      # (B,3)
    b = torch.stack([h_vals[:,1]-h_vals[:,0],
                     h_vals[:,2]-h_vals[:,0]], dim=1)  # (B,2)

    # Do geometry in float64 for robustness, cast back at the end
    dt = torch.float64
    e0d, e1d, bd = e0.to(dt), e1.to(dt), b.to(dt)

    # Gram matrix G = [[<e0,e0>, <e0,e1>],
    #                  [<e1,e0>, <e1,e1>]]
    G00 = (e0d*e0d).sum(dim=1)          # (B,)
    G01 = (e0d*e1d).sum(dim=1)          # (B,)
    G11 = (e1d*e1d).sum(dim=1)          # (B,)
    det = (G00*G11 - G01*G01)
    
    # Mask degenerate triangles
    mask_degenerate = det <= 1e-10
    det = det.clamp_min(1e-10)
    
    invG00 =  G11 / det
    invG01 = -G01 / det
    invG11 =  G00 / det

    # coefficients a in the (e0,e1) basis: a = G^{-1} b
    a0 = invG00*bd[:,0] + invG01*bd[:,1]    # (B,)
    a1 = invG01*bd[:,0] + invG11*bd[:,1]    # (B,)
    
    # Zero out coefficients for degenerate triangles (no reliable gradient)
    a0 = a0.masked_fill(mask_degenerate, 0.0)
    a1 = a1.masked_fill(mask_degenerate, 0.0)

    # gradient in R^3: g = a0*e0 + a1*e1
    gd = a0[:,None]*e0d + a1[:,None]*e1d     # (B,3)
    g = gd.to(h_vals.dtype)
    
    # Guard: if anything still slipped through, zero it (rare)
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


def contour_alignment_intrinsic(F: torch.Tensor,
                               faces: torch.Tensor,
                               edge_idx: torch.Tensor,
                               edge_tris: torch.Tensor,
                               beta_contour: float,
                               top_k: int = 2,
                               return_weights: bool = False,
                               verts: Optional[torch.Tensor] = None,
                               tri_xy: Optional[torch.Tensor] = None,
                               debug_log: bool = False) -> torch.Tensor:
    """
    Contour alignment loss with proper 3D alignment.
    
    Key improvements:
    - Computes boundary tangents in 3D space for proper alignment
    - No SVD or global plane fitting
    - Pair gating to avoid O(C^2) noise
    - Robust Charbonnier penalty
    
    Args:
        F: (N, C) multi-channel field values
        faces: (T, 3) face indices
        edge_idx: (E, 2) edge vertex indices
        edge_tris: (E, 2) adjacent triangle indices (-1 for boundary)
        beta_contour: Temperature parameter for edge crossing detection
        top_k: Number of top channels to consider (pair gating)
        return_weights: Whether to return edge weights
        verts: (N, 3) vertex positions in 3D (required for 3D alignment)
        tri_xy: (T, 3, 2) DEPRECATED - kept for compatibility, not used
        
    Returns:
        loss: Scalar contour alignment loss (normalized)
        weights: (E,) edge weights if return_weights=True
    """
    device, dtype = F.device, F.dtype
    C = F.shape[1]
    
    # Ensure input tensors are on the same device as F
    edge_idx = edge_idx.to(device)
    edge_tris = edge_tris.to(device)
    faces = faces.to(device)
    
    # Need 3D vertices for proper alignment
    if verts is None:
        raise ValueError("verts must be provided for 3D contour alignment")
    verts = verts.to(device)
    
    # No need to precompute frames - we'll compute normals per active triangle
    
    # Filter out boundary edges
    valid_mask = (edge_tris[:, 0] >= 0) & (edge_tris[:, 1] >= 0)
    valid_edges = edge_idx[valid_mask]  # (E_valid, 2)
    valid_tris = edge_tris[valid_mask]  # (E_valid, 2)
    
    if valid_edges.shape[0] == 0:
        return torch.tensor(0., device=device, dtype=dtype)
    
    # Edge midpoint field values
    f_mid = 0.5 * (F[valid_edges[:, 0]] + F[valid_edges[:, 1]])  # (E_valid, C)
    
    # Pair gating: select top 2 channels per edge
    if top_k < C and C > 1:
        top_vals, top_idx = torch.topk(f_mid, k=min(top_k, C), dim=1)  # (E_valid, top_k)
        # Use first two channels from top-k
        chan_i = top_idx[:, 0]  # (E_valid,)
        chan_j = top_idx[:, 1] if top_idx.shape[1] > 1 else top_idx[:, 0]
    else:
        # Use fixed channel pair for all edges
        chan_i = torch.zeros(valid_edges.shape[0], dtype=torch.long, device=device)
        chan_j = torch.ones(valid_edges.shape[0], dtype=torch.long, device=device)
    
    # Edge crossing weights
    da = F[valid_edges[:, 0], chan_i] - F[valid_edges[:, 0], chan_j]  # (E_valid,)
    db = F[valid_edges[:, 1], chan_i] - F[valid_edges[:, 1], chan_j]  # (E_valid,)
    w = torch.sigmoid(-beta_contour * da * db)  # (E_valid,)
    
    # NEW: Confidence gate - only treat high-confidence crossings as boundaries
    conf = 0.5 * (da.abs() + db.abs())
    k = 5.0  # sharpness
    m = 0.3  # margin threshold (could be scheduled from 0.1 -> 0.4)
    w = w * torch.sigmoid(k * (conf - m))
    
    # Soft gating to maintain gradient flow
    w = w.clamp_min(1e-3)  # Keep gradients flowing
    
    # Use all edges with non-zero weights for soft gating
    active_edges = valid_edges
    active_tris = valid_tris
    active_chan_i = chan_i
    active_chan_j = chan_j
    active_w = w
    
    # Additional robustness: downweight edges where a third channel is close
    if C > 2:
        # Get the 3rd highest value at edge midpoints
        # Note: we're using all edges now (soft gating)
        top3_vals, _ = torch.topk(f_mid, k=min(3, C), dim=1)
        if top3_vals.shape[1] == 3:
            # Difference between 2nd and 3rd channels
            gap = top3_vals[:, 1] - top3_vals[:, 2]
            # Downweight when gap is small (triple point)
            triple_weight = torch.sigmoid(10.0 * (gap - 0.1))
            active_w = active_w * triple_weight
    
    # Store original edges for gradient/length gating
    original_active_edges = active_edges
    
    # Get triangles
    tL = active_tris[:, 0]  # (E_active,)
    tR = active_tris[:, 1]  # (E_active,)
    
    # Vectorized height computation for left triangles
    faces_L = faces[tL]  # (E_active, 3)
    F_L = F[faces_L]  # (E_active, 3, C)
    h_L = F_L[torch.arange(len(tL)), :, active_chan_i] - \
          F_L[torch.arange(len(tL)), :, active_chan_j]  # (E_active, 3)
    
    # Vectorized height computation for right triangles  
    faces_R = faces[tR]  # (E_active, 3)
    F_R = F[faces_R]  # (E_active, 3, C)
    h_R = F_R[torch.arange(len(tR)), :, active_chan_i] - \
          F_R[torch.arange(len(tR)), :, active_chan_j]  # (E_active, 3)
    
    # Get triangle vertices for intrinsic gradient computation
    v0L = verts[faces[tL, 0]]  # (E_active, 3)
    v1L = verts[faces[tL, 1]]  # (E_active, 3)
    v2L = verts[faces[tL, 2]]  # (E_active, 3)
    
    v0R = verts[faces[tR, 0]]  # (E_active, 3)
    v1R = verts[faces[tR, 1]]  # (E_active, 3)
    v2R = verts[faces[tR, 2]]  # (E_active, 3)
    
    # Compute intrinsic 3D gradients using Gram matrix
    g_L_3d = grad3d_intrinsic(h_L, v0L, v1L, v2L)  # (E_active, 3)
    g_R_3d = grad3d_intrinsic(h_R, v0R, v1R, v2R)  # (E_active, 3)
    
    # Compute triangle normals using safe normalization
    e0L = v1L - v0L
    e1L = v2L - v0L
    n_L = safe_normalize(torch.cross(e0L, e1L, dim=1), dim=1, eps=1e-6)
    
    e0R = v1R - v0R
    e1R = v2R - v0R
    n_R = safe_normalize(torch.cross(e0R, e1R, dim=1), dim=1, eps=1e-6)
    
    # Project gradients into the triangle plane first (improves stability)
    g_L_3d = g_L_3d - (g_L_3d * n_L).sum(dim=1, keepdim=True) * n_L
    g_R_3d = g_R_3d - (g_R_3d * n_R).sum(dim=1, keepdim=True) * n_R
    
    # Boundary tangent in 3D: tau = normalize(n × g)
    tau_L = torch.cross(n_L, g_L_3d, dim=1)  # (E_active, 3)
    tau_R = torch.cross(n_R, g_R_3d, dim=1)  # (E_active, 3)
    
    # Normalize tangents using safe normalization
    tau_L = safe_normalize(tau_L, dim=1, eps=1e-6)
    tau_R = safe_normalize(tau_R, dim=1, eps=1e-6)
    
    # Guard: if any NaN snuck in
    tau_L = torch.nan_to_num(tau_L)
    tau_R = torch.nan_to_num(tau_R)
    active_w = torch.nan_to_num(active_w, nan=1e-3, posinf=1.0, neginf=1e-3)
    
    # Gradient magnitude gating: stronger signals = more reliable tangents
    mag_L = g_L_3d.norm(dim=1)  # (E_active,)
    mag_R = g_R_3d.norm(dim=1)
    
    # Down-weight edges if either adjacent triangle was degenerate
    tiny = 1e-12
    deg_mask = (mag_L < tiny) | (mag_R < tiny)
    active_w = active_w * (~deg_mask).float()
    
    grad_gate = torch.sqrt(mag_L * mag_R)  # geometric mean
    # Normalize but keep relative strength (ignore NaNs)
    scale = torch.nanmedian(grad_gate).detach().clamp_min(1e-6)  # ignore NaNs
    grad_gate = (grad_gate / scale).clamp(0.0, 2.0)  # Cap at 2.0 for high β stability
    grad_gate = torch.nan_to_num(grad_gate, nan=0.0, posinf=2.0, neginf=0.0)
    
    # Edge length gating: longer edges matter more
    edge_vec = verts[original_active_edges[:, 1]] - verts[original_active_edges[:, 0]]
    edge_len = edge_vec.norm(dim=1)  # (E_active,)
    len_gate = (edge_len / (edge_len.mean().detach() + 1e-12)).clamp(0.5, 2.0)
    
    # Apply both gates to weights
    active_w = active_w * grad_gate * len_gate
    
    # Debug logging if requested
    if debug_log:
        print(f"[DEBUG contour] grad_gate: min={grad_gate.min():.6f}, max={grad_gate.max():.6f}, "
              f"median={torch.nanmedian(grad_gate):.6f}, %nan={torch.isnan(grad_gate).float().mean():.1%}")
        print(f"[DEBUG contour] active_w: min={active_w.min():.6f}, max={active_w.max():.6f}, "
              f"sum={active_w.sum():.3f}")
    
    # Alignment in 3D: 1 - |cos θ|
    cos_angle = (tau_L * tau_R).sum(dim=1).abs().clamp(max=1.0)  # (E_active,)
    misalignment = 1.0 - cos_angle
    
    # Charbonnier penalty
    epsilon = 1e-6
    loss_contrib = torch.sqrt(misalignment * misalignment + epsilon)
    
    # Weighted mean
    total_loss = (active_w * loss_contrib).sum()
    total_weight = active_w.sum() + 1e-9
    
    if return_weights:
        # Return full edge weights (including inactive edges)
        full_weights = torch.zeros(edge_idx.shape[0], device=device, dtype=dtype)
        # Map valid edges back
        valid_indices = torch.where(valid_mask)[0]
        full_weights[valid_indices] = active_w
        return total_loss / total_weight, full_weights
    
    return total_loss / total_weight


def smoothness_cotan(F: torch.Tensor, 
                    I: torch.Tensor, 
                    J: torch.Tensor, 
                    W: torch.Tensor) -> torch.Tensor:
    """
    Cotangent Laplacian smoothness loss.
    This replaces the unnormalized edge-based smoothness (report section 4.3.3.2).
    
    Args:
        F: (N, C) multi-channel field values
        I: (K,) source vertex indices
        J: (K,) target vertex indices  
        W: (K,) cotangent weights
        
    Returns:
        loss: Normalized smoothness loss
    """
    diff = F[I] - F[J]  # (K, C)
    squared_diff = (diff * diff).sum(dim=-1)  # (K,)
    
    # Weighted sum normalized by total weight
    numerator = (W * squared_diff).sum()
    denominator = W.sum().clamp_min(1e-12)
    
    return numerator / denominator


def area_fractions_and_kl(F: torch.Tensor, 
                         faces: torch.Tensor, 
                         tri_area: torch.Tensor, 
                         beta_area: float,
                         use_entropy_regularization: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Area balance loss using KL divergence to uniform distribution.
    This replaces L1 loss which has vanishing gradients (report section 4.3.2).
    
    Args:
        F: (N, C) multi-channel field values
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        beta_area: Temperature parameter for softmax
        
    Returns:
        loss: KL divergence loss
        frac: (C,) area fractions per channel
    """
    C = F.shape[1]
    
    # Barycentric sampling points (center + edge midpoints)
    bary = torch.tensor([[1/3, 1/3, 1/3],
                        [1/2, 1/2, 0.0],
                        [1/2, 0.0, 1/2],
                        [0.0, 1/2, 1/2]], device=F.device, dtype=F.dtype)  # (4, 3)
    
    # Sample field at barycentric points
    Ft = F[faces]  # (T, 3, C)
    Ft_sampled = torch.einsum('sb,tbc->tsc', bary, Ft)  # (T, 4, C)
    
    # Softmax probabilities
    P = torch.softmax(beta_area * Ft_sampled, dim=-1)  # (T, 4, C)
    P_mean = P.mean(dim=1)  # (T, C) average over sample points
    
    # Area-weighted channel probabilities
    area_per_channel = (tri_area[:, None] * P_mean).sum(dim=0)  # (C,)
    total_area = tri_area.sum().clamp_min(1e-12)
    frac = area_per_channel / total_area
    
    # KL(frac || uniform) = sum_c frac_c * log(frac_c / uniform)
    uniform = 1.0 / C
    # Clamp frac to avoid log(0)
    frac = frac.clamp(min=1e-8, max=1-1e-8)
    kl_loss = (frac * torch.log(frac / uniform)).sum()
    
    # During warmup, use variance loss for stronger gradient signal
    if use_entropy_regularization:
        # Variance of area fractions from uniform
        variance_loss = ((frac - uniform) ** 2).sum() * C * 10.0  # Scale up for stronger signal
        
        # Also add entropy regularization to prevent winner-take-all
        P_mean_clamped = P_mean.clamp(min=1e-8, max=1-1e-8)
        entropy_per_tri = -(P_mean_clamped * torch.log(P_mean_clamped)).sum(dim=1)  # (T,)
        mean_entropy = (tri_area * entropy_per_tri).sum() / total_area
        max_entropy = -torch.log(torch.tensor(1.0/C, device=F.device))
        entropy_loss = (max_entropy - mean_entropy) * 0.1
        
        # Return variance + entropy loss instead of KL during warmup
        return variance_loss + entropy_loss, frac
    
    return kl_loss, frac


def pin_loss(F: torch.Tensor, 
            pin_idx: torch.Tensor, 
            pin_target: torch.Tensor,
            use_huber: bool = True,
            delta: float = 1.0) -> torch.Tensor:
    """
    Soft pinning penalty (annealed to hard constraint).
    This replaces hard projection after each step (report section 4.4.7).
    
    Args:
        F: (N, C) multi-channel field values
        pin_idx: (P,) indices of pinned vertices
        pin_target: (P, C) target values for pinned vertices
        use_huber: Whether to use Huber loss for robustness
        delta: Huber loss threshold
        
    Returns:
        loss: Pin constraint loss
    """
    if pin_idx.numel() == 0:
        return torch.tensor(0., device=F.device, dtype=F.dtype)
    
    diff = F[pin_idx] - pin_target  # (P, C)
    
    if use_huber:
        # Huber loss for robustness
        abs_diff = diff.abs()
        delta_t = torch.full_like(abs_diff, delta)
        quadratic = torch.minimum(abs_diff, delta_t)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic**2 + delta * linear
        return loss.mean()
    else:
        # Standard L2 loss
        return (diff * diff).mean()


def compute_boundary_stats(F: torch.Tensor,
                          edge_idx: torch.Tensor,
                          verts: torch.Tensor,
                          beta_contour: float,
                          top_k: int = 2) -> Tuple[float, float]:
    """
    Compute boundary length and active edge fraction for monitoring.
    
    Args:
        F: (N, C) multi-channel field values
        edge_idx: (E, 2) edge vertex indices
        verts: (N, 3) vertex positions
        beta_contour: Temperature parameter
        top_k: Number of top channels to consider
        
    Returns:
        length: Estimated boundary length
        active_fraction: Fraction of edges with w > 0.5
    """
    device = F.device
    C = F.shape[1]
    
    # Ensure edge_idx is on same device
    edge_idx = edge_idx.to(device)
    
    # Edge midpoint values
    f_mid = 0.5 * (F[edge_idx[:, 0]] + F[edge_idx[:, 1]])  # (E, C)
    
    # Get top 2 channels per edge
    if top_k < C and C > 1:
        top_vals, top_idx = torch.topk(f_mid, k=min(top_k, C), dim=1)
        chan_i = top_idx[:, 0]
        chan_j = top_idx[:, 1] if top_idx.shape[1] > 1 else top_idx[:, 0]
    else:
        chan_i = torch.zeros(edge_idx.shape[0], dtype=torch.long, device=device)
        chan_j = torch.ones(edge_idx.shape[0], dtype=torch.long, device=device)
    
    # Edge crossing weights
    da = F[edge_idx[:, 0], chan_i] - F[edge_idx[:, 0], chan_j]
    db = F[edge_idx[:, 1], chan_i] - F[edge_idx[:, 1], chan_j]
    w = torch.sigmoid(-beta_contour * da * db)
    
    # Active edges (w > 0.5)
    active_mask = w > 0.5
    active_fraction = active_mask.float().mean().item()
    
    # Boundary length (sum of active edge lengths)
    if active_mask.any():
        active_edges = edge_idx[active_mask]
        edge_lengths = (verts[active_edges[:, 0]] - verts[active_edges[:, 1]]).norm(dim=1)
        length = edge_lengths.sum().item()
    else:
        length = 0.0
    
    return length, active_fraction


def compute_boundary_length_estimate(F: torch.Tensor,
                                   edge_idx: torch.Tensor,
                                   verts: torch.Tensor,
                                   beta_contour: float,
                                   top_k: int = 2) -> float:
    """
    Backward compatibility wrapper.
    """
    length, _ = compute_boundary_stats(F, edge_idx, verts, beta_contour, top_k)
    return length


def non_boundary_margin_loss(F: torch.Tensor, 
                            edge_idx: torch.Tensor, 
                            edge_weights: torch.Tensor,
                            tau: float = 0.3) -> torch.Tensor:
    """
    Encourage vertices not on boundaries to have clear winner channels.
    
    Args:
        F: (N, C) field values
        edge_idx: (E, 2) edge indices
        edge_weights: (E,) boundary weights from contour loss
        tau: Minimum margin to enforce
        
    Returns:
        loss: Margin sharpening loss
    """
    # Ensure tensors are on same device
    device = F.device
    edge_idx = edge_idx.to(device)
    edge_weights = edge_weights.to(device)
    
    # Edges that are clearly not on boundaries
    nb_mask = (edge_weights.detach() < 0.2)
    if not nb_mask.any():
        return F.new_tensor(0.0)
    
    va = edge_idx[nb_mask, 0]
    vb = edge_idx[nb_mask, 1]
    
    # Get margins (top1 - top2) for vertices
    def margin(v_idx):
        top2_vals, _ = torch.topk(F[v_idx], k=2, dim=1)
        return top2_vals[:, 0] - top2_vals[:, 1]
    
    # Take minimum margin across edge endpoints
    m = torch.minimum(margin(va), margin(vb))
    
    # Penalize margins below tau
    return torch.nn.functional.relu(tau - m).mean()


def total_variation_loss(F: torch.Tensor,
                        edge_idx: torch.Tensor,
                        edge_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Total variation regularization for additional smoothness.
    
    Args:
        F: (N, C) multi-channel field values
        edge_idx: (E, 2) edge vertex indices
        edge_lengths: (E,) optional edge lengths for weighting
        
    Returns:
        loss: Total variation loss
    """
    va, vb = edge_idx.T
    diff = F[va] - F[vb]  # (E, C)
    
    # L1 norm per edge
    tv_per_edge = diff.abs().sum(dim=-1)  # (E,)
    
    if edge_lengths is not None:
        # Weight by inverse edge length
        weights = 1.0 / (edge_lengths + 1e-12)
        tv_per_edge = tv_per_edge * weights
        normalizer = weights.sum()
    else:
        normalizer = edge_idx.shape[0]
    
    return tv_per_edge.sum() / normalizer


def potts_smoothness_loss(F: torch.Tensor,
                         edge_idx: torch.Tensor,
                         edge_weights: torch.Tensor,
                         beta_area: float,
                         gamma: float = 2.0) -> torch.Tensor:
    """
    Potts/TV-style smoothness on soft probabilities to reduce speckles.
    
    Args:
        F: (N, C) field values
        edge_idx: (E, 2) edge indices
        edge_weights: (E,) boundary weights from contour loss (activity)
        beta_area: Temperature for softmax
        gamma: Exponent for gating away from boundaries (default 2.0 for softer gating)
        
    Returns:
        loss: Potts smoothness loss
    """
    device = F.device
    edge_idx = edge_idx.to(device)
    edge_weights = edge_weights.to(device)
    
    # Compute soft probabilities
    p = torch.softmax(beta_area * F, dim=1)  # (N, C)
    
    # Get edge endpoint probabilities
    va, vb = edge_idx.T
    p_a = p[va]  # (E, C)
    p_b = p[vb]  # (E, C)
    
    # Potts loss: 1 - p_i^T p_j = 0.5 * ||p_i - p_j||^2
    potts_per_edge = 1.0 - (p_a * p_b).sum(dim=1)  # (E,)
    
    # Gate away from boundaries
    gate = (1.0 - edge_weights.detach()).pow(gamma)
    
    # Weighted mean
    return (gate * potts_per_edge).sum() / (gate.sum() + 1e-9)


def boundary_length_regularizer(edge_idx: torch.Tensor,
                               edge_weights: torch.Tensor,
                               verts: torch.Tensor) -> torch.Tensor:
    """
    Regularize total boundary length to reduce ragged seams (scale-invariant).
    
    Args:
        edge_idx: (E, 2) edge indices
        edge_weights: (E,) boundary weights (activity)
        verts: (N, 3) vertex positions
        
    Returns:
        loss: Normalized boundary length penalty
    """
    device = verts.device
    edge_idx = edge_idx.to(device)
    edge_weights = edge_weights.to(device)
    
    # Edge lengths
    va, vb = edge_idx.T
    edge_lengths = (verts[va] - verts[vb]).norm(dim=1)  # (E,)
    
    # Normalize by total edge length for scale invariance
    denom = edge_lengths.sum().clamp_min(1e-9)
    
    # Weighted sum of active edge lengths, normalized
    return (edge_weights * edge_lengths).sum() / denom


def normal_axis_losses(verts: torch.Tensor,
                      faces: torch.Tensor,
                      tri_area: torch.Tensor,
                      F_field: torch.Tensor,
                      beta_area: float,
                      axis_per_channel: torch.Tensor,
                      eps: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normal-based losses to encourage axis-aligned patches.
    
    Args:
        verts: (N, 3) vertex positions
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        F_field: (N, C) multi-channel field values
        beta_area: Temperature for softmax
        axis_per_channel: (C, 3) unit vectors for each channel's target axis
        eps: Small value for numerical stability
        
    Returns:
        loss_align: Mean axis misalignment across channels
        loss_disp: Mean within-patch normal dispersion across channels
    """
    device = verts.device
    
    # Face normals (unit)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=1)
    n = safe_normalize(n, dim=1, eps=1e-6)  # (T, 3)
    
    # Soft probabilities per vertex and then per face (average over the 3 vertices)
    p_v = torch.softmax(beta_area * F_field, dim=1)  # (N, C)
    p_f = (p_v[faces[:, 0]] + p_v[faces[:, 1]] + p_v[faces[:, 2]]) / 3.0  # (T, C)
    
    # Area-weighted per-channel mean normals
    w = tri_area[:, None] * p_f  # (T, C)
    m = (w[:, :, None] * n[:, None, :]).sum(dim=0)  # (C, 3)
    m = safe_normalize(m, dim=1, eps=1e-6)  # (C, 3)
    
    # (a) Axis alignment: 1 - |dot(mean_normal, axis)|
    axes = safe_normalize(axis_per_channel, dim=1, eps=1e-6)
    misalign = 1.0 - (m * axes).sum(dim=1).abs()  # (C,)
    loss_align = misalign.mean()
    
    # (b) Dispersion: 1 - cos^2 between each face normal and its patch mean
    cos = (n[:, None, :] * m[None, :, :]).sum(dim=2).clamp(-1, 1)  # (T, C)
    disp_per_face = (1.0 - cos**2)  # (T, C)
    loss_disp = (w * disp_per_face).sum(dim=0) / (w.sum(dim=0) + eps)  # (C,)
    loss_disp = loss_disp.mean()
    
    return loss_align, loss_disp


def area_kl_to_prior(frac: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from area fractions to a non-uniform prior.
    Useful when you want 5 patches instead of 6.
    
    Args:
        frac: (C,) actual area fractions that sum to 1
        prior: (C,) target area fractions that sum to 1
        
    Returns:
        KL divergence loss
    """
    eps = 1e-8
    f = frac.clamp(eps, 1.0)
    q = prior.clamp(eps, 1.0)
    return (f * (f / q).log()).sum()


def contour_alignment_intrinsic_v2(
    F: torch.Tensor,
    faces: torch.Tensor,
    edge_idx: torch.Tensor,
    edge_tris: torch.Tensor,
    verts: torch.Tensor,
    beta_contour: float = 6.0,
    eps: float = 1e-6,
    return_weights: bool = False,
    debug_log: bool = False,
) -> torch.Tensor:
    """
    Robust contour alignment:
      - all-pairs per-edge with per-edge softmax over pairs
      - soft-OR coverage weighting per edge
      - intrinsic 3D gradient/tangent as before
    """
    device, dtype = F.device, F.dtype
    N, C = F.shape
    edge_idx   = edge_idx.to(device)
    edge_tris  = edge_tris.to(device)
    faces      = faces.to(device)
    verts      = verts.to(device)

    # only interior edges
    valid = (edge_tris[:,0] >= 0) & (edge_tris[:,1] >= 0)
    if not torch.any(valid):
        z = F.new_tensor(0.0)
        return (z, torch.zeros_like(valid, dtype=F.dtype)) if return_weights else z

    va = edge_idx[valid,0]
    vb = edge_idx[valid,1]
    tL = edge_tris[valid,0]
    tR = edge_tris[valid,1]

    # ----- pairwise crossing weights on edges -----
    # all channel pairs
    ii, jj = torch.triu_indices(C, C, offset=1, device=device)  # P = C*(C-1)//2
    # values at endpoints for all pairs
    da = F[va][:, ii] - F[va][:, jj]      # (E,P)
    db = F[vb][:, ii] - F[vb][:, jj]      # (E,P)

    # crossing score: sign change & confidence
    w_pairs = torch.sigmoid(-beta_contour * da * db)           # (E,P)
    conf    = 0.5*(da.abs() + db.abs())
    w_pairs = w_pairs * torch.sigmoid(5.0*(conf - 0.2))        # soften but keep grads
    w_pairs = w_pairs.clamp_min(1e-6)

    # per-edge mixing over pairs (prevents cherry-picking)
    pair_mix = w_pairs / (w_pairs.sum(dim=1, keepdim=True) + eps)  # (E,P)

    # soft-OR edge activity (coverage)
    phi = 1.0 - torch.prod(1.0 - w_pairs, dim=1)  # (E,)

    # ----- per-pair 3D tangents on each side -----
    faces_L = faces[tL]                     # (E,3)
    faces_R = faces[tR]
    F_L     = F[faces_L]                    # (E,3,C)
    F_R     = F[faces_R]

    # build h values for all pairs: (E,3,P)
    hL = F_L[:,:,ii] - F_L[:,:,jj]
    hR = F_R[:,:,ii] - F_R[:,:,jj]

    # triangle vertices
    v0L, v1L, v2L = [verts[faces_L[:,k]] for k in (0,1,2)]
    v0R, v1R, v2R = [verts[faces_R[:,k]] for k in (0,1,2)]

    # reuse your intrinsic gradient in 3D
    def grad_many(h, v0, v1, v2):
        # h: (E,3,P) -> (E,P,3)
        E, _, P = h.shape
        h_flat  = h.permute(0,2,1).reshape(E*P, 3)
        g = grad3d_intrinsic(h_flat, v0.repeat_interleave(P,0),
                                       v1.repeat_interleave(P,0),
                                       v2.repeat_interleave(P,0))
        return g.reshape(E, P, 3)

    gL = grad_many(hL, v0L, v1L, v2L)     # (E,P,3)
    gR = grad_many(hR, v0R, v1R, v2R)

    # normals per triangle side
    def normals(v0, v1, v2):
        e0 = v1 - v0
        e1 = v2 - v0
        n  = torch.cross(e0, e1, dim=1)
        return safe_normalize(n, dim=1, eps=1e-6)

    nL = normals(v0L, v1L, v2L)           # (E,3)
    nR = normals(v0R, v1R, v2R)

    # project gradients into the plane (stability)
    def proj_in_plane(g, n):
        # g: (E,P,3), n: (E,3)
        dot = (g * n[:,None,:]).sum(dim=2, keepdim=True)
        return g - dot * n[:,None,:]

    gL = proj_in_plane(gL, nL)
    gR = proj_in_plane(gR, nR)

    # tangents tau = n x g
    tauL = safe_normalize(torch.cross(nL[:,None,:], gL, dim=2), dim=2, eps=1e-6)  # (E,P,3)
    tauR = safe_normalize(torch.cross(nR[:,None,:], gR, dim=2), dim=2, eps=1e-6)

    cosang = (tauL * tauR).sum(dim=2).abs().clamp_max(1.0)  # (E,P)
    mis    = 1.0 - cosang                                   # (E,P)

    # per-edge expected misalignment over pairs
    mis_edge = (pair_mix * mis).sum(dim=1)                  # (E,)

    # gradient gate per edge: geometric mean of magnitudes across pairs
    gLm = gL.norm(dim=2).mean(dim=1) + eps                  # (E,)
    gRm = gR.norm(dim=2).mean(dim=1) + eps
    grad_gate = torch.sqrt(gLm * gRm)
    scale = torch.nanmedian(grad_gate).detach().clamp_min(1e-6)
    grad_gate = (grad_gate/scale).clamp(0.0, 2.0)
    grad_gate = torch.nan_to_num(grad_gate, nan=0.0, posinf=2.0, neginf=0.0)

    # length gate
    edge_vec = verts[vb] - verts[va]
    edge_len = edge_vec.norm(dim=1)
    len_gate = (edge_len / (edge_len.mean().detach() + 1e-12)).clamp(0.5, 2.0)

    # final per-edge weight
    w_edge = (phi.clamp_min(1e-4) * grad_gate * len_gate)   # (E,)

    # Charbonnier
    loss_edge = torch.sqrt(mis_edge*mis_edge + 1e-6)

    num = (w_edge * loss_edge).sum()
    den = (w_edge.sum() + 1e-9)
    loss = num / den

    if debug_log:
        afrac = (phi > 0.5).float().mean().item()
        print(f"[contour_v2] loss={loss.item():.4g}  act_frac={afrac:.3f}  "
              f"w_sum={w_edge.sum().item():.2f}")

    if return_weights:
        full = torch.zeros(edge_idx.shape[0], dtype=dtype, device=device)
        full[valid] = phi   # use φ as "boundary-ness" for other terms
        return loss, full

    return loss


def triple_point_barrier(F: torch.Tensor, faces: torch.Tensor, tri_area: torch.Tensor, beta_area: float = 10.0, margin: float = 0.10) -> torch.Tensor:
    """
    Discourage three equal channels in one triangle (speckles/"Y" junctions everywhere).
    Looks at the softmaxed per-triangle distribution and pushes the (2nd−3rd) gap above a margin.
    
    Args:
        F: (N, C) multi-channel field values
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        beta_area: Temperature for softmax
        margin: Minimum gap between 2nd and 3rd channel
        
    Returns:
        loss: Triple point barrier loss
    """
    # soft probabilities per triangle (avg over vertices)
    p_v = torch.softmax(beta_area * F, dim=1)                      # (N,C)
    p_t = (p_v[faces[:,0]] + p_v[faces[:,1]] + p_v[faces[:,2]])/3  # (T,C)
    top3, _ = torch.topk(p_t, k=min(3, F.shape[1]), dim=1)         # (T,3)
    if top3.shape[1] < 3:
        return F.new_tensor(0.0)
    gap = top3[:,1] - top3[:,2]                                    # (T,)
    # penalize when 3rd channel too close to 2nd
    pen = torch.nn.functional.relu(margin - gap)
    w   = tri_area / (tri_area.sum() + 1e-9)
    return (w * pen).sum()


def area_balance_loss(
    F: torch.Tensor,                # (N,C)
    faces: torch.Tensor,            # (T,3)
    tri_area: torch.Tensor,         # (T,)
    beta_area: float,
    use_straight_through: bool = True,
    method: str = "rev_kl",         # "rev_kl" | "js" | "l2"
    min_frac: Optional[float] = None,
    max_frac: Optional[float] = None,
    barrier_w: float = 0.05,
    eps: float = 1e-8,
    entropy_weight: float = 0.0     # Add entropy regularization
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Area balance that actually resists collapse.
    - rev_kl  : cross-entropy to uniform = KL(uniform || frac) + const
    - js      : symmetric Jensen–Shannon (safer than either KL)
    - l2      : simple squared error to uniform
    Optionally uses straight-through hard counting to align with argmax.
    
    Args:
        F: (N, C) multi-channel field values
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        beta_area: Temperature for softmax
        use_straight_through: Use hard counting in forward, soft in backward
        method: Loss method - "rev_kl", "js", or "l2"
        min_frac: Minimum allowed fraction per channel (default 0.5/C)
        max_frac: Maximum allowed fraction per channel (default 2.0/C)
        barrier_w: Weight for soft barrier penalty
        eps: Small value for numerical stability
        
    Returns:
        loss: Area balance loss
        frac: (C,) area fractions per channel
    """
    device, dtype = F.device, F.dtype
    N, C = F.shape
    uniform = F.new_full((C,), 1.0 / C)

    # Vertex probabilities
    p = torch.softmax(beta_area * F, dim=1)  # (N,C)

    if use_straight_through:
        # forward = hard one-hot; backward = soft p
        hard = torch.zeros_like(p).scatter_(1, p.argmax(dim=1, keepdim=True), 1.0)
        p = hard - p.detach() + p

    # Face probs (average of vertices)
    p_f = (p[faces[:, 0]] + p[faces[:, 1]] + p[faces[:, 2]]) / 3.0  # (T,C)

    # Area per channel and fractions
    A_c = (tri_area[:, None] * p_f).sum(dim=0)                      # (C,)
    total = tri_area.sum().clamp_min(eps)
    frac = (A_c / total).clamp(eps, 1.0 - eps)                      # (C,)

    # Losses
    if method == "rev_kl":
        # = - sum_c (1/C) log frac_c  (punishes tiny frac_c strongly)
        loss = -(uniform * frac.log()).sum()
    elif method == "js":
        m = 0.5 * (frac + uniform)
        loss = 0.5 * ((frac * (frac.add(eps).log() - m.add(eps).log())).sum()
                      + (uniform * (uniform.add(eps).log() - m.add(eps).log())).sum())
    elif method == "l2":
        loss = ((frac - uniform) ** 2).sum() * C  # Scale by C for consistency
    else:
        raise ValueError("method must be 'rev_kl', 'js', or 'l2'")

    # Soft barrier box around 1/C to keep every channel alive
    if min_frac is None: min_frac = 0.5 / C     # e.g. 50% of target
    if max_frac is None: max_frac = 2.0 / C     # e.g. 200% of target
    barrier = (torch.relu(min_frac - frac).pow(2) + torch.relu(frac - max_frac).pow(2)).sum()
    loss = loss + barrier_w * barrier
    
    # Add entropy regularization to encourage decisive assignments
    if entropy_weight > 0 and use_straight_through:
        # Compute entropy at vertex level (before ST)
        p_soft = torch.softmax(beta_area * F, dim=1)
        vertex_entropy = -(p_soft * (p_soft + eps).log()).sum(dim=1).mean()
        # Lower entropy = more decisive assignments
        loss = loss + entropy_weight * vertex_entropy

    return loss, frac


def compute_boundary_stats_v2(F: torch.Tensor,
                             edge_idx: torch.Tensor,
                             verts: torch.Tensor,
                             beta_contour: float) -> Tuple[float, float, float]:
    """
    Compute boundary length and active edge fraction for monitoring (v2 version).
    Uses soft-OR coverage from all pairs instead of top-k gating.
    
    Args:
        F: (N, C) multi-channel field values
        edge_idx: (E, 2) edge vertex indices
        verts: (N, 3) vertex positions
        beta_contour: Temperature parameter
        
    Returns:
        length: Estimated boundary length
        active_fraction: Fraction of edges with φ > 0.5
        median_phi: Median coverage value
    """
    device = F.device
    C = F.shape[1]
    
    # Ensure edge_idx is on same device
    edge_idx = edge_idx.to(device)
    
    # all channel pairs
    ii, jj = torch.triu_indices(C, C, offset=1, device=device)
    
    # values at endpoints for all pairs
    va, vb = edge_idx.T
    da = F[va][:, ii] - F[va][:, jj]      # (E,P)
    db = F[vb][:, ii] - F[vb][:, jj]      # (E,P)
    
    # crossing weights for all pairs
    w_pairs = torch.sigmoid(-beta_contour * da * db)
    conf = 0.5*(da.abs() + db.abs())
    w_pairs = w_pairs * torch.sigmoid(5.0*(conf - 0.2))
    
    # soft-OR coverage per edge
    phi = 1.0 - torch.prod(1.0 - w_pairs, dim=1)  # (E,)
    
    # Active edges (φ > 0.5)
    active_mask = phi > 0.5
    active_fraction = active_mask.float().mean().item()
    median_phi = phi.median().item()
    
    # Boundary length (sum of active edge lengths)
    if active_mask.any():
        active_edges = edge_idx[active_mask]
        edge_lengths = (verts[active_edges[:, 0]] - verts[active_edges[:, 1]]).norm(dim=1)
        length = edge_lengths.sum().item()
    else:
        length = 0.0
    
    return length, active_fraction, median_phi


def compute_hard_area_fractions(
    F: torch.Tensor,
    faces: torch.Tensor,
    tri_area: torch.Tensor
) -> torch.Tensor:
    """
    Compute area fractions based on hard argmax assignments.
    This shows what's actually happening in the visualization.
    
    Args:
        F: (N, C) multi-channel field values
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        
    Returns:
        frac_hard: (C,) hard area fractions
    """
    # Hard assignments at vertices
    labels = F.argmax(dim=1)  # (N,)
    hard_v = torch.zeros_like(F).scatter_(1, labels[:, None], 1.0)  # (N, C)
    
    # Face assignments (majority vote or average)
    hard_f = (hard_v[faces[:, 0]] + hard_v[faces[:, 1]] + hard_v[faces[:, 2]]) / 3.0  # (T, C)
    
    # Area per channel
    A_hard = (tri_area[:, None] * hard_f).sum(0)  # (C,)
    total = A_hard.sum().clamp_min(1e-12)
    frac_hard = A_hard / total
    
    return frac_hard


def margin_separation_loss(
    F: torch.Tensor,
    tau: float = 0.5
) -> torch.Tensor:
    """
    Encourage separation between top channels at each vertex.
    This helps break symmetry and form distinct regions.
    
    Args:
        F: (N, C) field values
        tau: Minimum margin between top-2 channels
        
    Returns:
        loss: Margin separation loss
    """
    # Get top 2 values at each vertex
    top2, _ = torch.topk(F, k=min(2, F.shape[1]), dim=1)
    
    if top2.shape[1] < 2:
        return F.new_tensor(0.0)
    
    # Margin between top 2
    margin = top2[:, 0] - top2[:, 1]
    
    # Penalize small margins
    loss = torch.relu(tau - margin).mean()
    
    return loss


ChatGPT said:
Request for GPT-5 Pro
Reasoned for 5m 12s
absolutely—here’s a cleaned, stability‑first rewrite that packages the whole thing into one drop‑in loss module. it keeps the spirit of your “intrinsic (no SVD) contour alignment + cotan smoothness + KL/JS area + soft pins” idea, but adds a bunch of convergence guards (pair‑mixing with stop‑grad, Charbonnier penalties, scale‑invariant normalizations, degenerate‑triangle masking, safe dtype casting, and weight scheduling). it’s designed to be plug‑and‑play with your training loop and meshes. the design choices line up with what you described in your report (esp. Ch. 4.3–4.4)

what’s different / why it’s stable
Intrinsic, pair‑mixed contour alignment: all channel pairs per edge, soft‑OR coverage, 3D tangents from intrinsic gradients; pair mixing uses stop‑grad so the model can’t “cheat” by zeroing hard pairs. Charbonnier penalty, grad/length gates, and activity floor keep gradients alive even early in training.

Cotan Laplacian smoothness: edge‑weighted and normalized by total weight → scale‑invariant.

Area balance: reverse‑KL (or JS) to uniform; optional straight‑through counting with entropy regularization. box barrier to keep every class alive early.

Soft pins (Huber): no hard projection; schedule the weight.

Triple‑point barrier: gently separates 2nd vs 3rd probs per face to avoid “Y‑junctions everywhere.”

Schedules + guards: β (temperature) anneal, component weight ramps, gradient clipping, mixed precision friendliness, consistent eps, FP64 for small linear solves, boundary edges safely ignored in contour term.

code: patch_losses.py
python
Copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# --------------------------- utils -------------------------------------------

def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

def stopgrad(x: torch.Tensor) -> torch.Tensor:
    # convenient alias: treat as constant in backward
    return x.detach()

@torch.no_grad()
def _sorted_edges_of_faces(faces: torch.Tensor) -> torch.Tensor:
    # faces: (T,3) -> edges unique (E,2), both sorted per edge and unique across mesh
    i0, i1, i2 = faces[:,0], faces[:,1], faces[:,2]
    e01 = torch.stack([torch.minimum(i0,i1), torch.maximum(i0,i1)], dim=1)
    e12 = torch.stack([torch.minimum(i1,i2), torch.maximum(i1,i2)], dim=1)
    e20 = torch.stack([torch.minimum(i2,i0), torch.maximum(i2,i0)], dim=1)
    edges = torch.cat([e01, e12, e20], dim=0)
    edges = torch.unique(edges, dim=0)
    return edges

@torch.no_grad()
def build_edge_tris(faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (edges, edge_tris):
      edges     : (E,2) sorted vertex ids
      edge_tris : (E,2) triangle ids, -1 in the second slot for boundary edges
    """
    device = faces.device
    T = faces.shape[0]
    i0, i1, i2 = faces[:,0], faces[:,1], faces[:,2]

    def pack_edges(a,b,t):
        a_, b_ = torch.minimum(a,b), torch.maximum(a,b)
        return torch.stack([a_, b_, t], dim=1)  # (T,3)

    E01 = pack_edges(i0, i1, torch.arange(T, device=device))
    E12 = pack_edges(i1, i2, torch.arange(T, device=device))
    E20 = pack_edges(i2, i0, torch.arange(T, device=device))
    all_edges = torch.cat([E01, E12, E20], dim=0)  # (3T,3)

    # sort by (v_low, v_high)
    idx = torch.lexsort((all_edges[:,1], all_edges[:,0]))
    se = all_edges[idx]  # sorted edges: (v_low, v_high, tri)

    # sweep to pair triangles
    v0, v1, tri = se[:,0], se[:,1], se[:,2]
    # find starts of runs where (v0,v1) changes
    run_break = torch.ones_like(v0, dtype=torch.bool)
    run_break[1:] = (v0[1:] != v0[:-1]) | (v1[1:] != v1[:-1])

    starts = torch.nonzero(run_break, as_tuple=False).flatten()
    ends   = torch.cat([starts[1:], torch.tensor([len(se)], device=device)])

    edges, et = [], []
    for s, e in zip(starts.tolist(), ends.tolist()):
        block = se[s:e]          # all occurrences of this edge (1 or 2 tris)
        a, b = block[0,0].item(), block[0,1].item()
        if block.shape[0] == 1:
            edges.append((a,b))
            et.append((block[0,2].item(), -1))
        else:
            edges.append((a,b))
            et.append((block[0,2].item(), block[1,2].item()))
    edges     = torch.tensor(edges, device=device, dtype=torch.long)
    edge_tris = torch.tensor(et, device=device, dtype=torch.long)
    return edges, edge_tris

@torch.no_grad()
def face_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0, v1, v2 = vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]]
    return 0.5 * torch.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1)

@torch.no_grad()
def cotan_laplacian_weights(vertices: torch.Tensor, faces: torch.Tensor):
    """
    Returns I,J,W for sum_{(i,j)} W_ij ||F_i - F_j||^2.
    """
    v0, v1, v2 = faces[:,0], faces[:,1], faces[:,2]
    p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]

    e0 = p1 - p2; e1 = p2 - p0; e2 = p0 - p1  # edges opposite to v0,v1,v2
    # cot(alpha) = <u,v> / ||u x v||
    def cot(u, v):
        num = (u * v).sum(dim=1)
        den = torch.cross(u, v, dim=1).norm(dim=1).clamp_min(1e-12)
        return num / den

    c0 = cot(p1 - p0, p2 - p0)  # at v0
    c1 = cot(p2 - p1, p0 - p1)  # at v1
    c2 = cot(p0 - p2, p1 - p2)  # at v2

    # accumulate symmetric weights: w_ij += (cot_k + cot_l)/2
    I = torch.cat([v1, v2, v2, v0, v0, v1])
    J = torch.cat([v2, v1, v0, v2, v1, v0])
    W = 0.5 * torch.cat([c0, c0, c1, c1, c2, c2])  # shares around the triangle

    # coalesce duplicates
    K = torch.stack([I, J], dim=1)
    K, idx = torch.unique(K, dim=0, return_inverse=True)
    W_coalesced = torch.zeros(K.shape[0], device=vertices.device, dtype=vertices.dtype)
    W_coalesced.index_add_(0, idx, W)
    I, J = K[:,0], K[:,1]
    return I.long(), J.long(), W_coalesced

def grad3d_intrinsic(h_vals: torch.Tensor,
                     v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    triangle-wise intrinsic gradient in R^3 via Gram solve.
    h_vals: (B,3)   values at (v0,v1,v2)
    v0,v1,v2: (B,3) vertex positions
    returns g: (B,3)
    """
    e0 = v1 - v0
    e1 = v2 - v0
    b  = torch.stack([h_vals[:,1]-h_vals[:,0],
                      h_vals[:,2]-h_vals[:,0]], dim=1)   # (B,2)

    dt = torch.float64
    e0d, e1d, bd = e0.to(dt), e1.to(dt), b.to(dt)
    G00 = (e0d*e0d).sum(dim=1); G01 = (e0d*e1d).sum(dim=1); G11 = (e1d*e1d).sum(dim=1)
    det = (G00*G11 - G01*G01).clamp_min(1e-12)
    invG00 =  G11 / det
    invG01 = -G01 / det
    invG11 =  G00 / det

    a0 = invG00*bd[:,0] + invG01*bd[:,1]
    a1 = invG01*bd[:,0] + invG11*bd[:,1]
    gd = a0[:,None]*e0d + a1[:,None]*e1d
    return torch.nan_to_num(gd.to(h_vals.dtype))

# --------------------------- losses ------------------------------------------

def contour_alignment_intrinsic_v2_detached(
    F: torch.Tensor, faces: torch.Tensor, edges: torch.Tensor, edge_tris: torch.Tensor,
    verts: torch.Tensor, beta_contour: float = 6.0, eps: float = 1e-6,
    return_weights: bool = False
):
    """
    Robust intrinsic contour alignment (no SVD):
      - all channel pairs per edge (soft-OR coverage φ)
      - per-edge pair-mixing with stop-grad to prevent 'pair cherry-picking'
      - 3D tangents: tau = n x grad(h) (intrinsic)
      - Charbonnier penalty with grad/length gating
    """
    device = F.device
    N, C = F.shape

    valid = (edge_tris[:,0] >= 0) & (edge_tris[:,1] >= 0)
    if not torch.any(valid):
        z = F.new_tensor(0.0)
        return (z, torch.zeros(edges.shape[0], device=device, dtype=F.dtype)) if return_weights else z

    va, vb = edges[valid,0], edges[valid,1]
    tL, tR = edge_tris[valid,0], edge_tris[valid,1]

    ii, jj = torch.triu_indices(C, C, offset=1, device=device)           # P pairs
    da = F[va][:, ii] - F[va][:, jj]                                     # (E,P)
    db = F[vb][:, ii] - F[vb][:, jj]                                     # (E,P)

    w_pairs = torch.sigmoid(-beta_contour * da * db)                      # sign change
    conf    = 0.5*(da.abs() + db.abs())
    w_pairs = w_pairs * torch.sigmoid(5.0*(conf - 0.2))                   # confidence gate
    w_pairs = w_pairs.clamp_min(1e-6)

    # pair mixing weights (stop-grad to avoid minimizing via trivial pair selection)
    pair_mix = stopgrad(w_pairs) / (stopgrad(w_pairs).sum(dim=1, keepdim=True) + eps)  # (E,P)
    # soft-OR coverage (gradient flows here)
    phi = 1.0 - torch.prod(1.0 - w_pairs, dim=1)                          # (E,)

    faces_L = faces[tL]; faces_R = faces[tR]
    F_L     = F[faces_L]  # (E,3,C)
    F_R     = F[faces_R]

    hL = F_L[:,:,ii] - F_L[:,:,jj]                                       # (E,3,P)
    hR = F_R[:,:,ii] - F_R[:,:,jj]

    v0L, v1L, v2L = [verts[faces_L[:,k]] for k in (0,1,2)]
    v0R, v1R, v2R = [verts[faces_R[:,k]] for k in (0,1,2)]

    def grad_many(h, v0, v1, v2):
        E, _, P = h.shape
        h_flat = h.permute(0,2,1).reshape(E*P, 3)
        g = grad3d_intrinsic(h_flat, v0.repeat_interleave(P,0),
                                       v1.repeat_interleave(P,0),
                                       v2.repeat_interleave(P,0))
        return g.reshape(E, P, 3)

    gL = grad_many(hL, v0L, v1L, v2L)                                     # (E,P,3)
    gR = grad_many(hR, v0R, v1R, v2R)

    def tri_normals(a,b,c):
        n = torch.cross(b - a, c - a, dim=1)
        return safe_normalize(n, dim=1, eps=1e-6)

    nL = tri_normals(v0L, v1L, v2L)                                       # (E,3)
    nR = tri_normals(v0R, v1R, v2R)

    def proj_in_plane(g, n):
        dot = (g * n[:,None,:]).sum(dim=2, keepdim=True)
        return g - dot * n[:,None,:]

    gL = proj_in_plane(gL, nL); gR = proj_in_plane(gR, nR)
    tauL = safe_normalize(torch.cross(nL[:,None,:], gL, dim=2), dim=2, eps=1e-6)
    tauR = safe_normalize(torch.cross(nR[:,None,:], gR, dim=2), dim=2, eps=1e-6)

    cosang = (tauL * tauR).sum(dim=2).abs().clamp_max(1.0)               # (E,P)
    mis    = 1.0 - cosang                                                # (E,P)
    mis_edge = (pair_mix * mis).sum(dim=1)                               # (E,)

    # gates
    gLm = gL.norm(dim=2).mean(dim=1) + eps
    gRm = gR.norm(dim=2).mean(dim=1) + eps
    grad_gate = torch.sqrt(gLm * gRm)
    scale = torch.nanmedian(grad_gate).detach().clamp_min(1e-6)
    grad_gate = torch.nan_to_num((grad_gate/scale).clamp(0.0, 2.0), nan=0.0, posinf=2.0, neginf=0.0)

    edge_vec = verts[vb] - verts[va]
    edge_len = edge_vec.norm(dim=1)
    len_gate = (edge_len / (edge_len.mean().detach() + 1e-12)).clamp(0.5, 2.0)

    w_edge = (phi.clamp_min(1e-4) * grad_gate * len_gate)

    loss_edge = torch.sqrt(mis_edge*mis_edge + 1e-6)
    num = (w_edge * loss_edge).sum()
    den = w_edge.sum().clamp_min(1e-9)
    loss = num / den

    if return_weights:
        full_phi = torch.zeros(edges.shape[0], device=device, dtype=F.dtype)
        full_phi[valid] = phi
        return loss, full_phi
    return loss

def smoothness_cotan(F: torch.Tensor, I: torch.Tensor, J: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    diff2 = (F[I] - F[J]).pow(2).sum(dim=1)   # (K,)
    return (W * diff2).sum() / W.sum().clamp_min(1e-12)

def area_balance(
    F: torch.Tensor, faces: torch.Tensor, tri_area: torch.Tensor, beta_area: float,
    method: str = "rev_kl", use_st: bool = True, entropy_w: float = 0.0,
    eps: float = 1e-8, min_frac: Optional[float] = None, max_frac: Optional[float] = None, barrier_w: float = 0.05
):
    N, C = F.shape
    p = torch.softmax(beta_area * F, dim=1)
    if use_st:
        hard = torch.zeros_like(p).scatter_(1, p.argmax(dim=1, keepdim=True), 1.0)
        p = hard - p.detach() + p  # straight-through

    p_f = (p[faces[:,0]] + p[faces[:,1]] + p[faces[:,2]]) / 3.0  # (T,C)

    A_c = (tri_area[:,None] * p_f).sum(dim=0)                    # (C,)
    total = tri_area.sum().clamp_min(eps)
    frac = (A_c / total).clamp(eps, 1.0-eps)                     # (C,)

    uniform = F.new_full((C,), 1.0/C)
    if method == "rev_kl":      # -sum (1/C) log frac
        loss = -(uniform * frac.log()).sum()
    elif method == "js":
        m = 0.5*(frac + uniform)
        loss = 0.5 * ((frac * (frac.add(eps).log() - m.add(eps).log())).sum()
                    + (uniform * (uniform.add(eps).log() - m.add(eps).log())).sum())
    elif method == "l2":
        loss = ((frac - uniform)**2).sum() * C
    else:
        raise ValueError("method should be 'rev_kl', 'js', or 'l2'")

    if min_frac is None: min_frac = 0.5 / C
    if max_frac is None: max_frac = 2.0 / C
    barrier = (torch.relu(min_frac - frac).pow(2) + torch.relu(frac - max_frac).pow(2)).sum()
    loss = loss + barrier_w * barrier

    if entropy_w > 0 and use_st:
        p_soft = torch.softmax(beta_area * F, dim=1)
        ent = -(p_soft * (p_soft + eps).log()).sum(dim=1).mean()
        loss = loss + entropy_w * ent
    return loss, frac

def pin_huber(F: torch.Tensor, pin_idx: torch.Tensor, pin_target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    if pin_idx.numel() == 0: return F.new_tensor(0.0)
    diff = F[pin_idx] - pin_target
    absd = diff.abs()
    quad = torch.minimum(absd, diff.new_tensor(delta))
    lin  = absd - quad
    return (0.5*quad**2 + delta*lin).mean()

def triple_point_barrier(F: torch.Tensor, faces: torch.Tensor, tri_area: torch.Tensor, beta: float = 8.0, margin: float = 0.10):
    p = torch.softmax(beta * F, dim=1)
    p_t = (p[faces[:,0]] + p[faces[:,1]] + p[faces[:,2]])/3
    top3, _ = torch.topk(p_t, k=min(3, F.shape[1]), dim=1)
    if top3.shape[1] < 3: return F.new_tensor(0.0)
    gap = top3[:,1] - top3[:,2]
    pen = torch.relu(margin - gap)
    w = tri_area / (tri_area.sum().clamp_min(1e-9))
    return (w * pen).sum()

def potts_on_probs(F: torch.Tensor, edges: torch.Tensor, edge_weights: torch.Tensor, beta: float = 8.0, gamma: float = 2.0):
    p = torch.softmax(beta * F, dim=1)
    va, vb = edges[:,0], edges[:,1]
    potts = 1.0 - (p[va] * p[vb]).sum(dim=1)              # = 0.5||p_i - p_j||^2
    gate  = (1.0 - stopgrad(edge_weights)).clamp(0,1).pow(gamma)
    return (gate * potts).sum() / (gate.sum().clamp_min(1e-9))

# --------------------------- config + module ---------------------------------

@dataclass
class LossConfig:
    # temperatures
    beta_contour0: float = 3.0
    beta_contourT: float = 12.0
    beta_area0:    float = 2.0
    beta_areaT:    float = 40.0

    # weights (final targets; ramp from 0 except smooth)
    w_contour: float = 1.0
    w_smooth:  float = 0.2
    w_area:    float = 2.0
    w_potts:   float = 0.1
    w_pin:     float = 0.25
    w_triple:  float = 0.05

    # ramps
    ramp_frac: float = 0.2   # first 20% of iters ramp contour/area/pin/triple
    method_area: str = "rev_kl"
    use_straight_through: bool = True
    entropy_w: float = 0.0   # small (e.g. 0.01) if you want sharper decisions later

class PatchLoss(nn.Module):
    """
    Drop‑in combined loss with robust, intrinsic contour alignment (no SVD),
    cotan smoothness, area balance (KL/JS), soft pins, triple barrier, and Potts.
    """
    def __init__(self,
                 vertices: torch.Tensor,
                 faces: torch.Tensor,
                 pin_idx: Optional[torch.Tensor] = None,
                 pin_target: Optional[torch.Tensor] = None,
                 cfg: LossConfig = LossConfig()):
        super().__init__()
        self.register_buffer('V', vertices)
        self.register_buffer('F', faces.long())
        self.cfg = cfg

        # precompute topology & weights
        edges, edge_tris = build_edge_tris(self.F)
        I, J, W = cotan_laplacian_weights(self.V, self.F)
        A = face_areas(self.V, self.F)

        self.register_buffer('edges', edges)
        self.register_buffer('edge_tris', edge_tris)
        self.register_buffer('cot_I', I); self.register_buffer('cot_J', J); self.register_buffer('cot_W', W)
        self.register_buffer('tri_area', A)

        if pin_idx is None:
            pin_idx = torch.zeros(0, dtype=torch.long, device=vertices.device)
            pin_target = torch.zeros(0, vertices.shape[1] if pin_idx.numel() else 0,
                                     device=vertices.device, dtype=vertices.dtype)
        self.register_buffer('pin_idx', pin_idx.long())
        self.register_buffer('pin_target', pin_target)

    def _sched(self, it: Optional[int], T: Optional[int], v0: float, vT: float):
        if it is None or T is None: return vT
        t = min(max(it,0), T)
        return v0 + (vT - v0) * (t / float(T))

    def _ramp(self, it: Optional[int], T: Optional[int]):
        if it is None or T is None: return 1.0
        R = int(max(1, self.cfg.ramp_frac * T))
        return min(1.0, it / float(R))

    def forward(self, F_field: torch.Tensor, it: Optional[int] = None, T: Optional[int] = None):
        # schedules
        beta_c = self._sched(it, T, self.cfg.beta_contour0, self.cfg.beta_contourT)
        beta_a = self._sched(it, T, self.cfg.beta_area0,    self.cfg.beta_areaT)
        r = self._ramp(it, T)

        w_contour = self.cfg.w_contour * r
        w_area    = self.cfg.w_area    * r
        w_pin     = self.cfg.w_pin     * r
        w_triple  = self.cfg.w_triple  * r
        w_potts   = self.cfg.w_potts   * r
        w_smooth  = self.cfg.w_smooth  # no ramp so the field stays tame

        # contour (and boundary activity for Potts gate)
        loss_contour, phi = contour_alignment_intrinsic_v2_detached(
            F_field, self.F, self.edges, self.edge_tris, self.V,
            beta_contour=beta_c, return_weights=True
        )

        # smoothness (cotan)
        loss_smooth = smoothness_cotan(F_field, self.cot_I, self.cot_J, self.cot_W)

        # area balance
        loss_area, frac = area_balance(
            F_field, self.F, self.tri_area, beta_area=beta_a,
            method=self.cfg.method_area, use_st=self.cfg.use_straight_through,
            entropy_w=self.cfg.entropy_w
        )

        # extras
        loss_pin = pin_huber(F_field, self.pin_idx, self.pin_target) if self.pin_idx.numel() else F_field.new_tensor(0.0)
        loss_trp = triple_point_barrier(F_field, self.F, self.tri_area, beta=self.cfg.beta_areaT)
        loss_pot = potts_on_probs(F_field, self.edges, phi, beta=beta_a, gamma=2.0)

        total = (w_contour * loss_contour +
                 w_smooth  * loss_smooth  +
                 w_area    * loss_area    +
                 w_pin     * loss_pin     +
                 w_triple  * loss_trp     +
                 w_potts   * loss_pot)

        stats = dict(
            total=float(total.detach()),
            contour=float(loss_contour.detach()),
            smooth=float(loss_smooth.detach()),
            area=float(loss_area.detach()),
            pin=float(loss_pin.detach()),
            triple=float(loss_trp.detach()),
            potts=float(loss_pot.detach()),
            beta_contour=float(beta_c),
            beta_area=float(beta_a),
            ramp=float(r),
            area_frac=frac.detach().cpu().numpy()
        )
        return total, stats, phi
usage snippet
python
Copy
# mesh tensors (on device)
V = vertices.to(device)       # (N,3)
F = faces.to(device).long()   # (T,3)

# optional pins
pin_idx    = torch.tensor(pinned_indices, device=device).long()   # (P,)
pin_target = torch.full((len(pinned_indices), 6), -1.0, device=device)
pin_target[torch.arange(len(pinned_indices)), torch.arange(len(pinned_indices))] = 1.0

cfg = LossConfig(
    beta_contour0=3.0, beta_contourT=12.0,
    beta_area0=2.0,    beta_areaT=40.0,
    w_contour=1.5, w_smooth=0.2, w_area=2.0, w_potts=0.1, w_pin=0.25, w_triple=0.05,
    ramp_frac=0.25, method_area="rev_kl", use_straight_through=True, entropy_w=0.0
)

loss_mod = PatchLoss(V, F, pin_idx, pin_target, cfg).to(device)

# training loop
opt = torch.optim.AdamW([f_param], lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.3)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
for it in range(steps):
    with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
        loss, stats, phi = loss_mod(f_param, it=it, T=steps)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(f_param, 5.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()

    # (optional) re‑pin softly, or keep letting pin_huber do it
    # f_param.data[pin_idx] = pin_target  # <- only if you really need hard pins late

    if (it+1) % 100 == 0:
        print(f"it {it+1:05d} | L={stats['total']:.3e} | Lc={stats['contour']:.3e} La={stats['area']:.3e} Ls={stats['smooth']:.3e} βc={stats['beta_contour']:.1f} βa={stats['beta_area']:.1f}")
quick notes & knobs
Pair mixing with stop‑grad is the biggest stability win: the contour term can’t minimize by sliding all weight to a convenient pair—misalignment needs to genuinely drop across pairs.

Charbonnier everywhere when distances get small avoids huge gradients near kinks.

Scale‑free normalizations: contour is averaged by weight sum; cotan by ∑W; area by total surface area.

Degenerate tri masking is hidden inside grad3d_intrinsic + projections.

Ramps: keep w_smooth on from step 0; ramp in contour/area/pins/triple over the first 20–25% iterations; anneal β’s linearly (or cosine) to make decisions crisp late.

Potts on probs helps de‑speckle interiors away from boundary; it’s softly gated by phi.

if you want, i can wire this straight into your existing optimization() loop and swap out the older plane‑SVD loss; or we can expose toggles to compare SVD vs intrinsic on the exact same schedule.






You said:
"""
Progress-gated temperature scheduling and optimization utilities.
Addresses the issue of premature hardening (report sections 4.4.3, 4.4.5).
"""
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np


@dataclass
class TempController:
    """
    Progress-gated temperature controller.
    Instead of time-based ramping, increase beta based on optimization progress.
    """
    # Current temperatures (start moderate for meaningful gradients)
    beta_contour: float = 1.0  # Start at 1 for contour
    beta_area: float = 2.0     # Start higher for area to get meaningful KL divergence
    
    # Maximum temperatures (further reduced for stability)
    beta_contour_max: float = 8.0   # Reduced from 12.0 - keep ≤ 2 early on
    beta_area_max: float = 4.0      # Reduced from 6.0 - keep ≤ 1.5 early on
    
    # Step sizes for increases (even smaller steps)
    step_up_contour: float = 0.3   # Reduced from 0.5
    step_up_area: float = 0.1      # Reduced from 0.15
    
    # Progress thresholds
    tau_area: float = 0.05  # Stricter - only increase β when area deviation < 0.05
    min_boundary_fraction: float = 0.02  # Lower threshold
    contour_improve_threshold: float = 0.005  # Reduced to allow more β increases
    
    # Cooldown tracking
    last_beta_update_step: int = -10000
    min_steps_between_updates: int = 1000  # Reduced cooldown for faster β ramping
    best_contour_loss_since_update: float = float('inf')
    
    # History tracking
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        'beta_contour': [],
        'beta_area': [],
        'area_deviation': [],
        'boundary_length': []
    })
    
    def check_and_update(self, 
                        area_fractions: torch.Tensor,
                        boundary_length: float,
                        bbox_diagonal: float,
                        step: int,
                        contour_loss: float) -> bool:
        """
        Check progress and potentially increase temperatures.
        Now requires contour improvement and enforces cooldown.
        
        Args:
            area_fractions: (C,) current area distribution
            boundary_length: Estimated boundary length
            bbox_diagonal: Mesh bounding box diagonal
            step: Current optimization step
            contour_loss: Current contour alignment loss
            
        Returns:
            updated: Whether temperatures were increased
        """
        # Compute area deviation from uniform
        uniform = 1.0 / area_fractions.shape[0]
        deviation = (area_fractions - uniform).abs().max().item()
        
        # Normalized boundary length
        boundary_fraction = boundary_length / bbox_diagonal
        
        # Store history
        self.history['area_deviation'].append(deviation)
        self.history['boundary_length'].append(boundary_fraction)
        self.history['beta_contour'].append(self.beta_contour)
        self.history['beta_area'].append(self.beta_area)
        
        # Check cooldown
        if step - self.last_beta_update_step < self.min_steps_between_updates:
            # Update best contour loss in this window
            self.best_contour_loss_since_update = min(
                self.best_contour_loss_since_update, contour_loss
            )
            return False
        
        # Handle first call or invalid best loss
        if not torch.isfinite(torch.tensor(self.best_contour_loss_since_update)):
            self.best_contour_loss_since_update = float(contour_loss)
        
        # Check improvement
        improvement = (self.best_contour_loss_since_update - contour_loss) / max(self.best_contour_loss_since_update, 1e-9)
        has_improved = improvement >= self.contour_improve_threshold
        
        # Adaptive thresholds based on current beta
        # Early on (low beta), be more lenient with area deviation
        adaptive_tau = self.tau_area * (1.0 + max(0, 5.0 - self.beta_contour) * 0.1)
        
        # Check all conditions
        updated = False
        if (has_improved and
            deviation < adaptive_tau and 
            boundary_fraction > self.min_boundary_fraction):
            
            # Adaptive step sizes - smaller steps at higher temperatures
            contour_step = self.step_up_contour * (1.0 - self.beta_contour / self.beta_contour_max * 0.5)
            area_step = self.step_up_area * (1.0 - self.beta_area / self.beta_area_max * 0.5)
            
            # Increase contour beta
            if self.beta_contour < self.beta_contour_max:
                self.beta_contour = min(
                    self.beta_contour + contour_step,
                    self.beta_contour_max
                )
                updated = True
            
            # Increase area beta (slower)
            if self.beta_area < self.beta_area_max:
                self.beta_area = min(
                    self.beta_area + area_step,
                    self.beta_area_max
                )
                updated = True
            
            if updated:
                self.last_beta_update_step = step
                self.best_contour_loss_since_update = float('inf')  # Reset
        else:
            # Track best loss in window
            self.best_contour_loss_since_update = min(
                self.best_contour_loss_since_update, contour_loss
            )
        
        return updated


@dataclass 
class OptimizationStage:
    """Configuration for a single optimization stage."""
    name: str
    start_fraction: float  # When to start (fraction of total steps)
    end_fraction: float    # When to end
    
    # Loss weights
    lambda_smooth: float = 1.0
    lambda_contour: float = 0.1
    lambda_area: float = 0.1
    lambda_pin: float = 0.01
    lambda_tv: Optional[float] = None  # Optional total variation
    
    # Learning rate
    lr: float = 1e-3
    cosine_decay: bool = False  # Use cosine decay for this stage
    
    # Pin constraint
    use_hard_pins: bool = False
    
    # Temperature control
    allow_temp_increase: bool = True


class TwoStageScheduler:
    """
    Two-stage optimization scheduler.
    Stage A: Coarse segmentation with low beta, strong smoothness
    Stage B: Refinement with higher beta, stronger alignment
    """
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        
        # Define stages with better transitions
        # Adjusted for 300k steps with smoothness-only warmup
        self.stages = [
            OptimizationStage(
                name="Stage 0: Smoothness + Area + Normal Warmup",
                start_fraction=0.0,
                end_fraction=0.0167,  # First 5k steps (5k/300k)
                lambda_smooth=1.0,  # High smoothness to form patches
                lambda_contour=0.001,  # Almost zero - no boundary alignment yet
                lambda_area=3.0,    # Very high area balance to ensure equal patches
                lambda_pin=0.0,    # No soft pin loss needed with hard pins
                lambda_tv=0.1,     # Strong TV for initial smoothing
                lr=5e-4,           # High LR to quickly form smooth patches
                use_hard_pins=True,  # Hard pins to anchor the field properly
                allow_temp_increase=False  # No temperature increase during warmup
            ),
            OptimizationStage(
                name="Stage A1: Early",
                start_fraction=0.0167,  # Start after 5k warmup
                end_fraction=0.06,  # 5k-18k steps
                lambda_smooth=0.5,  # Still strong smoothness
                lambda_contour=0.01,  # Very low contour to start
                lambda_area=1.0,    # Strong area balance to fight collapse
                lambda_pin=1.0,     # High pin weight to maintain anchors
                lambda_tv=0.05,     # Reduce TV
                lr=3e-5,            # Further reduced LR for stability
                use_hard_pins=False,
                allow_temp_increase=True
            ),
            OptimizationStage(
                name="Stage A2: Coarse",
                start_fraction=0.06,
                end_fraction=0.15,  # 18k-45k steps
                lambda_smooth=0.4,  # Still strong
                lambda_contour=0.02,  # Very gradual increase
                lambda_area=0.3,    # Moderate area balance
                lambda_pin=0.5,     # Moderate pin weight
                lambda_tv=0.02,     # Reduce TV
                lr=5e-4,
                use_hard_pins=False,
                allow_temp_increase=True
            ),
            OptimizationStage(
                name="Stage A3: Shape Formation",
                start_fraction=0.15,
                end_fraction=0.4,  # 45k-120k steps
                lambda_smooth=0.3,  
                lambda_contour=0.05,  # Still low base - let stall detector handle increases
                lambda_area=0.8,    # Strong area balance
                lambda_pin=0.01,
                lambda_tv=None,     # Drop TV
                lr=3e-4,
                use_hard_pins=False,
                allow_temp_increase=True
            ),
            OptimizationStage(
                name="Stage B: Refinement", 
                start_fraction=0.4,
                end_fraction=0.8,  # 120k-240k steps
                lambda_smooth=0.2,
                lambda_contour=0.1,  # Moderate base - stall detector will increase as needed
                lambda_area=0.5,   # Balanced
                lambda_pin=0.02,
                lambda_tv=None,     # No TV
                lr=2e-4,
                cosine_decay=True,  # Use cosine annealing
                use_hard_pins=False,
                allow_temp_increase=True
            ),
            OptimizationStage(
                name="Stage C: Final",
                start_fraction=0.8,
                end_fraction=1.0,  # 240k-300k steps
                lambda_smooth=0.1,
                lambda_contour=0.2,  # Base value - can grow via stall detector
                lambda_area=0.3,    # Keep some area balance
                lambda_pin=0.1,
                lambda_tv=None,
                lr=1e-4,
                use_hard_pins=True,  # Hard pins only at the end
                allow_temp_increase=True  # Allow final sharpening
            )
        ]
        
        self.current_stage_idx = 0
        
    def get_stage(self, step: int) -> OptimizationStage:
        """Get current optimization stage based on step."""
        fraction = step / self.total_steps
        
        # Find appropriate stage
        for i, stage in enumerate(self.stages):
            if stage.start_fraction <= fraction < stage.end_fraction:
                if i != self.current_stage_idx:
                    print(f"\n=== Switching to {stage.name} ===")
                    self.current_stage_idx = i
                return stage
        
        # Default to last stage
        return self.stages[-1]
    
    def get_lr(self, step: int, base_lr: float) -> float:
        """Get learning rate for current step with minimum clamp."""
        stage = self.get_stage(step)
        lr = stage.lr
        
        # Apply decay if specified (cosine schedule)
        if stage.cosine_decay:
            stage_start = int(stage.start_fraction * self.total_steps)
            stage_end = int(stage.end_fraction * self.total_steps)
            steps_in_stage = step - stage_start
            total_stage_steps = stage_end - stage_start
            
            # Cosine annealing within stage
            progress = min(steps_in_stage / max(total_stage_steps, 1), 1.0)
            min_lr = 1e-5  # Minimum learning rate
            lr = min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Ensure minimum learning rate
        return max(lr, 1e-5)


class GradientMonitor:
    """
    Monitor gradient statistics to detect vanishing/exploding gradients.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = {
            'total': [],
            'smooth': [],
            'contour': [],
            'area': [],
            'pin': []
        }
        
    def log_gradients(self, F: torch.Tensor, loss_grads: Dict[str, torch.Tensor]):
        """Log gradient norms for each loss component."""
        # Total gradient norm
        total_grad_norm = F.grad.norm().item() if F.grad is not None else 0.0
        self.history['total'].append(total_grad_norm)
        
        # Per-loss gradient norms (if available)
        for name, grad in loss_grads.items():
            if grad is not None:
                self.history[name].append(grad.norm().item())
        
        # Keep only recent history
        for key in self.history:
            if len(self.history[key]) > self.window_size:
                self.history[key] = self.history[key][-self.window_size:]
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get gradient statistics."""
        stats = {}
        
        for name, values in self.history.items():
            if values:
                values_array = np.array(values)
                stats[name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'median': float(np.median(values_array))
                }
            else:
                stats[name] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        
        return stats
    
    def check_health(self) -> Dict[str, bool]:
        """Check for gradient issues."""
        health = {}
        
        if self.history['total']:
            recent = np.array(self.history['total'][-20:])  # Last 20 steps
            
            # Check for vanishing gradients
            health['vanishing'] = np.mean(recent) < 1e-7
            
            # Check for exploding gradients  
            health['exploding'] = np.max(recent) > 1e3
            
            # Check for high variance
            if len(recent) > 1:
                health['unstable'] = np.std(recent) / (np.mean(recent) + 1e-8) > 10.0
            else:
                health['unstable'] = False
        else:
            health = {'vanishing': False, 'exploding': False, 'unstable': False}
        
        return health


class EarlyStopping:
    """
    Early stopping based on loss plateau detection.
    """
    
    def __init__(self, patience: int = 5000, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
        
    def check(self, loss: float, state_dict: Optional[Dict] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            loss: Current loss value
            state_dict: Optional model state to save
            
        Returns:
            should_stop: Whether to stop training
        """
        if loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = loss
            self.counter = 0
            if state_dict is not None:
                self.best_state = {k: v.clone() for k, v in state_dict.items()}
            return False
        else:
            # No improvement
            self.counter += 1
            return self.counter >= self.patience
    
    def get_best_state(self) -> Optional[Dict]:
        """Get the best model state."""
        return self.best_state

"""
Stall detection and dynamic lambda adjustment for improved convergence.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class StallDetector:
    """
    Detects optimization stalls and suggests parameter adjustments.
    """
    
    def __init__(self, 
                 window_size: int = 500,  # Reduced from 1000
                 patience: int = 1500,    # Reduced from 3000
                 improvement_threshold: float = 0.005,  # Reduced from 0.01
                 contour_loss_threshold: float = 0.002):
        """
        Args:
            window_size: Size of moving average window
            patience: Steps to wait before declaring stall
            improvement_threshold: Minimum relative improvement to not be stalled
            contour_loss_threshold: Minimum contour loss improvement
        """
        self.window_size = window_size
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.contour_loss_threshold = contour_loss_threshold
        
        # Loss history
        self.total_loss_history = deque(maxlen=window_size * 2)
        self.contour_loss_history = deque(maxlen=window_size * 2)
        self.area_dev_history = deque(maxlen=window_size * 2)
        
        # Stall tracking
        self.steps_since_improvement = 0
        self.best_loss = float('inf')
        self.best_contour_loss = float('inf')
        self.last_adjustment_step = -patience
        
        # Lambda tracking
        self.lambda_contour_adjustments = []
        
    def update(self, 
               total_loss: float, 
               contour_loss: float,
               area_deviation: float,
               step: int) -> Tuple[bool, Optional[float]]:
        """
        Update stall detector with current metrics.
        
        Returns:
            (is_stalled, suggested_lambda_contour_multiplier)
        """
        self.total_loss_history.append(total_loss)
        self.contour_loss_history.append(contour_loss)
        self.area_dev_history.append(area_deviation)
        
        # Not enough history yet
        if len(self.total_loss_history) < self.window_size:
            return False, None
        
        # Compute moving averages
        recent_start = len(self.total_loss_history) - self.window_size
        old_start = max(0, len(self.total_loss_history) - 2 * self.window_size)
        
        # Ensure we have data for old window
        if old_start >= recent_start:
            return False, None
            
        recent_total = np.mean(list(self.total_loss_history)[recent_start:])
        old_total = np.mean(list(self.total_loss_history)[old_start:recent_start])
        
        recent_contour = np.mean(list(self.contour_loss_history)[recent_start:])
        old_contour = np.mean(list(self.contour_loss_history)[old_start:recent_start])
        
        recent_area_dev = np.mean(list(self.area_dev_history)[recent_start:])
        
        # Check for improvement - focus on CONTOUR loss only
        contour_improvement = (old_contour - recent_contour) / max(old_contour, 1e-9)
        
        # Only consider contour loss for stall detection
        has_improved = contour_improvement > self.contour_loss_threshold
        
        if has_improved:
            self.steps_since_improvement = 0
            self.best_loss = min(self.best_loss, recent_total)
            self.best_contour_loss = min(self.best_contour_loss, recent_contour)
        else:
            self.steps_since_improvement += 1
        
        # Check if we're stalled
        is_stalled = (
            self.steps_since_improvement > self.patience and
            step - self.last_adjustment_step > self.patience // 2  # Can adjust more frequently
        )
        
        suggested_multiplier = None
        
        # Only ramp up after patches have formed (low area deviation) and enough steps
        patches_formed = recent_area_dev < 0.05  # Stricter threshold
        min_steps_before_ramp = 20000  # Don't ramp in first 20k steps
        
        if is_stalled and patches_formed and step > min_steps_before_ramp:
            # Only increase if contour loss is not already very low
            if recent_contour > 0.01:  # Only if there's room for improvement
                # Very small increase
                suggested_multiplier = 1.02  # 2% increase only
            else:
                # Contour loss already low, no need to increase
                suggested_multiplier = 1.0
        else:
            # No increase if patches not formed or too early
            suggested_multiplier = None
        
        # Only record adjustment if we're suggesting an increase
        if suggested_multiplier is not None and suggested_multiplier > 1.0:
            self.last_adjustment_step = step
            self.lambda_contour_adjustments.append({
                'step': step,
                'multiplier': suggested_multiplier,
                'contour_loss': recent_contour,
                'area_dev': recent_area_dev
            })
        
        return is_stalled, suggested_multiplier
    
    def get_adaptive_lambda_contour(self, 
                                   base_lambda: float,
                                   current_step: int,
                                   max_lambda: float = 5.0,
                                   smooth_transitions: bool = True,
                                   total_steps: int = 300000) -> float:
        """
        Get adaptively adjusted lambda_contour based on stall history.
        
        Args:
            base_lambda: Base lambda from schedule
            current_step: Current optimization step
            max_lambda: Maximum allowed lambda_contour
            smooth_transitions: Whether to smooth stage transitions
            
        Returns:
            Adjusted lambda_contour
        """
        # Apply all historical adjustments
        adjusted_lambda = base_lambda
        
        for adjustment in self.lambda_contour_adjustments:
            if adjustment['step'] <= current_step:
                adjusted_lambda *= adjustment['multiplier']
        
        # NO automatic growth - only increase when stalled
        # This makes the optimization more conservative
        
        # Cap at maximum
        adjusted_lambda = min(adjusted_lambda, max_lambda)
        
        # Smooth transitions: limit how much lambda can grow relative to base
        if smooth_transitions:
            # Maximum growth factor relative to base lambda
            max_growth = 25.0  # λ can be at most 25x the base value (increased)
            adjusted_lambda = min(adjusted_lambda, base_lambda * max_growth)
        
        return adjusted_lambda
    
    def get_stats(self) -> Dict:
        """Get current stall detector statistics."""
        if len(self.total_loss_history) < self.window_size:
            return {
                'steps_since_improvement': self.steps_since_improvement,
                'num_adjustments': len(self.lambda_contour_adjustments),
                'is_warming_up': True
            }
        
        recent_start = len(self.total_loss_history) - self.window_size
        recent_total = np.mean(list(self.total_loss_history)[recent_start:])
        recent_contour = np.mean(list(self.contour_loss_history)[recent_start:])
        recent_area_dev = np.mean(list(self.area_dev_history)[recent_start:])
        
        return {
            'steps_since_improvement': self.steps_since_improvement,
            'num_adjustments': len(self.lambda_contour_adjustments),
            'recent_total_loss': recent_total,
            'recent_contour_loss': recent_contour,
            'recent_area_dev': recent_area_dev,
            'is_warming_up': False
        }
"""
Smooth lambda scheduling with gradual transitions.
"""
import torch
import numpy as np
from typing import Dict, Optional


class SmoothLambdaScheduler:
    """
    Provides smooth transitions for lambda values between stages.
    """
    
    def __init__(self, transition_steps: int = 2000):
        """
        Args:
            transition_steps: Number of steps to smoothly transition between stages
        """
        self.transition_steps = transition_steps
        self.last_stage_name = None
        self.transition_start_step = None
        self.old_lambdas = {}
        self.new_lambdas = {}
        
    def get_smooth_lambda(self,
                         lambda_name: str,
                         base_value: float,
                         current_step: int,
                         stage_name: str) -> float:
        """
        Get smoothly transitioned lambda value.
        
        Args:
            lambda_name: Name of lambda (e.g., 'contour', 'smooth', 'area')
            base_value: Base value from current stage (or adaptive value)
            current_step: Current optimization step
            stage_name: Current stage name
            
        Returns:
            Smoothly transitioned lambda value
        """
        # For lambda_contour, the base_value is already adaptive - just return it
        # The smooth scheduler was incorrectly caching and overriding adaptive values
        if lambda_name == 'contour':
            return base_value
            
        # For other lambdas, do smooth transitions between stages
        # Check if we're in a new stage
        if stage_name != self.last_stage_name:
            if self.last_stage_name is not None:
                # Starting a transition
                self.transition_start_step = current_step
                self.old_lambdas = self.new_lambdas.copy()
                self.new_lambdas[lambda_name] = base_value
            else:
                # First stage - no transition
                self.new_lambdas[lambda_name] = base_value
                self.old_lambdas[lambda_name] = base_value
            
            self.last_stage_name = stage_name
        
        # Update target value
        if lambda_name not in self.new_lambdas:
            self.new_lambdas[lambda_name] = base_value
            self.old_lambdas[lambda_name] = base_value
        else:
            # Update new_lambdas to track the current base value
            self.new_lambdas[lambda_name] = base_value
        
        # Check if we're in a transition
        if (self.transition_start_step is not None and 
            current_step < self.transition_start_step + self.transition_steps and
            lambda_name in self.old_lambdas):
            
            # Compute transition progress
            progress = (current_step - self.transition_start_step) / self.transition_steps
            progress = np.clip(progress, 0.0, 1.0)
            
            # Smooth transition using cosine interpolation
            smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))
            
            # Interpolate between old and new values
            old_val = self.old_lambdas.get(lambda_name, base_value)
            new_val = self.new_lambdas.get(lambda_name, base_value)
            
            return old_val + (new_val - old_val) * smooth_progress
        
        # No transition - return current value
        return self.new_lambdas.get(lambda_name, base_value)
    
    def reset(self):
        """Reset the scheduler state."""
        self.last_stage_name = None
        self.transition_start_step = None
        self.old_lambdas = {}
        self.new_lambdas = {}

"""
Improved training pipeline for ReLU mesh segmentation.
Addresses all convergence issues identified in the report.
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Import our modules
from utils.mesh_preprocessing import (
    load_volume_tet_mesh_and_extract_surface,
    preprocess_mesh,
    pick_axis_aligned_anchors
)
from losses.improved_losses import (
    contour_alignment_intrinsic,
    contour_alignment_intrinsic_v2,
    smoothness_cotan,
    area_fractions_and_kl,
    area_balance_loss,
    compute_hard_area_fractions,
    pin_loss,
    compute_boundary_stats,
    compute_boundary_stats_v2,
    compute_boundary_length_estimate,
    total_variation_loss,
    non_boundary_margin_loss,
    potts_smoothness_loss,
    boundary_length_regularizer,
    normal_axis_losses,
    area_kl_to_prior,
    triple_point_barrier,
    margin_separation_loss
)
from losses.soft_pairs_contour import contour_alignment_soft_pairs
from optimization.temperature_control import (
    TempController,
    TwoStageScheduler,
    GradientMonitor
)
from optimization.stall_detector import StallDetector
from optimization.smooth_scheduler import SmoothLambdaScheduler


def initialize_field(n_vertices: int, n_channels: int, 
                    pinned_indices: torch.Tensor,
                    device: torch.device,
                    verts_xyz: torch.Tensor,
                    channel_labels: Optional[Dict[int, str]] = None) -> torch.Tensor:
    """
    Initialize multi-channel field with coordinate bias to reduce early ties.
    
    Args:
        n_vertices: Number of vertices
        n_channels: Number of channels (6 for axis-aligned segmentation)
        pinned_indices: Indices of pinned vertices
        device: PyTorch device
        verts_xyz: (N, 3) vertex coordinates
        
    Returns:
        F: (N, C) initialized field values
    """
    F = torch.zeros(n_vertices, n_channels, device=device)
    
    # Normalize coordinates
    xyz = (verts_xyz - verts_xyz.mean(0, keepdim=True)) / (verts_xyz.std(0, keepdim=True) + 1e-6)
    
    # Channels: [+X, -X, +Y, -Y, +Z, -Z] - aligned with channel_labels
    if n_channels >= 6:
        # Scale down initial values even more to prevent early instability
        scale = 0.05  # reduced from 0.1
        
        # Initialize based on channel labels if provided
        if channel_labels:
            for c, label in channel_labels.items():
                if c < n_channels:
                    if label == '+X':
                        F[:, c] = xyz[:, 0] * scale
                    elif label == '-X':
                        F[:, c] = -xyz[:, 0] * scale
                    elif label == '+Y':
                        F[:, c] = xyz[:, 1] * scale
                    elif label == '-Y':
                        F[:, c] = -xyz[:, 1] * scale
                    elif label == '+Z':
                        F[:, c] = xyz[:, 2] * scale
                    elif label == '-Z':
                        F[:, c] = -xyz[:, 2] * scale
        else:
            # Default initialization if no labels
            F[:, 0] =  xyz[:, 0] * scale  # +X
            F[:, 1] = -xyz[:, 0] * scale  # -X
            F[:, 2] =  xyz[:, 1] * scale  # +Y
            F[:, 3] = -xyz[:, 1] * scale  # -Y
            F[:, 4] =  xyz[:, 2] * scale  # +Z
            F[:, 5] = -xyz[:, 2] * scale  # -Z
    
    # Add very small noise
    F += 0.0005 * torch.randn_like(F)  # reduced from 0.001
    
    # Keep anchor contrast - now properly mapped to channels
    # Each pin gets +1 on its channel, -1 on all others
    for c in range(min(n_channels, len(pinned_indices))):
        F[pinned_indices[c], :] = -1.0
        F[pinned_indices[c], c] = 1.0
    
    return F


def compute_pin_targets(pinned_indices: torch.Tensor, 
                       n_channels: int,
                       device: torch.device,
                       channel_labels: Optional[Dict[int, str]] = None) -> torch.Tensor:
    """
    Compute target values for pinned vertices.
    
    Args:
        pinned_indices: (P,) indices of pinned vertices
        n_channels: Number of channels
        device: PyTorch device
        
    Returns:
        targets: (P, C) target values
    """
    P = len(pinned_indices)
    targets = torch.full((P, n_channels), -1.0, device=device)
    
    # Each pin index i corresponds to channel i (order preserved)
    for i in range(min(P, n_channels)):
        targets[i, i] = 1.0
    
    return targets


def train_mesh_segmentation(
    mesh_data: Dict[str, torch.Tensor],
    n_channels: int = 6,
    n_steps: int = 100000,
    device: str = 'cuda',
    output_dir: Optional[Path] = None,
    checkpoint_freq: int = 5000,
    log_freq: int = 500,
    verbose: bool = True,
    resume_from: Optional[str] = None,
    use_soft_pairs: bool = False,
    use_5_patch_prior: bool = False,
    use_v2_contour: bool = True,
    use_improved_area: bool = True
) -> Tuple[torch.Tensor, Dict]:
    """
    Main training function with improved optimization.
    
    Args:
        mesh_data: Preprocessed mesh data from preprocess_mesh()
        n_channels: Number of segmentation channels
        n_steps: Number of optimization steps
        device: PyTorch device
        output_dir: Directory for saving checkpoints
        checkpoint_freq: Frequency of checkpointing
        log_freq: Frequency of logging
        verbose: Whether to print progress
        
    Returns:
        F: Optimized field values
        history: Training history
    """
    # Extract mesh data
    verts = mesh_data['vertices']
    faces = mesh_data['faces']
    tri_area = mesh_data['tri_area']
    tri_xy = mesh_data['tri_xy']
    edge_idx = mesh_data['edge_idx']
    edge_tris = mesh_data['edge_tris']
    I = mesh_data['cotan_I']
    J = mesh_data['cotan_J']
    W = mesh_data['cotan_W']
    pinned_indices = mesh_data['pinned_indices']
    channel_labels = mesh_data.get('channel_labels', {i: f'Ch{i}' for i in range(6)})
    stats = mesh_data['stats']
    
    # Initialize or load from checkpoint
    start_step = 0
    opt_path = None
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        ckpt = np.load(resume_from)
        F = torch.tensor(ckpt['field_values'], device=device, dtype=torch.float32)
        F = nn.Parameter(F)
        start_step = int(ckpt['step'])
        
        # Initialize controllers with saved state
        temp_ctrl = TempController()
        temp_ctrl.beta_contour = float(ckpt.get('beta_contour', 4.0))
        temp_ctrl.beta_area = float(ckpt.get('beta_area', 2.5))
        temp_ctrl.last_beta_update_step = start_step - 1000  # Allow immediate updates
        stall_detector = StallDetector()
        smooth_scheduler = SmoothLambdaScheduler(transition_steps=2000)
        
        print(f"  Resuming from step {start_step}")
        print(f"  Current β: contour={temp_ctrl.beta_contour:.2f}, area={temp_ctrl.beta_area:.2f}")
        
        # Check if there's an optimizer state
        opt_path = Path(resume_from).parent / f"{Path(resume_from).stem}_optimizer.pt"
        if opt_path.exists():
            opt_state = torch.load(opt_path, map_location=device)
            if 'temp_ctrl_state' in opt_state:
                # Update temp controller history
                for k, v in opt_state['temp_ctrl_state'].items():
                    if hasattr(temp_ctrl, k):
                        setattr(temp_ctrl, k, v)
    else:
        # Initialize field with coordinate bias
        F = initialize_field(verts.shape[0], n_channels, pinned_indices, device, verts, channel_labels)
        F = nn.Parameter(F)
        temp_ctrl = TempController()
        stall_detector = StallDetector()
        smooth_scheduler = SmoothLambdaScheduler(transition_steps=2000)
    
    # Pin targets
    pin_targets = compute_pin_targets(pinned_indices, n_channels, device, channel_labels)
    
    # Optimizer - reduced learning rate for better stability
    optimizer = torch.optim.AdamW([F], lr=5e-5, weight_decay=0.0)  # reduced from 1e-4
    
    # Load optimizer state if resuming
    if resume_from and opt_path and opt_path.exists():
        opt_state = torch.load(opt_path, map_location=device)
        if 'optimizer_state' in opt_state:
            optimizer.load_state_dict(opt_state['optimizer_state'])
            print("  Loaded optimizer state")
    
    # Controllers
    scheduler = TwoStageScheduler(n_steps)
    grad_monitor = GradientMonitor()
    
    # History tracking
    history = {
        'loss': [],
        'loss_smooth': [],
        'loss_contour': [],
        'loss_area': [],
        'loss_pin': [],
        'loss_potts': [],
        'loss_boundary_length': [],
        'loss_normal_align': [],
        'loss_normal_disp': [],
        'loss_triple': [],
        'area_fractions': [],
        'beta_contour': [],
        'beta_area': [],
        'boundary_length': [],
        'active_edge_fraction': [],
        'lr': []
    }
    
    # Training loop
    start_time = time.time()
    
    for step in range(start_step, n_steps):
        # Get current stage
        prev_stage_name = scheduler.stages[scheduler.current_stage_idx].name if step > 0 else None
        stage = scheduler.get_stage(step)
        if prev_stage_name and stage.name != prev_stage_name:
            print(f"\n{'='*80}")
            print(f"Stage transition: {prev_stage_name} -> {stage.name}")
            print(f"New weights: λ_smooth={stage.lambda_smooth}, λ_contour={stage.lambda_contour}, λ_area={stage.lambda_area}")
            print(f"{'='*80}\n")
        
        # Update learning rate
        current_lr = scheduler.get_lr(step, stage.lr)
        
        # Temperature-aware LR scaling to prevent instability at high β
        temp_scale = min(1.0, 2.0 / max(1.0, float(temp_ctrl.beta_contour)))
        area_scale = min(1.0, 1.5 / max(1.0, float(temp_ctrl.beta_area)))
        scaled_lr = max(1e-5, current_lr * temp_scale * area_scale)
        
        for g in optimizer.param_groups:
            g['lr'] = scaled_lr
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Compute losses
        losses = {}
        
        # Smoothness loss (cotangent Laplacian)
        losses['smooth'] = smoothness_cotan(F, I, J, W)
        
        # Contour alignment (3D) - get edge weights too for margin loss
        # Add debug logging every 1000 steps
        debug_contour = (step % 1000 == 0) and verbose
        
        if use_v2_contour:
            # Use improved v2 contour with all-pairs mixing and soft-OR coverage
            contour_result = contour_alignment_intrinsic_v2(
                F, faces, edge_idx, edge_tris, verts,
                beta_contour=temp_ctrl.beta_contour,
                return_weights=True,
                debug_log=debug_contour
            )
            losses['contour'], edge_weights = contour_result
        elif use_soft_pairs:
            # Use soft all-pairs mixing for stable triple points
            contour_result = contour_alignment_soft_pairs(
                F, faces, edge_idx, edge_tris,
                beta_contour=temp_ctrl.beta_contour,
                return_weights=True,
                verts=verts
            )
            losses['contour'], edge_weights = contour_result
        else:
            # Use standard top-2 channel selection
            contour_result = contour_alignment_intrinsic(
                F, faces, edge_idx, edge_tris,
                beta_contour=temp_ctrl.beta_contour,
                return_weights=True,
                verts=verts,
                debug_log=debug_contour
            )
            losses['contour'], edge_weights = contour_result
        
        # Area balance
        if use_improved_area:
            # Use improved area loss with reverse KL and straight-through
            # Enable ST earlier to align with hard assignments
            use_straight_through = temp_ctrl.beta_area > 2.0 and step > 2000  # Enable ST much earlier
            barrier_weight = 0.1 if step < 10000 else 0.05 if step < 20000 else 0.02  # Stronger barrier early
            
            # For 5-patch prior, adjust the min_frac to allow one channel to be small
            if use_5_patch_prior:
                min_frac_val = 0.01 / n_channels  # Allow one channel to be very small
                max_frac_val = 2.5 / n_channels   # Allow larger channels
            else:
                min_frac_val = None  # Use defaults
                max_frac_val = None
            
            # Force straight-through from the beginning for better gradients
            use_straight_through = step > 100  # Enable ST almost immediately
            
            # Use L2 method for more direct gradient signal
            # Add entropy regularization to encourage decisive assignments
            entropy_w = 0.5 if step < 5000 else 0.2 if step < 10000 else 0.1
            losses['area'], area_frac = area_balance_loss(
                F, faces, tri_area, beta_area=temp_ctrl.beta_area,
                use_straight_through=use_straight_through,
                method="l2",  # L2 gives clearer gradients than reverse KL
                barrier_w=barrier_weight,
                min_frac=min_frac_val,
                max_frac=max_frac_val,
                entropy_weight=entropy_w
            )
            
            # Also compute hard fractions for monitoring
            area_frac_hard = compute_hard_area_fractions(F, faces, tri_area)
        else:
            # Original area loss
            use_entropy = stage.name == "Stage 0: Smoothness + Area + Normal Warmup"
            losses['area'], area_frac = area_fractions_and_kl(
                F, faces, tri_area, beta_area=temp_ctrl.beta_area,
                use_entropy_regularization=use_entropy
            )
            
            # Optionally use 5-patch prior instead of uniform
            if use_5_patch_prior and not use_entropy:
                # Build a 5+1 prior (last channel optional)
                prior = torch.full((n_channels,), 1.0/5.0, device=device, dtype=F.dtype)
                prior[-1] = 1e-3  # Let the last channel be nearly empty
                prior = prior / prior.sum()  # Renormalize
                
                # Replace area loss with KL to non-uniform prior
                losses['area'] = area_kl_to_prior(area_frac, prior)
            
            area_frac_hard = area_frac  # For compatibility
        
        # Pin constraint (soft or hard)
        if stage.use_hard_pins:
            # Hard projection - enforce exact values
            with torch.no_grad():
                F.data[pinned_indices] = pin_targets
            losses['pin'] = torch.tensor(0.0, device=device)
        else:
            # Soft penalty with stronger weight during early stages
            losses['pin'] = pin_loss(F, pinned_indices, pin_targets)
        
        # Optional total variation
        if stage.lambda_tv is not None:
            losses['tv'] = total_variation_loss(F, edge_idx)
        
        # Margin sharpening loss for Stage A
        if stage.name.startswith("Stage A"):
            losses['sharp'] = non_boundary_margin_loss(F, edge_idx, edge_weights, tau=0.2)
        else:
            losses['sharp'] = torch.tensor(0.0, device=device)
        
        # Potts smoothness on probabilities (to reduce speckles)
        losses['potts'] = potts_smoothness_loss(F, edge_idx, edge_weights, 
                                                temp_ctrl.beta_area, gamma=2.0)
        
        # Boundary length regularizer (to reduce ragged seams)
        losses['boundary_length'] = boundary_length_regularizer(edge_idx, edge_weights, verts)
        
        # Normal axis alignment losses (for axis-oriented patches)
        # Define axis vectors for each channel: [+X, -X, +Y, -Y, +Z, -Z]
        axis_vectors = torch.tensor([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], device=device, dtype=F.dtype)[:n_channels]
        
        losses['normal_align'], losses['normal_disp'] = normal_axis_losses(
            verts, faces, tri_area, F, temp_ctrl.beta_area, axis_vectors
        )
        
        # Triple point barrier to reduce Y-junctions and speckles
        losses['triple'] = triple_point_barrier(F, faces, tri_area, beta_area=temp_ctrl.beta_area, margin=0.10)
        
        # Margin separation loss to break symmetry (only in early stages)
        if step < 20000:
            losses['margin_sep'] = margin_separation_loss(F, tau=0.3)
        else:
            losses['margin_sep'] = torch.tensor(0.0, device=device)
        
        # Debug: print area fractions and normal losses
        if step % 1000 == 0 and verbose:
            if use_improved_area:
                st_status = "ON" if use_straight_through else "OFF"
                print(f"[DEBUG] Area fractions (soft): {area_frac.detach().cpu().numpy()}")
                print(f"[DEBUG] Area fractions (hard): {area_frac_hard.detach().cpu().numpy()}")
                print(f"[DEBUG] β_area={temp_ctrl.beta_area}, area_loss={losses['area'].item():.6f}, "
                      f"ST={st_status}, barrier_w={barrier_weight}, "
                      f"normal_align={losses['normal_align'].item():.4f}, normal_disp={losses['normal_disp'].item():.4f}")
            else:
                print(f"[DEBUG] Area fractions: {area_frac.detach().cpu().numpy()}, β_area={temp_ctrl.beta_area}, area_loss={losses['area'].item():.6f}, "
                      f"normal_align={losses['normal_align'].item():.4f}, normal_disp={losses['normal_disp'].item():.4f}")
            
            # Also print pin values to verify channel mapping
            if step == 0:
                print(f"\n[DEBUG] Pin mapping:")
                for i, idx in enumerate(pinned_indices[:6]):
                    label = channel_labels.get(i, f'Ch{i}')
                    values = F.data[idx].detach().cpu().numpy()
                    print(f"  Pin {i} ({label}) at vertex {idx}: {values}")
        
        # Check for stalls and get adaptive lambda_contour
        area_dev_soft = (area_frac - 1.0/n_channels).abs().max().item()
        area_dev_hard = (area_frac_hard - 1.0/n_channels).abs().max().item() if use_improved_area else area_dev_soft
        # Use hard deviation for stall detection when using improved area loss
        area_dev = area_dev_hard if use_improved_area else area_dev_soft
        is_stalled, suggested_multiplier = stall_detector.update(
            total_loss=losses['smooth'].item() + losses['contour'].item() + losses['area'].item(),
            contour_loss=losses['contour'].item(),
            area_deviation=area_dev,
            step=step
        )
        
        # Get adaptive lambda_contour (can grow up to 5.0)
        adaptive_lambda_contour = stall_detector.get_adaptive_lambda_contour(
            base_lambda=stage.lambda_contour,
            current_step=step,
            max_lambda=5.0,
            total_steps=n_steps  # Pass total steps for proper scaling
        )
        
        # Debug: log stage info every 5000 steps
        if step % 5000 == 0 and verbose:
            fraction = step / n_steps
            print(f"[DEBUG] Step {step}, fraction={fraction:.3f}, stage={stage.name}, "
                  f"base_lambda={stage.lambda_contour}, adaptive_lambda={adaptive_lambda_contour:.3f}")
        
        # Apply smooth transitions for all lambdas
        smooth_lambda_smooth = smooth_scheduler.get_smooth_lambda(
            'smooth', stage.lambda_smooth, step, stage.name
        )
        smooth_lambda_contour = smooth_scheduler.get_smooth_lambda(
            'contour', adaptive_lambda_contour, step, stage.name
        )
        smooth_lambda_area = smooth_scheduler.get_smooth_lambda(
            'area', stage.lambda_area, step, stage.name
        )
        smooth_lambda_pin = smooth_scheduler.get_smooth_lambda(
            'pin', stage.lambda_pin, step, stage.name
        )
        
        # Check for non-finite losses before combining
        for k, v in list(losses.items()):
            if not torch.isfinite(v):
                print(f"[DEBUG] Non-finite in {k}: {v.item()}")
                # Optional: dump some quick stats from contour module
                raise RuntimeError(f"Non-finite loss detected in {k}")
        
        # Total loss with smooth lambdas
        total_loss = (
            smooth_lambda_smooth * losses['smooth'] +
            smooth_lambda_contour * losses['contour'] +
            smooth_lambda_area * losses['area'] +
            smooth_lambda_pin * losses['pin']
        )
        
        if 'tv' in losses and stage.lambda_tv is not None:
            total_loss = total_loss + stage.lambda_tv * losses['tv']
        
        # Add margin loss with appropriate weight
        if stage.name.startswith("Stage A"):
            lambda_sharp = 0.08 if "A1" in stage.name or "A2" in stage.name else 0.04
            total_loss = total_loss + lambda_sharp * losses['sharp']
        
        # Add Potts smoothness (stronger early, weaker late)
        lambda_potts = 0.15 if "Stage A" in stage.name else 0.07
        total_loss = total_loss + lambda_potts * losses['potts']
        
        # Add boundary length regularizer (increased weight)
        lambda_boundary = 1e-3  # Stronger to remove zigzags
        total_loss = total_loss + lambda_boundary * losses['boundary_length']
        
        # Add normal losses
        if stage.name == "Stage 0: Smoothness + Area + Normal Warmup":
            # High normal losses during warmup to establish axis alignment early
            lambda_normal_align = 0.5
            lambda_normal_disp = 0.2
            total_loss = total_loss + lambda_normal_align * losses['normal_align']
            total_loss = total_loss + lambda_normal_disp * losses['normal_disp']
        elif step > 30000:  # Continue after initial stages
            lambda_normal_align = 0.2 if step > 60000 else 0.05
            lambda_normal_disp = 0.1 if step > 60000 else 0.02
            total_loss = total_loss + lambda_normal_align * losses['normal_align']
            total_loss = total_loss + lambda_normal_disp * losses['normal_disp']
        
        # Add triple point barrier with schedule
        if step < 20000:
            lambda_triple = 0.05  # Light during early stages
        elif step < 60000:
            lambda_triple = 0.05  # Maintain during middle
        else:
            lambda_triple = 0.03  # Reduce late
        total_loss = total_loss + lambda_triple * losses['triple']
        
        # Add margin separation loss (strong early, then decay)
        if step < 20000:
            lambda_margin = 0.5 if step < 5000 else 0.2
            total_loss = total_loss + lambda_margin * losses['margin_sep']
        
        # Check for NaN before backward
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\nWARNING: NaN/Inf detected at step {step}")
            print(f"  Total loss: {total_loss.item()}")
            print(f"  Individual losses: {[(k, v.item()) for k, v in losses.items()]}")
            print(f"  Field stats: min={F.min().item():.6f}, max={F.max().item():.6f}")
            
            # Skip this step
            optimizer.zero_grad()
            
            # Try to recover by reducing learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"  Reduced learning rate to {optimizer.param_groups[0]['lr']:.2e}")
            
            # Reset field to last checkpoint if too many NaN
            nan_count = getattr(train_mesh_segmentation, 'nan_count', 0) + 1
            train_mesh_segmentation.nan_count = nan_count
            
            if nan_count > 10:
                print("  Too many NaN occurrences, stopping training")
                break
                
            continue
        
        # Backward pass
        total_loss.backward()
        
        # Defensive gradient handling to keep training moving
        if (F.grad is None) or torch.isnan(F.grad).any() or torch.isinf(F.grad).any():
            with torch.no_grad():
                if F.grad is None:
                    F.grad = torch.zeros_like(F)
                    print(f"\nWARNING: No gradient computed at step {step}")
                else:
                    nan_count = torch.isnan(F.grad).sum().item()
                    inf_count = torch.isinf(F.grad).sum().item()
                    if nan_count > 0 or inf_count > 0:
                        print(f"\nWARNING: NaN/Inf in gradients at step {step} (NaN: {nan_count}, Inf: {inf_count})")
                        print(f"  Grad norm before fix: {F.grad.norm().item()}")
                        
                torch.nan_to_num_(F.grad, nan=0.0, posinf=0.0, neginf=0.0)
                F.grad.clamp_(-1.0, 1.0)  # Clip-by-value as safety
                
                # Halve LR for a few steps after a NaN event
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"  Reduced LR to {optimizer.param_groups[0]['lr']:.2e} after gradient issue")
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_([F], max_norm=5.0)
        
        # Additional gradient check
        if grad_norm > 100:
            print(f"\nWARNING: Large gradient norm {grad_norm:.2f} at step {step}")
        
        # Optimizer step
        optimizer.step()
        
        # Post-step NaN check with recovery attempt
        if not torch.isfinite(F).all():
            print(f"[DEBUG] F blew up at step {step}. Clamping and continuing once...")
            with torch.no_grad():
                torch.nan_to_num_(F, nan=0.0, posinf=10.0, neginf=-10.0)
                F.clamp_(-20.0, 20.0)  # keeps softmax & sigmoids sane
            # Only break if it happens repeatedly
            nan_count = getattr(train_mesh_segmentation, 'post_step_nan_count', 0) + 1
            train_mesh_segmentation.post_step_nan_count = nan_count
            if nan_count > 3:
                print(f"\nERROR: Repeated NaN in field after optimizer step")
                break
        
        # Update history
        history['loss'].append(total_loss.item())
        history['loss_smooth'].append(losses['smooth'].item())
        history['loss_contour'].append(losses['contour'].item()) 
        history['loss_area'].append(losses['area'].item())
        history['loss_pin'].append(losses['pin'].item())
        history['loss_potts'].append(losses['potts'].item())
        history['loss_boundary_length'].append(losses['boundary_length'].item())
        history['loss_normal_align'].append(losses['normal_align'].item())
        history['loss_normal_disp'].append(losses['normal_disp'].item())
        history['loss_triple'].append(losses['triple'].item())
        history['area_fractions'].append(area_frac.detach().cpu().numpy())
        history['beta_contour'].append(temp_ctrl.beta_contour)
        history['beta_area'].append(temp_ctrl.beta_area)
        history['lr'].append(current_lr)
        
        # Progress-based temperature update (every 400 steps)
        if step % 400 == 0 and stage.allow_temp_increase:
            # Store old best before update
            prev_best = temp_ctrl.best_contour_loss_since_update
            
            if use_v2_contour:
                boundary_length, active_frac, median_phi = compute_boundary_stats_v2(
                    F, edge_idx, verts, temp_ctrl.beta_contour
                )
            else:
                boundary_length, active_frac = compute_boundary_stats(
                    F, edge_idx, verts, temp_ctrl.beta_contour
                )
            history['boundary_length'].append(boundary_length)
            history['active_edge_fraction'].append(active_frac)
            
            updated = temp_ctrl.check_and_update(
                area_frac, boundary_length, stats['bbox_diagonal'], step,
                contour_loss=losses['contour'].item()
            )
            
            if updated and verbose:
                print(f"  -> Temperature increased: βc={temp_ctrl.beta_contour:.1f}, βa={temp_ctrl.beta_area:.1f}")
                if torch.isfinite(torch.tensor(prev_best)):
                    imp = (prev_best - losses['contour'].item()) / max(prev_best, 1e-9)
                    print(f"     (improvement: {imp:.1%})")
        
        # Logging
        if step % log_freq == 0 and verbose:
            # Get current boundary stats for logging
            if step > 0:
                if use_v2_contour:
                    _, active_frac, _ = compute_boundary_stats_v2(
                        F, edge_idx, verts, temp_ctrl.beta_contour
                    )
                else:
                    _, active_frac = compute_boundary_stats(
                        F, edge_idx, verts, temp_ctrl.beta_contour
                    )
            else:
                active_frac = 0.0
                
            # Calculate weighted losses to show actual contribution
            weighted_smooth = smooth_lambda_smooth * losses['smooth'].item()
            weighted_contour = smooth_lambda_contour * losses['contour'].item()
            weighted_area = smooth_lambda_area * losses['area'].item()
            
            # Add stall detection info
            if is_stalled and suggested_multiplier and suggested_multiplier > 1.0:
                print(f"  -> CONTOUR STALL DETECTED! λ_contour: {stage.lambda_contour:.3f} → {adaptive_lambda_contour:.3f} (×{suggested_multiplier:.2f})")
                
            # Calculate weighted normal losses
            weighted_normal_align = 0.0
            weighted_normal_disp = 0.0
            if stage.name == "Stage 0: Smoothness + Area + Normal Warmup":
                weighted_normal_align = 0.5 * losses['normal_align'].item()
                weighted_normal_disp = 0.2 * losses['normal_disp'].item()
            elif step > 30000:
                lambda_normal_align = 0.2 if step > 60000 else 0.05
                lambda_normal_disp = 0.1 if step > 60000 else 0.02
                weighted_normal_align = lambda_normal_align * losses['normal_align'].item()
                weighted_normal_disp = lambda_normal_disp * losses['normal_disp'].item()
            
            # Prepare area dev string
            if use_improved_area:
                area_dev_str = f"AreaDev: soft={area_dev_soft:.3f}, hard={area_dev_hard:.3f}"
            else:
                area_dev_str = f"AreaDev: {area_dev:.3f}"
                
            print(f"Step {step:6d}/{n_steps} | Loss: {total_loss.item():.6f} | "
                  f"Smooth: {losses['smooth'].item():.4f} ({weighted_smooth:.4f}) | "
                  f"Contour: {losses['contour'].item():.4f} ({weighted_contour:.4f}) [λ={smooth_lambda_contour:.2f}] | "
                  f"Area: {losses['area'].item():.4f} ({weighted_area:.4f}) | "
                  f"NormAlign: {losses['normal_align'].item():.4f} ({weighted_normal_align:.4f}) | "
                  f"NormDisp: {losses['normal_disp'].item():.4f} ({weighted_normal_disp:.4f}) | "
                  f"{area_dev_str} | "
                  f"ActiveEdge: {active_frac:.1%} | "
                  f"βc: {temp_ctrl.beta_contour:.1f} | "
                  f"LR: {scaled_lr:.2e}")
            
            # Check gradient health
            grad_monitor.log_gradients(F, {})
            health = grad_monitor.check_health()
            if any(health.values()):
                print(f"  -> Gradient issues: {health}")
        
        # Checkpointing
        if output_dir and step % checkpoint_freq == 0 and step > 0:
            # Save comprehensive checkpoint as npz
            checkpoint_data = {
                # Core data
                'step': step,
                'field_values': F.data.cpu().numpy(),
                'vertices': verts.cpu().numpy(),
                'faces': faces.cpu().numpy(),
                'pinned_indices': pinned_indices.cpu().numpy(),
                'channel_labels': channel_labels,
                
                # Current state
                'beta_contour': temp_ctrl.beta_contour,
                'beta_area': temp_ctrl.beta_area,
                'stage_name': stage.name,
                'learning_rate': current_lr,
                
                # Loss values
                'total_loss': total_loss.item(),
                'loss_smooth': losses['smooth'].item(),
                'loss_contour': losses['contour'].item(),
                'loss_area': losses['area'].item(),
                'loss_pin': losses['pin'].item(),
                'loss_potts': losses['potts'].item(),
                'loss_boundary_length': losses['boundary_length'].item(),
                'loss_triple': losses['triple'].item(),
                
                # Metrics
                'area_fractions': area_frac.detach().cpu().numpy(),
                'area_deviation': area_dev,
                
                # Mesh data
                'edge_idx': edge_idx.cpu().numpy(),
                'tri_area': tri_area.cpu().numpy(),
                
                # Stage info
                'lambda_smooth': stage.lambda_smooth,
                'lambda_contour': stage.lambda_contour,
                'lambda_area': stage.lambda_area,
                'lambda_pin': stage.lambda_pin,
            }
            
            # Add boundary stats if available
            if step % 400 == 0:
                if use_v2_contour:
                    boundary_length, active_frac, median_phi = compute_boundary_stats_v2(
                        F, edge_idx, verts, temp_ctrl.beta_contour
                    )
                    checkpoint_data['median_phi'] = median_phi
                else:
                    boundary_length, active_frac = compute_boundary_stats(
                        F, edge_idx, verts, temp_ctrl.beta_contour
                    )
                checkpoint_data['boundary_length'] = boundary_length
                checkpoint_data['active_edge_fraction'] = active_frac
            
            # Save as compressed npz
            np.savez_compressed(
                output_dir / f'checkpoint_{step:06d}.npz',
                **checkpoint_data
            )
            
            # Also save optimizer state separately (for resuming)
            torch.save({
                'optimizer_state': optimizer.state_dict(),
                'temp_ctrl_state': temp_ctrl.__dict__,
                'scheduler_state': scheduler.current_stage_idx
            }, output_dir / f'checkpoint_{step:06d}_optimizer.pt')
        
    
    # Final hard pinning to ensure exact values
    with torch.no_grad():
        F.data[pinned_indices] = pin_targets
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\nTraining completed in {elapsed:.1f} seconds")
    
    return F.data, history


def main():
    parser = argparse.ArgumentParser(description='Improved ReLU Mesh Segmentation Training')
    parser.add_argument('--mesh', type=str, required=True,
                       help='Path to mesh file (VTK/VTU format)')
    parser.add_argument('--output-dir', type=str, default='results_improved',
                       help='Output directory for results')
    parser.add_argument('--n-steps', type=int, default=200000,
                       help='Number of training steps')
    parser.add_argument('--n-channels', type=int, default=6,
                       help='Number of segmentation channels')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='PyTorch device (cuda/cpu)')
    parser.add_argument('--checkpoint-freq', type=int, default=5000,
                       help='Checkpoint frequency')
    parser.add_argument('--log-freq', type=int, default=500,
                       help='Logging frequency')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint npz file')
    parser.add_argument('--use-soft-pairs', action='store_true',
                       help='Use soft all-pairs contour loss for better stability')
    parser.add_argument('--use-5-patch-prior', action='store_true',
                       help='Use non-uniform area prior for 5 patches instead of 6')
    parser.add_argument('--no-v2-contour', action='store_true',
                       help='Disable improved v2 contour loss (use original instead)')
    parser.add_argument('--no-improved-area', action='store_true',
                       help='Disable improved area loss with reverse KL (use original instead)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    print(f"Using device: {args.device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load mesh
    print(f"Loading mesh from {args.mesh}...")
    vertices_np, faces_np = load_volume_tet_mesh_and_extract_surface(args.mesh)
    print(f"Loaded mesh: {vertices_np.shape[0]} vertices, {faces_np.shape[0]} faces")
    
    # Preprocess mesh
    print("Preprocessing mesh...")
    mesh_data = preprocess_mesh(vertices_np, faces_np, device=args.device)
    
    # Print mesh statistics
    stats = mesh_data['stats']
    print(f"Mesh statistics:")
    print(f"  - Bounding box diagonal: {stats['bbox_diagonal']:.3f}")
    print(f"  - Total area: {stats['total_area']:.3f}")
    print(f"  - Mean edge length: {stats['mean_edge_length']:.3f}")
    
    # Train
    if args.resume_from:
        print(f"\nResuming training from checkpoint...")
    else:
        print(f"\nStarting training for {args.n_steps} steps...")
    
    F_optimized, history = train_mesh_segmentation(
        mesh_data,
        n_channels=args.n_channels,
        n_steps=args.n_steps,
        device=args.device,
        output_dir=output_dir,
        checkpoint_freq=args.checkpoint_freq,
        log_freq=args.log_freq,
        verbose=True,
        resume_from=args.resume_from,
        use_soft_pairs=args.use_soft_pairs,
        use_5_patch_prior=args.use_5_patch_prior,
        use_v2_contour=not args.no_v2_contour,
        use_improved_area=not args.no_improved_area
    )
    
    # Save results
    print("\nSaving results...")
    
    # Get final boundary stats
    if not args.no_v2_contour:
        final_boundary_length, final_active_frac, final_median_phi = compute_boundary_stats_v2(
            F_optimized, mesh_data['edge_idx'], mesh_data['vertices'], 
            history['beta_contour'][-1] if history['beta_contour'] else 8.0
        )
    else:
        final_boundary_length, final_active_frac = compute_boundary_stats(
            F_optimized, mesh_data['edge_idx'], mesh_data['vertices'], 
            history['beta_contour'][-1] if history['beta_contour'] else 8.0
        )
    
    # Comprehensive final result
    final_data = {
        # Mesh data
        'vertices': vertices_np,
        'faces': faces_np,
        'edge_idx': mesh_data['edge_idx'].cpu().numpy(),
        'tri_area': mesh_data['tri_area'].cpu().numpy(),
        
        # Optimization result
        'field_values': F_optimized.cpu().numpy(),
        'pinned_indices': mesh_data['pinned_indices'].cpu().numpy(),
        
        # Final metrics
        'final_loss': history['loss'][-1] if history['loss'] else 0.0,
        'final_beta_contour': history['beta_contour'][-1] if history['beta_contour'] else 8.0,
        'final_beta_area': history['beta_area'][-1] if history['beta_area'] else 4.0,
        'final_boundary_length': final_boundary_length,
        'final_active_edge_fraction': final_active_frac,
        'final_area_fractions': history['area_fractions'][-1] if history['area_fractions'] else None,
        
        # Training info
        'total_steps': args.n_steps,
        'mesh_file': args.mesh,
        'n_channels': args.n_channels,
        
        # Mesh statistics
        'bbox_diagonal': mesh_data['stats']['bbox_diagonal'],
        'total_area': mesh_data['stats']['total_area'],
        'num_vertices': mesh_data['stats']['num_vertices'],
        'num_faces': mesh_data['stats']['num_faces'],
    }
    
    # Save comprehensive npz
    np.savez_compressed(
        output_dir / 'final_result.npz',
        **final_data
    )
    
    # Save history as JSON
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json_history = convert_to_json_serializable(history)
        json.dump(json_history, f, indent=2)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()


ChatGPT said:
Request for GPT-5 Pro
Reasoned for 5m 57s
heck yes — here’s a single, drop‑in loss module that bakes in all the stabilizers + convergence tricks you’ve been exploring (intrinsic 3D contour alignment, cotan smoothness, robust area balance, soft pins, Potts/TV, triple‑point barrier, boundary‑length, normal alignment, etc.). It’s designed to be plug‑and‑play with your tensors and easy to weight/schedule. I kept the numerics super defensive (float64 geometry where it matters, safe normalizations, soft‑OR coverage, Charbonnier, gating by gradient magnitude & edge length, NaN guards, straight‑through counting for area, etc.).

This is my “one file, one call” way to compute: total loss, all parts, and useful diagnostics. It should fit right into your training loop. (It also matches the intent of your report’s sections on contour/area/smoothness scheduling .)

loss_core.py (rewritten + stabilized)
python
Copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# ------------------------------
# Small numerics helpers
# ------------------------------

def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

@torch.no_grad()
def _nan_fix_(t: torch.Tensor, clamp: Optional[Tuple[float,float]] = None):
    torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp is not None:
        t.clamp_(clamp[0], clamp[1])

def grad3d_intrinsic(h_vals: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    # h_vals: (B,3) ; v*: (B,3)
    # Returns gradient in R^3 lying in the triangle plane
    e0 = v1 - v0
    e1 = v2 - v0
    b  = torch.stack([h_vals[:,1]-h_vals[:,0], h_vals[:,2]-h_vals[:,0]], dim=1)

    dt = torch.float64
    e0d, e1d, bd = e0.to(dt), e1.to(dt), b.to(dt)
    G00 = (e0d*e0d).sum(1)
    G01 = (e0d*e1d).sum(1)
    G11 = (e1d*e1d).sum(1)
    det = (G00*G11 - G01*G01).clamp_min(1e-10)

    invG00 =  G11 / det
    invG01 = -G01 / det
    invG11 =  G00 / det
    a0 = invG00*bd[:,0] + invG01*bd[:,1]
    a1 = invG01*bd[:,0] + invG11*bd[:,1]
    gd = a0[:,None]*e0d + a1[:,None]*e1d
    g = gd.to(h_vals.dtype)
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

# ------------------------------
# Pair-all intrinsic 3D contour alignment (defensive)
# ------------------------------

def contour_allpairs_v3(
    Fv: torch.Tensor,           # (N,C)
    faces: torch.Tensor,        # (T,3)
    edge_idx: torch.Tensor,     # (E,2)
    edge_tris: torch.Tensor,    # (E,2) -1 for boundary
    verts: torch.Tensor,        # (N,3)
    beta: float = 6.0,
    eps: float = 1e-6,
    return_phi: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Robust 3D contour alignment:
      * all channel pairs per edge
      * soft-OR coverage φ for "is-boundary-ness"
      * intrinsic gradient/tangent (in-plane) for each side
      * Charbonnier on misalignment
      * gradient+length gating to stabilize weights
    Returns: (scalar_loss, phi_per_edge)
    """
    device, dtype = Fv.device, Fv.dtype
    N, C = Fv.shape
    edge_idx = edge_idx.to(device)
    edge_tris = edge_tris.to(device)
    faces = faces.to(device)
    verts = verts.to(device)

    # Keep only interior edges (have triangles on both sides)
    valid = (edge_tris[:,0] >= 0) & (edge_tris[:,1] >= 0)
    if not torch.any(valid):
        zero = Fv.new_tensor(0.0)
        return zero, torch.zeros(edge_idx.shape[0], device=device, dtype=dtype)

    va = edge_idx[valid,0]
    vb = edge_idx[valid,1]
    tL = edge_tris[valid,0]
    tR = edge_tris[valid,1]

    # Build all channel-pair indices (upper-triangular)
    ii, jj = torch.triu_indices(C, C, offset=1, device=device)

    # Endpoint differences for all pairs
    da = Fv[va][:,ii] - Fv[va][:,jj]   # (E,P)
    db = Fv[vb][:,ii] - Fv[vb][:,jj]   # (E,P)

    # Crossing likelihood per pair on the edge
    w_pairs = torch.sigmoid(-beta * da * db)                       # sign-change
    conf    = 0.5*(da.abs() + db.abs())
    w_pairs = w_pairs * torch.sigmoid(5.0 * (conf - 0.2))          # confidence gate
    w_pairs = w_pairs.clamp_min(1e-6)

    # Soft-OR coverage φ per edge (how boundary-like this edge is)
    phi = 1.0 - torch.prod(1.0 - w_pairs, dim=1)                   # (E,)

    # Triangle vertices on each side
    fL = faces[tL] ; fR = faces[tR]                                # (E,3)
    FL = Fv[fL] ; FR = Fv[fR]                                      # (E,3,C)

    # h values for each pair (E,3,P)
    hL = FL[:,:,ii] - FL[:,:,jj]
    hR = FR[:,:,ii] - FR[:,:,jj]

    def _grad_many(h, face_idx):
        # h: (E,3,P) -> g: (E,P,3)
        E, _, P = h.shape
        v0, v1, v2 = [verts[face_idx[:,k]] for k in (0,1,2)]
        h_flat = h.permute(0,2,1).reshape(E*P, 3)
        g = grad3d_intrinsic(h_flat, v0.repeat_interleave(P,0),
                                       v1.repeat_interleave(P,0),
                                       v2.repeat_interleave(P,0))
        return g.reshape(E, P, 3), v0, v1, v2

    gL, v0L, v1L, v2L = _grad_many(hL, fL)
    gR, v0R, v1R, v2R = _grad_many(hR, fR)

    def _normals(v0, v1, v2):
        n = torch.cross(v1 - v0, v2 - v0, dim=1)
        return safe_normalize(n, dim=1, eps=1e-6)

    nL = _normals(v0L, v1L, v2L)
    nR = _normals(v0R, v1R, v2R)

    # Project gradients into triangle planes (stability)
    def _proj(g, n):
        dot = (g * n[:,None,:]).sum(2, keepdim=True)
        return g - dot * n[:,None,:]

    gL = _proj(gL, nL)
    gR = _proj(gR, nR)

    # Tangents τ = n × g, normalized
    tauL = safe_normalize(torch.cross(nL[:,None,:], gL, dim=2), dim=2, eps=1e-6)    # (E,P,3)
    tauR = safe_normalize(torch.cross(nR[:,None,:], gR, dim=2), dim=2, eps=1e-6)

    # Misalignment (expected over pairs)
    cosang = (tauL * tauR).sum(2).abs().clamp_max(1.0)  # (E,P)
    mis    = 1.0 - cosang

    # Per-edge mixing over pairs (prevent cherry-picking)
    mix = w_pairs / (w_pairs.sum(1, keepdim=True) + eps)
    mis_edge = (mix * mis).sum(1)                                     # (E,)

    # Gates: gradient strength & edge length
    gLm = gL.norm(dim=2).mean(dim=1) + eps
    gRm = gR.norm(dim=2).mean(dim=1) + eps
    grad_gate = torch.sqrt(gLm * gRm)
    med = torch.nanmedian(grad_gate).detach().clamp_min(1e-6)
    grad_gate = (grad_gate / med).clamp(0.0, 2.0)
    grad_gate = torch.nan_to_num(grad_gate, nan=0.0, posinf=2.0, neginf=0.0)

    evec = verts[vb] - verts[va]
    elen = evec.norm(dim=1)
    len_gate = (elen / (elen.mean().detach() + 1e-12)).clamp(0.5, 2.0)

    # Final weights per edge (use φ as coverage + gates)
    w_edge = (phi.clamp_min(1e-4) * grad_gate * len_gate)

    # Charbonnier
    loss_edge = torch.sqrt(mis_edge * mis_edge + 1e-6)
    loss = (w_edge * loss_edge).sum() / (w_edge.sum() + 1e-9)

    # Expand back to E-length phi if requested
    if return_phi:
        full_phi = torch.zeros(edge_idx.shape[0], dtype=dtype, device=device)
        full_phi[valid] = phi
        return loss, full_phi
    return loss, torch.zeros(edge_idx.shape[0], dtype=dtype, device=device)

# ------------------------------
# Cotangent Laplacian smoothness
# ------------------------------

def loss_smooth_cotan(Fv: torch.Tensor, I: torch.Tensor, J: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    diff = Fv[I] - Fv[J]               # (K,C)
    num  = (W * (diff*diff).sum(-1)).sum()
    den  = W.sum().clamp_min(1e-12)
    return num / den

# ------------------------------
# Area balance (straight-through + robust box barrier)
# ------------------------------

def loss_area_balance(
    Fv: torch.Tensor,
    faces: torch.Tensor,
    tri_area: torch.Tensor,
    beta: float = 3.0,
    method: str = "l2",              # "l2" | "rev_kl" | "js"
    st: bool = True,
    barrier_w: float = 0.05,
    min_frac: Optional[float] = None,
    max_frac: Optional[float] = None,
    entropy_w: float = 0.0,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, C = Fv.shape
    uniform = Fv.new_full((C,), 1.0/C)

    p = torch.softmax(beta * Fv, dim=1)          # (N,C)
    if st:
        # straight-through: forward=onehot(argmax), backward=soft
        hard = torch.zeros_like(p).scatter_(1, p.argmax(1, keepdim=True), 1.0)
        p = hard - p.detach() + p

    # per-face probs (mean of three vertices)
    pf = (p[faces[:,0]] + p[faces[:,1]] + p[faces[:,2]]) / 3.0      # (T,C)

    # area per channel + fractions
    A_c  = (tri_area[:,None] * pf).sum(0)                           # (C,)
    Atot = tri_area.sum().clamp_min(eps)
    frac = (A_c / Atot).clamp(eps, 1.0 - eps)

    if method == "l2":
        loss = ((frac - uniform)**2).sum() * C
    elif method == "rev_kl":
        loss = -(uniform * (frac + eps).log()).sum()
    elif method == "js":
        m = 0.5*(frac + uniform)
        loss = 0.5*((frac*(frac.add(eps).log() - m.add(eps).log())).sum()
                     + (uniform*(uniform.add(eps).log() - m.add(eps).log())).sum())
    else:
        raise ValueError("method must be 'l2', 'rev_kl', or 'js'")

    if min_frac is None: min_frac = 0.5 / C
    if max_frac is None: max_frac = 2.0 / C
    box = (torch.relu(min_frac - frac).pow(2) + torch.relu(frac - max_frac).pow(2)).sum()
    loss = loss + barrier_w * box

    if entropy_w > 0 and st:
        p_soft = torch.softmax(beta * Fv, dim=1)
        ent = -(p_soft * (p_soft + eps).log()).sum(1).mean()
        loss = loss + entropy_w * ent
    return loss, frac

# ------------------------------
# Other small regularizers
# ------------------------------

def loss_potts(Fv: torch.Tensor, edge_idx: torch.Tensor, phi: torch.Tensor, beta_area: float, gamma: float = 2.0) -> torch.Tensor:
    # Potts/TV on probabilities, gated away from boundaries
    p = torch.softmax(beta_area * Fv, dim=1)
    va, vb = edge_idx.T
    pa, pb = p[va], p[vb]
    potts = 1.0 - (pa * pb).sum(dim=1)      # = 0.5||pa-pb||^2
    gate = (1.0 - phi.detach()).pow(gamma)
    return (gate * potts).sum() / (gate.sum() + 1e-9)

def loss_total_variation(Fv: torch.Tensor, edge_idx: torch.Tensor, edge_len: Optional[torch.Tensor] = None) -> torch.Tensor:
    va, vb = edge_idx.T
    d = (Fv[va] - Fv[vb]).abs().sum(dim=1)
    if edge_len is None:
        return d.mean()
    w = 1.0 / (edge_len + 1e-12)
    return (w * d).sum() / w.sum()

def loss_boundary_length(edge_idx: torch.Tensor, phi: torch.Tensor, verts: torch.Tensor) -> torch.Tensor:
    va, vb = edge_idx.T
    L = (verts[va] - verts[vb]).norm(dim=1)
    denom = L.sum().clamp_min(1e-9)
    return (phi * L).sum() / denom

def loss_triple_point_barrier(Fv: torch.Tensor, faces: torch.Tensor, tri_area: torch.Tensor, beta: float = 10.0, margin: float = 0.10) -> torch.Tensor:
    p = torch.softmax(beta * Fv, dim=1)                                   # (N,C)
    pt = (p[faces[:,0]] + p[faces[:,1]] + p[faces[:,2]])/3                # (T,C)
    top3, _ = torch.topk(pt, k=min(3, Fv.shape[1]), dim=1)                # (T,3)
    if top3.shape[1] < 3: return Fv.new_tensor(0.0)
    gap = top3[:,1] - top3[:,2]
    pen = torch.relu(margin - gap)
    w   = tri_area / (tri_area.sum() + 1e-9)
    return (w * pen).sum()

def loss_soft_pins(Fv: torch.Tensor, pin_idx: torch.Tensor, pin_target: torch.Tensor, huber_delta: float = 1.0) -> torch.Tensor:
    if pin_idx.numel() == 0:
        return Fv.new_tensor(0.0)
    diff = Fv[pin_idx] - pin_target
    absd = diff.abs()
    quad = torch.minimum(absd, torch.full_like(absd, huber_delta))
    lin  = absd - quad
    return (0.5*quad**2 + huber_delta*lin).mean()

def loss_margin_sharpen(Fv: torch.Tensor, edge_idx: torch.Tensor, phi: torch.Tensor, tau: float = 0.3) -> torch.Tensor:
    # Sharpen only far from boundaries (phi small)
    nb = (phi.detach() < 0.2)
    if not torch.any(nb): return Fv.new_tensor(0.0)
    va = edge_idx[nb,0]; vb = edge_idx[nb,1]
    def margin(v):
        top2,_ = torch.topk(Fv[v], k=min(2, Fv.shape[1]), dim=1)
        if top2.shape[1] < 2: return Fv.new_zeros(v.shape[0])
        return top2[:,0] - top2[:,1]
    m = torch.minimum(margin(va), margin(vb))
    return torch.relu(tau - m).mean()

def loss_normals_axis(
    verts: torch.Tensor, faces: torch.Tensor, tri_area: torch.Tensor,
    Fv: torch.Tensor, beta_area: float, axis_per_channel: torch.Tensor, eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor]:
    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=1)
    n = safe_normalize(n, dim=1, eps=1e-6)                                # (T,3)

    p_v = torch.softmax(beta_area * Fv, dim=1)
    p_f = (p_v[faces[:,0]] + p_v[faces[:,1]] + p_v[faces[:,2]]) / 3.0     # (T,C)
    w = tri_area[:,None] * p_f
    m = (w[:,:,None] * n[:,None,:]).sum(0)
    m = safe_normalize(m, dim=1, eps=1e-6)

    axes = safe_normalize(axis_per_channel, dim=1, eps=1e-6)
    mis = 1.0 - (m * axes).sum(1).abs()
    loss_align = mis.mean()

    cos = (n[:,None,:] * m[None,:,:]).sum(2).clamp(-1,1)
    disp = (1.0 - cos**2)
    loss_disp = (w*disp).sum(0) / (w.sum(0)+eps)
    return loss_align.mean(), loss_disp.mean()

# ------------------------------
# Loss wrapper (one call)
# ------------------------------

@dataclass
class LossWeights:
    smooth: float = 1.0
    contour: float = 0.1
    area: float = 0.1
    pin: float = 0.01
    potts: float = 0.1
    tv: float = 0.0
    boundary_len: float = 1e-3
    normals_align: float = 0.0
    normals_disp: float = 0.0
    triple: float = 0.03
    sharpen: float = 0.0

class MeshSegLoss(nn.Module):
    """
    One-call stabilized energy for multi-channel mesh segmentation.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Fv: torch.Tensor,                      # (N,C)
        verts: torch.Tensor, faces: torch.Tensor,
        edge_idx: torch.Tensor, edge_tris: torch.Tensor,
        tri_area: torch.Tensor,
        cotan_I: torch.Tensor, cotan_J: torch.Tensor, cotan_W: torch.Tensor,
        pin_idx: Optional[torch.Tensor] = None,
        pin_target: Optional[torch.Tensor] = None,
        axis_per_channel: Optional[torch.Tensor] = None,
        # Temps
        beta_contour: float = 6.0, beta_area: float = 3.0,
        # Weights
        w: LossWeights = LossWeights(),
        # Area options
        area_method: str = "l2",
        area_st: bool = True,
        area_barrier_w: float = 0.05,
        area_entropy_w: float = 0.0,
        # Extras
        tv_edge_len: Optional[torch.Tensor] = None,
        sharpen_tau: float = 0.2,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        parts: Dict[str, torch.Tensor] = {}
        aux: Dict[str, torch.Tensor] = {}

        # --- Core terms ---
        parts['smooth'] = loss_smooth_cotan(Fv, cotan_I, cotan_J, cotan_W)

        parts['contour'], phi = contour_allpairs_v3(
            Fv, faces, edge_idx, edge_tris, verts, beta=beta_contour, return_phi=True
        )
        aux['phi'] = phi

        parts['area'], frac = loss_area_balance(
            Fv, faces, tri_area, beta=beta_area, method=area_method,
            st=area_st, barrier_w=area_barrier_w, entropy_w=area_entropy_w
        )
        aux['area_frac'] = frac

        if pin_idx is not None and pin_target is not None:
            parts['pin'] = loss_soft_pins(Fv, pin_idx, pin_target)
        else:
            parts['pin'] = Fv.new_tensor(0.0)

        # --- Helpful regularizers ---
        parts['potts'] = loss_potts(Fv, edge_idx, phi, beta_area, gamma=2.0)

        parts['boundary_len'] = loss_boundary_length(edge_idx, phi, verts)

        parts['tv'] = loss_total_variation(Fv, edge_idx, tv_edge_len) if w.tv > 0 else Fv.new_tensor(0.0)

        parts['triple'] = loss_triple_point_barrier(Fv, faces, tri_area, beta=max(8.0, beta_area), margin=0.10)

        parts['sharpen'] = loss_margin_sharpen(Fv, edge_idx, phi, tau=sharpen_tau)

        # Optional normal-axis terms
        if axis_per_channel is not None and (w.normals_align > 0 or w.normals_disp > 0):
            la, ld = loss_normals_axis(verts, faces, tri_area, Fv, beta_area, axis_per_channel)
            parts['normals_align'] = la
            parts['normals_disp']  = ld
        else:
            parts['normals_align'] = Fv.new_tensor(0.0)
            parts['normals_disp']  = Fv.new_tensor(0.0)

        # --- Total ---
        total = (
            w.smooth        * parts['smooth'] +
            w.contour       * parts['contour'] +
            w.area          * parts['area'] +
            w.pin           * parts['pin'] +
            w.potts         * parts['potts'] +
            w.boundary_len  * parts['boundary_len'] +
            w.tv            * parts['tv'] +
            w.triple        * parts['triple'] +
            w.sharpen       * parts['sharpen'] +
            w.normals_align * parts['normals_align'] +
            w.normals_disp  * parts['normals_disp']
        )

        # NaN/Inf defense (soft recovery)
        if not torch.isfinite(total):
            for k,v in parts.items():
                _nan_fix_(v)
            _nan_fix_(total)
        return total, parts, aux
How to use it in your loop
python
Copy
# Build once
loss_fn = MeshSegLoss().to(device)

# Each step
weights = LossWeights(
    smooth=0.3,
    contour=0.06,
    area=0.5,
    pin=0.02,
    potts=0.12,
    tv=0.0,
    boundary_len=1e-3,
    normals_align=0.05,
    normals_disp=0.02,
    triple=0.05,
    sharpen=0.06,
)

total, parts, aux = loss_fn(
    F, verts, faces, edge_idx, edge_tris, tri_area,
    I, J, W,
    pin_idx=pinned_indices,
    pin_target=pin_targets,
    axis_per_channel=torch.tensor([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], device=F.device, dtype=F.dtype)[:F.shape[1]],

    beta_contour=temp_ctrl.beta_contour,   # plug your gated temps here
    beta_area=temp_ctrl.beta_area,

    w=weights,
    area_method="l2",                      # "l2" gives strong, clean gradients
    area_st=True,                          # straight-through hard counting (aligns with argmax)
    area_barrier_w=0.02,                   # stronger early, taper later
    area_entropy_w=0.1 if step < 10_000 else 0.0,

    tv_edge_len=None,                      # if you precomputed edge lengths, pass them
    sharpen_tau=0.2,
)

# Backprop
optimizer.zero_grad(set_to_none=True)
total.backward()
torch.nn.utils.clip_grad_norm_([F], 5.0)
optimizer.step()
Why this tends to converge (quick notes)
Contour alignment uses all-pairs soft coverage, intrinsic 3D gradients, and Charbonnier—no plane SVDs, no brittle top‑k only; it mixes pairs per edge and gates by local gradient strength + edge length so shaky triangles don’t dominate.

Area balance is straight‑through (forward hard, backward soft) to agree with the argmax you’ll use for viz, with a box barrier around 
1
/
𝐶
1/C to keep all channels alive and optional entropy warmup for decisiveness.

Smoothness is cotan Laplacian (scale‑aware) not unweighted edge diffs.

Pins are Huber for robustness (you can still flip on hard pins at the very end).

Potts/TV runs on probabilities, gated away from boundaries using φ so it kills speckles inside patches without smearing seams.

Triple‑point barrier keeps Y‑junction spam under control.

Boundary‑length is normalized by total length (scale‑invariant) to discourage zigzags.

Normals alignment/dispersion (optional) helps axis‑oriented patches on shapes where that prior makes sense.

If you want, I can also bundle a tiny “stage → weights” shim and a progress‑gated temperature scaler to mirror your scheduler (the class here drops right into that setup). And if you hit any specific NaN or plateau regimes, send me a quick snippet of parts/aux at that step and I’ll tighten the gates even further.