# Pipeline Specification: Geodesic Voronoi via Argmin over Per‑Seed Distance Channels

**Goal.** Learn per‑vertex, per‑seed scalar fields \(f_k:\,V\to\mathbb{R}\) such that \(f_k(x)\approx d(x,s_k)\) (geodesic distance from seed \(s_k\) to point \(x\) on the surface).  
**Segmentation.** The geodesic Voronoi label at any point \(x\) is \(\operatorname*{arg\,min}_k f_k(x)\).  
**Robustness.** All constraints are intrinsic (edge lengths, face areas, in‑face gradients), so the method works on uneven/skinny meshes.

---

## 0. Notation & Inputs

- **Mesh:** Triangular surface mesh \(\partial M=(V,\mathcal{T})\), with
  - vertex positions \(X_v\in\mathbb{R}^3\), \(v\in V\), \(|V|=n\),
  - faces \(\tau=(v_0,v_1,v_2)\in\mathcal{T}\), \(|\mathcal{T}|=m\).
- **Seeds:** \(S=\{s_1,\dots,s_C\}\subseteq V\), one seed per channel.
- **Channels:** Trainable per‑vertex values \(f(v)\in\mathbb{R}^C\). Channel \(k\) is \(f_k(\cdot)\).
- **Units:** Normalize geometry so that the **mean edge length** is 1:
  \[
  h=\frac{1}{|\mathcal{E}|}\sum_{e=(a,b)\in\mathcal{E}}\|X_b-X_a\|_2,\quad X_v\leftarrow X_v/h.
  \]
  (Store the scale \(h\) to rescale distances back after training.)

---

## 1. Precomputation (deterministic, once)

### 1.1 Edges and adjacency
- Build undirected **interior edge** set \(\mathcal{E}\). For each \(e=(a,b)\in\mathcal{E}\), store:
  - incident faces \(L, R \in \mathcal{T}\) (left/right w.r.t. edge orientation; order arbitrary but fixed),
  - edge length \(\ell_e=\|X_b-X_a\|_2\).

### 1.2 Per‑face geometry
For each face \(\tau=(v_0,v_1,v_2)\):
- Edge vectors \(E_0 = X_{v_1}-X_{v_0}\), \(E_1 = X_{v_2}-X_{v_0}\).
- Area \(A(\tau)=\tfrac{1}{2}\|E_0\times E_1\|_2\).
- Unit normal \(N_\tau=\frac{E_0\times E_1}{\|E_0\times E_1\|_2+\varepsilon_n}\) with \(\varepsilon_n=10^{-15}\).
- Gram matrix
  \[
  G_\tau=\begin{bmatrix}\langle E_0,E_0\rangle & \langle E_0,E_1\rangle\\ \langle E_1,E_0\rangle & \langle E_1,E_1\rangle\end{bmatrix}.
  \]
- **Stable inverse of a \(2\times2\) Gram** (double precision):
  \[
  \det G=d=G_{00}G_{11}-G_{01}G_{10}.
  \]
  If \(d<\delta_{\det}\) (default \(\delta_{\det}=10^{-14}\cdot(G_{00}+G_{11})\)), use **Tikhonov**:
  \[
  G^{-1}\approx\big(G+\delta_{\mathrm{tik}} I\big)^{-1},\quad \delta_{\mathrm{tik}}=\max(\delta_{\det},10^{-12}).
  \]
  Otherwise, exact inverse:
  \[
  G^{-1}=\frac{1}{d}\begin{bmatrix} G_{11} & -G_{01}\\ -G_{10} & G_{00}\end{bmatrix}.
  \]

### 1.3 Areas, medians, means
- Total area \(A_{\mathrm{tot}}=\sum_{\tau}A(\tau)\).
- Mean edge length \(\bar\ell=\frac{1}{|\mathcal{E}|}\sum_{e}\ell_e\) (after normalization, \(\bar\ell\approx 1\)).
- For any per‑edge scalar \(q(e)\), define robust normalizers:
  - \(\operatorname{mean}_e q = \frac{1}{|\mathcal{E}|}\sum_e q(e)\),
  - \(\operatorname{median}_e q\) via nth‑element (linear‑time selection).

### 1.4 Seed neighborhoods (fixed masks)
For each seed \(s_k\), build a geodesic‑graph ball \(\mathcal{N}(s_k)\) by Dijkstra on \((V,\mathcal{E})\) with edge weights \(\ell_e\):
- radius \(r_0=2\,\bar\ell\),
- \(\mathcal{N}(s_k)=\{v\in V\mid \operatorname{dist}_{\text{graph}}(v,s_k)\le r_0\}\).
Cache these sets for the **seed anchor** and **Eikonal mask**.

---

## 2. Field evaluation and in‑face gradients

### 2.1 Per‑face interpolation
For a face \(\tau=(v_0,v_1,v_2)\) and barycentric coords \(\lambda=(\lambda_0,\lambda_1,\lambda_2),\ \sum\lambda_i=1\):
\[
f_k(p)=\sum_{i=0}^{2}\lambda_i f_k(v_i).
\]

### 2.2 In‑face gradient of a scalar field (per channel)
Given per‑vertex values \((f_k(v_0),f_k(v_1),f_k(v_2))\) on \(\tau\):
\[
b=\begin{bmatrix} f_k(v_1)-f_k(v_0)\\ f_k(v_2)-f_k(v_0)\end{bmatrix},\qquad
\begin{bmatrix}a_0\\ a_1\end{bmatrix}=G_\tau^{-1}b,
\]
\[
\nabla_\tau f_k = a_0 E_0 + a_1 E_1.
\]
**Project** to the face plane (robustness):
\[
\tilde\nabla_\tau f_k = \nabla_\tau f_k - \big(\nabla_\tau f_k\cdot N_\tau\big)N_\tau.
\]
Use the same for pairwise differences \(g_{ij}=f_i-f_j\).

---

## 3. Edge activity and bisector gating (no gradient through gates)

For edge \(e=(v_a,v_b)\), define for each **unordered pair** \((i,j),\,i<j\):
- Endpoint differences: \(F_a^{ij}=f_i(v_a)-f_j(v_a),\ F_b^{ij}=f_i(v_b)-f_j(v_b)\).
- Logistic **sign‑change** detector with confidence gate:
  \[
  \sigma(x)=\frac{1}{1+e^{-x}},\quad
  \tilde w_{ij}(e)=\sigma\!\big(-\beta_e F_a^{ij}F_b^{ij}\big)\cdot \sigma\!\Big(\gamma\big(\tfrac{|F_a^{ij}|+|F_b^{ij}|}{2}-\tau_0\big)\Big),
  \]
  default \(\beta_e=6,\ \gamma=5,\ \tau_0=0.2\).
- Mixture:
  \[
  \pi_{ij}(e)=\frac{\tilde w_{ij}(e)}{\sum_{p<q}\tilde w_{pq}(e)+\varepsilon_\pi},\qquad \varepsilon_\pi=10^{-12}.
  \]
- Soft‑OR activity:
  \[
  \phi(e)=1-\prod_{p<q}\big(1-\tilde w_{pq}(e)\big).
  \]
- Gradient/length gates:
  For each incident face \(\tau\in\{L,R\}\), compute
  \(\bar g_\tau(e)=\operatorname{mean}_{i<j}\|\tilde\nabla_\tau g_{ij}\|_2\),
  then
  \[
  \Gamma_{\mathrm{grad}}(e)=\sqrt{\bar g_L(e)\,\bar g_R(e)},\quad
  \widehat\Gamma_{\mathrm{grad}}(e)=\operatorname{clip}\!\left(\frac{\Gamma_{\mathrm{grad}}(e)}{\operatorname{median}_e \Gamma_{\mathrm{grad}}(e)},\,0,\,2\right),
  \]
  \[
  \Lambda_{\mathrm{len}}(e)=\operatorname{clip}\!\left(\frac{\ell_e}{\operatorname{mean}_e \ell_e},\,0.5,\,2\right).
  \]
- Final **edge weight**:
  \[
  w_e=\max(\phi(e),10^{-4})\cdot \widehat\Gamma_{\mathrm{grad}}(e)\cdot \Lambda_{\mathrm{len}}(e).
  \]

**Implementation:** Compute \(\tilde w, \pi, \phi, w_e\) with **no gradient** (stop‑grad / `.detach()`), to prevent the model from trivially turning gates off.

**Efficiency (optional):** For each edge, restrict to **active pairs** formed by the union of top‑\(K\) channels (by softmin \(\alpha_k\), §4.1) at \(v_a\) and \(v_b\); small \(K\in\{2,3\}\) makes the pair set \(\le 9\).

---

## 4. Softmin and masks

### 4.1 Softmin (temperature \(\beta\))
For any vertex or face‑sample \(x\):
\[
\alpha_k^{(\beta)}(x)=\frac{\exp(-\beta f_k(x))}{\sum_{t=1}^{C}\exp(-\beta f_t(x))},\qquad
d_\beta(x)=-\frac{1}{\beta}\log\sum_{t=1}^{C}\exp(-\beta f_t(x)).
\]
Use \(\beta\in[2,30]\) (anneal upward during training). Hard \(\arg\min\) only at inference.

### 4.2 Eikonal masks
- **Seed mask** \(M_{\text{seed}}(\tau,k)=1\) if any vertex of \(\tau\) is in \(\mathcal{N}(s_k)\), else 0.
- **Bisector mask** \(M_{\text{bis}}(\tau)=1\) if \(\max_{e\subset \partial\tau}\phi(e)\ge \tau_{\text{bis}}\) (default \(\tau_{\text{bis}}=0.3\)), else 0.
- **Eikonal active mask:** \(M_{\text{eik}}(\tau,k)=1-M_{\text{seed}}(\tau,k)\,\lor\,M_{\text{bis}}(\tau)\) (i.e., active if **not** near seeds **and** not on detected bisectors).

---

## 5. Losses

Define **Charbonnier** \(\rho_\delta(t)=\sqrt{t^2+\delta^2}\) with \(\delta=10^{-6}\).

### 5.1 Seed anchoring (per seed \(k\))
Let margin \(m=r_0\) and dominance weight \(\mu=0.5\):
\[
L_{\text{seed}}=\frac{1}{C}\sum_{k=1}^{C}\left[
\frac{1}{|\mathcal{N}(s_k)|}\sum_{v\in\mathcal{N}(s_k)} f_k(v)^2\;+\;
\frac{\mu}{(C-1)|\mathcal{N}(s_k)|}\sum_{j\ne k}\sum_{v\in\mathcal{N}(s_k)} \big[m-(f_j(v)-f_k(v))\big]_+^2
\right].
\]

### 5.2 Eikonal residual (unit speed in faces, masked)
\[
L_{\text{eik}}=\frac{1}{C\,A_{\mathrm{tot}}}\sum_{k=1}^{C}\ \sum_{\tau\in\mathcal{T}} A(\tau)\, M_{\text{eik}}(\tau,k)\; \rho_\delta\!\left(\ \|\tilde\nabla_\tau f_k\|_2-1\ \right).
\]

### 5.3 Edgewise 1‑Lipschitz hinge (speed limit per edge)
\[
L_{\text{lip}}=\frac{1}{C\,|\mathcal{E}|}\sum_{k=1}^{C}\ \sum_{e=(a,b)\in\mathcal{E}}\ \Big[\ |f_k(a)-f_k(b)|-\ell_e\ \Big]_+^2.
\]

### 5.4 Bisector Hamilton–Jacobi (HJ) boundary law (only on active edges)
For each interior edge \(e\) with faces \(L,R\) and pair \((i,j)\):
- Per‑face **pair normal**:
  \[
  \tilde\nabla^{L}_{ij}=\tilde\nabla^{L} f_i - \tilde\nabla^{L} f_j,\quad
  \tilde\nabla^{R}_{ij}=\tilde\nabla^{R} f_i - \tilde\nabla^{R} f_j,
  \]
  \[
  \hat{\mathbf{n}}_{ij}^L = \frac{\tilde\nabla^{L}_{ij}}{\|\tilde\nabla^{L}_{ij}\|_2+\varepsilon_g},\quad
  \hat{\mathbf{n}}_{ij}^R = \frac{\tilde\nabla^{R}_{ij}}{\|\tilde\nabla^{R}_{ij}\|_2+\varepsilon_g},\quad \varepsilon_g=10^{-12}.
  \]
- Averaged unit **bisector normal**:
  \[
  \hat{\mathbf{n}}_{ij}(e)=\frac{\hat{\mathbf{n}}_{ij}^L+\hat{\mathbf{n}}_{ij}^R}{\|\hat{\mathbf{n}}_{ij}^L+\hat{\mathbf{n}}_{ij}^R\|_2+\varepsilon_g}.
  \]
  (If one side is missing/degenerate, use the other.)
- Per‑face **tangent** directions:
  \[
  \hat{\boldsymbol{\tau}}_{ij}^{L}=\frac{N_L\times \hat{\mathbf{n}}_{ij}^{L}}{\|N_L\times \hat{\mathbf{n}}_{ij}^{L}\|_2+\varepsilon_g},\quad
  \hat{\boldsymbol{\tau}}_{ij}^{R}=\frac{N_R\times \hat{\mathbf{n}}_{ij}^{R}}{\|N_R\times \hat{\mathbf{n}}_{ij}^{R}\|_2+\varepsilon_g},
  \]
  averaged:
  \[
  \hat{\boldsymbol{\tau}}_{ij}(e)=\frac{\hat{\boldsymbol{\tau}}_{ij}^{L}+\hat{\boldsymbol{\tau}}_{ij}^{R}}{\|\hat{\boldsymbol{\tau}}_{ij}^{L}+\hat{\boldsymbol{\tau}}_{ij}^{R}\|_2+\varepsilon_g}.
  \]
- **Averaged channel gradients** at \(e\): \(\bar\nabla f_k(e)=\tfrac{1}{2}\big(\tilde\nabla^{L} f_k + \tilde\nabla^{R} f_k\big)\).
- **Edge term** (two scalar projections):
  \[
  t_{\text{norm}}(e,i,j)=\big(\bar\nabla f_i+\bar\nabla f_j\big)\cdot \hat{\mathbf{n}}_{ij}(e),\quad
  t_{\text{tan}}(e,i,j)=\big(\bar\nabla f_i-\bar\nabla f_j\big)\cdot \hat{\boldsymbol{\tau}}_{ij}(e).
  \]
- **Loss** (mixture over pairs, gated and weighted):
  \[
  L_{\text{HJ}}=\frac{1}{\sum_{e} w_e + 10^{-12}}\sum_{e\in\mathcal{E}} w_e \sum_{i<j} \pi_{ij}(e)\ \Big(\rho_\delta\big(t_{\text{norm}}(e,i,j)\big)+\rho_\delta\big(t_{\text{tan}}(e,i,j)\big)\Big).
  \]
All gates \(\tilde w,\pi,\phi,w_e\) are **detached** from gradient.

### 5.5 Total loss
\[
L=\lambda_{\text{seed}}L_{\text{seed}}+\lambda_{\text{eik}}L_{\text{eik}}+\lambda_{\text{lip}}L_{\text{lip}}+\lambda_{\text{HJ}}L_{\text{HJ}}.
\]
**Default weights:** \((\lambda_{\text{seed}},\lambda_{\text{eik}},\lambda_{\text{lip}},\lambda_{\text{HJ}})=(1.0,\,4.0,\,2.0,\,1.0)\).

---

## 6. Initialization (choose one)

### Option A — Graph‑geodesic (Dijkstra) initialization **(simple, robust)**
For each seed \(k\), run Dijkstra on \((V,\mathcal{E})\) with edge costs \(\ell_e\) to get distances \(D_k(v)\). Initialize
\[
f_k^{(0)}(v)\leftarrow D_k(v).
\]

### Option B — Heat‑method initialization **(refined)**
1) Assemble cotangent **stiffness** \(S\) and diagonal **mass** \(M\) (lumped vertex areas). With the standard cotan weights \(w_{ij}=\tfrac{1}{2}(\cot\alpha_{ij}+\cot\beta_{ij})\):
   \[
   S_{ij}=\begin{cases}
   -w_{ij}, & i\ne j,\ (i,j)\in\mathcal{E},\\[2pt]
   \sum_{t\ne i} w_{it}, & i=j,\\
   0, & \text{otherwise},
   \end{cases}
   \quad M_{ii}=A_i\ \ (\text{Voronoi or barycentric area}).
   \]
2) Choose \(t=\bar\ell^2\) (after normalization, \(t=1\)).
3) For each seed \(k\), set \(u_0=M\,\delta_{s_k}\) (1 at seed’s mass, 0 elsewhere) and solve the SPD system
   \[
   (M + tS)\,u = u_0.
   \]
4) Compute per‑face \(\tilde\nabla_\tau u\) and the **unit vector field** \(X_\tau=-\frac{\tilde\nabla_\tau u}{\|\tilde\nabla_\tau u\|_2+\varepsilon_g}\).
5) Assemble discrete divergence \(\text{div}X\) at vertices (e.g., per face distribute \((X_\tau\cdot E_i^\perp)\) appropriately); solve
   \[
   S\,d = \text{div}X,\quad \text{then set } d(s_k)=0\ \text{(fix constant)}.
   \]
6) Initialize \(f_k^{(0)}\leftarrow d\).

Use intrinsic Delaunay flips if available for better \(S\). If unsure, prefer **Option A**.

---

## 7. Optimization protocol

- **Parameters:** \(\{f(v)\in\mathbb{R}^C\}_{v\in V}\) (one tensor of shape \(n\times C\)).
- **Optimizer:** AdamW (lr \(=5\times10^{-3}\), betas \(=(0.9,0.999)\), weight‑decay \(=10^{-4}\)).
- **Precision:** geometry & Gram ops in **float64**; parameters and losses in **float32**; cast gradients safely.
- **Gradient clipping:** global‑norm clip at 1.0.
- **Batching (recommended for large meshes):**
  - Sample face batch \(\mathcal{T}_b\) by probability \(p(\tau)\propto A(\tau)\) for \(L_{\text{eik}}\).
  - Sample edge batch \(\mathcal{E}_b\) by probability \(p(e)\propto \ell_e\) for \(L_{\text{lip}}\) and \(L_{\text{HJ}}\).
  - Always include all seed‑ring vertices for \(L_{\text{seed}}\).
  - Replace sums by sums over batches; keep the same normalizers but scale by inverse sampling prob (or simply use the **mean over batch** as an unbiased estimator).
- **Temperature anneal:** \(\beta: 2\to 20\) linearly over first 70% of steps; hold at 20.
- **Staging (stability):**
  - **Stage A (1–2k iters):** optimize \(L_{\text{seed}}+L_{\text{lip}}\) only (enforce anchors and speed limits).
  - **Stage B (next 3–5k):** add \(L_{\text{eik}}\).
  - **Stage C (final):** add \(L_{\text{HJ}}\) to sharpen seams.
- **Convergence checks:**
  - Mean Lipschitz hinge \(\to 0\).
  - Eikonal residual mean \(\approx 0\) away from masks.
  - On active edges, \(f_i\approx f_j\) and HJ terms small.

---

## 8. Inference

### 8.1 Labels and distance
- **Per‑vertex labels:** \(\ell(v)=\operatorname*{arg\,min}_k f_k(v)\).
- **Per‑face (continuous) labels:** sample a fixed set of barycentric points \(B=\{(1/3,1/3,1/3),(1/2,1/2,0),(1/2,0,1/2),(0,1/2,1/2)\}\); compute \(\ell(p)=\operatorname*{arg\,min}_k f_k(p)\).
- **Distance:** \(d(v)=\min_k f_k(v)\). Rescale by original \(h\) if geometry was normalized.

### 8.2 Boundary extraction (Voronoi seams)
An edge \(e=(a,b)\) is on the boundary iff \(\ell(a)\ne \ell(b)\). Robust version: also require \(\phi(e)\ge \tau_{\text{bis}}\) and \(|f_i(a)-f_j(a)|,|f_i(b)-f_j(b)|\le \eta\) for the winning pair \((i,j)\) at \(e\) (e.g., \(\eta=0.05\)).

---

## 9. Computational complexity & speed notes

- Per iteration (full‑batch): \(O\big(C(|\mathcal{T}|+|\mathcal{E}|)\big)\) for gradients + Lipschitz; \(O(C^2|\mathcal{E}|)\) worst‑case for pair mixtures (use **active pairs** with \(K=2\) or 3 to reduce to \(O(K^2|\mathcal{E}|)\)).
- Use **vectorized** per‑face Gram solves and per‑edge operations.
- Cache per‑face \(G_\tau^{-1}\), \(A(\tau)\), \(N_\tau\); cache \(\ell_e\), face adjacency.

---

## 10. Numerical hygiene

- **Determinant clamping** for \(G_\tau^{-1}\) (Section 1.2).
- **Projection** of gradients to face plane (Section 2.2).
- **Stop‑grad** through \(\tilde w,\pi,\phi,w_e\).
- **Mask** \(L_{\text{eik}}\) near seeds and on active bisectors (Section 4.2).
- **Charbonnier** penalty \(\rho_\delta\) instead of pure square for HJ and Eikonal.
- **Skip degenerate faces:** if \(A(\tau)<10^{-16}\), exclude \(\tau\) from \(L_{\text{eik}}\).

---

## 11. Why this converges to distance (summary of guarantees)

- **Lipschitz hinge** enforces \(|f_k(u)-f_k(v)|\le \ell_{uv}\) on every edge \(\Rightarrow f_k(x)\le\) length of **any** path from \(s_k\) to \(x\) \(\Rightarrow f_k(x)\le d(x,s_k)\).
- **Eikonal residual** drives \(\|\tilde\nabla f_k\|\to 1\) wherever allowed, i.e., it **uses** the speed budget without violating Lipschitz, pushing \(f_k\) **up** until touching the geodesic ceiling.
- **Seed anchor** fixes \(f_k(s_k)=0\).
- Together, among all anchored 1‑Lipschitz functions, the **maximal** one is the distance; the losses push \(f_k\) to that maximal function.
- **HJ law** ensures the equal‑distance sets are sharp and correctly oriented; **argmin** over channels yields the geodesic Voronoi partition.

---

## 12. Default hyperparameters (good starting point)

- \(\lambda_{\text{seed}}=1.0,\ \lambda_{\text{eik}}=4.0,\ \lambda_{\text{lip}}=2.0,\ \lambda_{\text{HJ}}=1.0\).
- \(\beta\) (softmin temperature): start 2, end 20.
- \(r_0=2\,\bar\ell\) (seed ring radius), \(m=r_0\) (anchor margin), \(\mu=0.5\).
- \(\tau_{\text{bis}}=0.3\), \(\varepsilon_n=10^{-15}\), \(\varepsilon_g=10^{-12}\), \(\varepsilon_\pi=10^{-12}\).
- Gate params: \(\beta_e=6,\ \gamma=5,\ \tau_0=0.2\).
- Optimizer: AdamW lr \(5\cdot10^{-3}\), weight‑decay \(10^{-4}\), grad clip 1.0.

---

## 13. Minimal training loop (pseudocode)

Inputs: V, T, seeds S, normalized X (mean edge = 1)
Precompute: edges E with LR faces + lengths; per-face G_inv, area, normal; seed rings; normalizers.
initialize f[n, C] # using Dijkstra (Option A) or Heat Method (Option B)

for step in 1..MaxSteps:
beta = anneal_temperature(step)
# Sample batches (or use full sets)
Tb = sample_faces_by_area(T) # for L_eik
Eb = sample_edges_by_length(E) # for L_lip, L_HJ

makefile
Copy code
# --- Forward: per-face gradients (vectorized over Tb, channels)
grad_fk_Tb = face_gradients_projected(f, Tb, G_inv, E0, E1, N)  # shape: [|Tb|, C, 3]
# For HJ: also compute pairwise difference gradients on Eb's incident faces

# --- Gates on Eb (STOP-GRAD)
w_tilde, pi, phi, w_e = compute_edge_gates_detached(f, Eb)

# --- Masks for Eikonal
M_eik = compute_eikonal_mask(Tb, seed_rings, phi, tau_bis)

# --- Losses
L_seed = seed_anchor_loss(f, seed_rings, r0, m, mu)
L_eik  = eikonal_loss(grad_fk_Tb, M_eik, areas_Tb, delta)
L_lip  = lipschitz_hinge_loss(f, Eb, edge_lengths)
L_HJ   = hj_boundary_loss(f, Eb, w_e, pi, N_LR, G_inv, E0E1_LR, delta)

L = λ_seed*L_seed + λ_eik*L_eik + λ_lip*L_lip + λ_HJ*L_HJ

optimizer.zero_grad()
L.backward()
clip_grad_norm_(f, 1.0)
optimizer.step()
yaml
Copy code

---

## 14. Output

- Distance per vertex: \(d(v)=\min_k f_k(v)\) (rescale by original \(h\) if needed).
- Voronoi label per vertex: \(\ell(v)=\operatorname*{arg\,min}_k f_k(v)\).
- Boundary edges: \(\{e=(a,b)\mid \ell(a)\ne \ell(b)\}\) (optionally gated by \(\phi\)).