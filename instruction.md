\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{lmodern}
\usepackage{hyperref}

\begin{document}

\begin{verbatim}
Below is a concrete way to take your **2D gradient‑alignment** idea and adapt it to the **3D, 6‑channel surface segmentation** you describe. I’ll (1) recap the 2D formulation, (2) show the exact 3D generalization on a triangle mesh surface, (3) explain how to handle the **6‑channel field** via pairwise differences, (4) give **soft decisions** for edge crossings and **triple intersections**, (5) extend gradient alignment to **triple points**, and (6) account for **face‑normal differences** so the loss is curvature‑aware. I’ll include step‑by‑step instructions and loss formulas you can drop into your code.

---

## 1) Reminder: 2D gradient‑alignment (what we’re generalizing)

In 2D, for a triangle with vertices \(p_1,p_2,p_3\) and scalar values \(f_1,f_2,f_3\), the piecewise‑linear field has a **constant gradient per triangle** \(\nabla f\), obtained by solving
\[
\begin{bmatrix} (p_2-p_1) & (p_3-p_1)\end{bmatrix}
\begin{bmatrix} g_x \\ g_y \end{bmatrix}
=
\begin{bmatrix} f_2-f_1 \\ f_3-f_1 \end{bmatrix}.
\]
You then **normalize** gradients in triangles that cross the zero level set and **penalize** differences of these directions across such triangles to straighten the isoline (the “gradient alignment loss”) :contentReference[oaicite:0]{index=0} (Ch. 3 §3.2.1–3.2.4, pp. 8–9; two‑triangle straightening demo in Fig. 3.1, p. 10).

---

## 2) 3D surface version: per‑face (tangential) gradient on a triangle

On a 3D surface mesh, each face is a triangle lying in some plane. For a face \(t\) with vertices \(\mathbf{x}_1,\mathbf{x}_2,\mathbf{x}_3\in\mathbb{R}^3\) and per‑vertex scalar values \(f_1,f_2,f_3\), the **tangential gradient** (still constant over the face) is the minimum‑norm vector \(\mathbf{g}_t\) that satisfies
\[
\mathbf{g}_t \cdot (\mathbf{x}_2-\mathbf{x}_1) = f_2 - f_1,\qquad
\mathbf{g}_t \cdot (\mathbf{x}_3-\mathbf{x}_1) = f_3 - f_1.
\]
A convenient, numerically stable way to compute it is to parametrize \(\mathbf{g}_t\) in the triangle’s local basis:
- Let \( \mathbf{e}_1=\mathbf{x}_2-\mathbf{x}_1,\ \mathbf{e}_2=\mathbf{x}_3-\mathbf{x}_1\) and \(E=\begin{bmatrix}\mathbf{e}_1 & \mathbf{e}_2\end{bmatrix}\in\mathbb{R}^{3\times 2}\).
- Solve \( (E^\top E)\,\mathbf{a} = \begin{bmatrix}f_2-f_1\\ f_3-f_1\end{bmatrix} \) for \(\mathbf{a}\in\mathbb{R}^2\).
- Then \(\mathbf{g}_t = E\,\mathbf{a}\in\mathbb{R}^3\) (this lies in the triangle’s plane).

You will use this routine **for any scalar field on the mesh** (we’ll apply it to channel differences in §3).

---

## 3) From 1 scalar to **6‑channel** field: work with pairwise differences

You already introduced a **6‑channel vertex field** \( \mathbf{f}(v)=[f_1,\dots,f_6] \) and assign a region by \(\arg\max_c f_c\); boundaries between regions \(i\) and \(j\) are zero sets of \(h_{ij}=f_i-f_j\). Within each face, \(h_{ij}\) is linear, so its zero set is a segment or empty—exactly what we want for piecewise‑linear boundaries :contentReference[oaicite:1]{index=1} (Ch. 4 §4.2.1–§4.2.3, eqs. (4.1)–(4.4), pp. 14–15).

**Key point for alignment:** compute a **per‑face gradient of the pairwise difference**
\[
\mathbf{g}_{ij}(t)\equiv \nabla h_{ij}\ \text{on face }t,
\]
by applying the 3D recipe above to the three per‑vertex values \(\big(h_{ij}(\mathbf{x}_1),h_{ij}(\mathbf{x}_2),h_{ij}(\mathbf{x}_3)\big)\). (Equivalently, compute \(\nabla f_i\) and \(\nabla f_j\) per face and subtract; both are constant per face.)

We will **align directions** of \(\mathbf{g}_{ij}\) across faces **only where \(i\)–\(j\) actually separates** the surface; that’s where “soft decisions” come in.

---

## 4) **Soft decision**: does an \(i\)–\(j\) boundary cross this edge / this face?

### 4.1 Edge‑level soft crossing (for pair \(i,j\))
For an edge \((a,b)\) with endpoints \(v_a,v_b\), let
\[
d_a = h_{ij}(v_a)=f_i(v_a)-f_j(v_a),\quad d_b = h_{ij}(v_b)=f_i(v_b)-f_j(v_b).
\]
A hard test \(d_a\,d_b<0\) means a sign change; instead use the **logistic soft gate**
\[
w^{\text{edge}}_{ij}(a,b)\;=\;\sigma\!\big(-\beta\,d_a\,d_b\big),\qquad \sigma(x)=\tfrac{1}{1+e^{-x}},
\]
with \(\beta>0\) ramped over training (start small, increase) to transition from soft to hard. This is exactly the **edge-intersection softening** you already used for contour alignment :contentReference[oaicite:2]{index=2} (Ch. 4 §4.3.1.1, eq. (4.6), p. 16).

> (Optional) You can also precompute the **interpolated crossing point** on the edge for diagnostics or for your plane‑fitting loss:
\[
\alpha=\frac{|d_a|}{|d_a|+|d_b|+\varepsilon},\qquad
\mathbf{p}_{ij}^{(a,b)}=\mathbf{x}_a+\alpha(\mathbf{x}_b-\mathbf{x}_a),
\]
as in eqs. (4.7)–(4.8) :contentReference[oaicite:3]{index=3} (p. 17).

### 4.2 Face‑level soft crossing (for pair \(i,j\))
For a triangle \(t\) with edges \(e_1,e_2,e_3\), lift edge gates to a **face gate** via a smooth OR:
\[
w^{\text{face}}_{ij}(t) \;=\; 1 - \prod_{k=1}^{3}\big(1 - w^{\text{edge}}_{ij}(e_k)\big).
\]
This is near 1 when any edge likely carries an \(i\)–\(j\) crossing, near 0 otherwise.

We will use \(w^{\text{edge}}\) to weight **cross‑face** comparisons across the shared edge, and \(w^{\text{face}}\) to decide whether to include the face in per‑face penalties.

---

## 5) Gradient‑alignment **across adjacent faces** (curvature‑aware)

Two neighboring faces \(t\) and \(s\) share an edge with unit direction \(\hat{\mathbf{e}}\). Their normals are \(\mathbf{n}_t\) and \(\mathbf{n}_s\). The \(i\)–\(j\) isoline ought to pass smoothly across the edge; to compare directions **coherently on a curved surface** we should **parallel‑transport** one face’s gradient to the other before comparing.

1) **Compute dihedral rotation** about \(\hat{\mathbf{e}}\).  
   Let the signed dihedral angle be \(\delta\). Build the 3D rotation \(R_{\hat{\mathbf{e}}}(\delta)\) (Rodrigues formula).

2) **Transport** the gradient from \(t\) to \(s\):  
   \(\tilde{\mathbf{g}}_{ij}(t\!\to\! s) \;=\; R_{\hat{\mathbf{e}}}(\delta)\,\mathbf{g}_{ij}(t).\)  
   (Because \(\mathbf{g}_{ij}\) is tangential, rotating about the shared edge carries it into the neighbor’s tangent plane.)

3) **Normalize** both gradients:  
   \(\widehat{\mathbf{g}}=\tilde{\mathbf{g}}/\|\tilde{\mathbf{g}}\|,\quad \widehat{\mathbf{g}}'=\mathbf{g}_{ij}(s)/\|\mathbf{g}_{ij}(s)\|.\)

4) **Weight** by the edge crossing probability:
\[
w^{\text{pair}}_{ij}(t\!\leftrightarrow\! s)\;=\;w^{\text{edge}}_{ij}(\text{shared edge}).
\]

5) **Penalty** (per channel pair and adjacent face pair):
\[
\ell_{ij}(t,s)\;=\;w^{\text{pair}}_{ij}(t\!\leftrightarrow\! s)\,\big\|\widehat{\mathbf{g}}_{ij}(t\!\to\! s)-\widehat{\mathbf{g}}_{ij}(s)\big\|^2.
\]

6) **Sum** over all channel pairs \(i<j\) (15 pairs) and all adjacent face pairs:
\[
\boxed{\;L_{\text{grad-3D}} \;=\; \sum_{i<j}\ \sum_{(t,s)\ \text{adjacent}}\ \ell_{ij}(t,s)\; }.
\]

This is the direct 3D generalization of your 2D gradient‑alignment idea, with a **necessary curvature correction** (rotation by the dihedral angle). Without it, you’d over‑penalize naturally curved boundaries and bias the optimizer toward flattening.  

> Your report already notes separate “contour alignment” ideas (edge crossings, plane fitting) for 3D; this gradient alignment complements that by explicitly **aligning the direction field** \(\nabla (f_i-f_j)\) across faces, analogous to Ch. 3 in 2D and consistent with Ch. 4’s multi‑channel setup and intersection tests :contentReference[oaicite:4]{index=4} (Ch. 3 §3.2; Ch. 4 §4.2–§4.3.1, pp. 8–18).

---

## 6) **Soft triple‑intersection** on a face and how to *use* it

A **triple intersection** of channels \(c_0,c_1,c_2\) on a face occurs when
\[
f_{c_0}(p)=f_{c_1}(p)=f_{c_2}(p),
\]
equivalently \(h_{c_0c_1}(p)=0\) and \(h_{c_0c_2}(p)=0\). Because each \(f_c\) is barycentrically linear, you can **solve for the barycentric point** \((\alpha,\beta,\gamma)\) with a 2×2 linear system (Cramer’s rule), as in eqs. (4.13)–(4.23). Then \(p^\star=\alpha\mathbf{x}_0+\beta\mathbf{x}_1+\gamma\mathbf{x}_2\) is the triple point candidate (when \(\alpha,\beta,\gamma\ge 0\)) :contentReference[oaicite:5]{index=5} (Ch. 4 §4.3.1.2, pp. 17–18).

### 6.1 Soft **decision** that a face hosts a \(c_0\!-\!c_1\!-\!c_2\) triple point
Use the **soft weight** (from your text) combining per‑vertex softmaxes and in‑simplex gates:
\[
w^{\text{triple}}_{c_0c_1c_2}(t) \;=\; \pi_0(c_0)\,\pi_1(c_1)\,\pi_2(c_2)\;\cdot\;S(\alpha)S(\beta)S(\gamma),
\]
where \(\pi_i(c)=\frac{e^{f_c(v_i)}}{\sum_k e^{f_k(v_i)}}\) and \(S(x)=\sigma(kx)\) softly enforces \(\alpha,\beta,\gamma\ge 0\) (inside triangle). This is exactly your **soft triple gate** (eqs. (4.24)–(4.25)) :contentReference[oaicite:6]{index=6} (p. 18).

### 6.2 How to **apply gradient alignment** at a triple point
Two useful, differentiable constraints use only quantities you already compute:

**(A) Triple‑point consistency (values):**  
At a true triple point \(p^\star\), *all* pairwise differences vanish. Distance from a point to the zero set of \(h_{ij}\) in a plane is \(|h_{ij}(p)|/\|\nabla h_{ij}\|\). Penalize the three residuals at \(p^\star\):
\[
\ell^{\text{triple-val}}_{c_0c_1c_2}(t)
=
w^{\text{triple}}_{c_0c_1c_2}(t)\sum_{(i,j)\in\{(0,1),(0,2),(1,2)\}}
\frac{\big(h_{c_ic_j}(p^\star)\big)^2}{\|\mathbf{g}_{c_ic_j}(t)\|^2+\varepsilon}.
\]
This pulls the three zero‑lines to **concur at the same point** on that face.

**(B) Triple‑junction **direction** coherence (in‑face):**  
On a face with unit normal \(\mathbf{n}_t\), the **isoline direction** is
\[
\mathbf{t}_{ij}(t) \;=\; \frac{\mathbf{n}_t\times \mathbf{g}_{ij}(t)}{\|\mathbf{n}_t\times \mathbf{g}_{ij}(t)\|},
\]
i.e., tangent to the zero‑line and orthogonal to \(\mathbf{g}_{ij}\). At a clean triple junction we want these three branch directions to be **well‑separated and stable**. Two options:

- **Neutral (no preferred angle):** encourage the three directions to be distinct (avoid collapsing into 1 line):
  \[
  \ell^{\text{triple-dir}}_{\text{neutral}}
  =
  w^{\text{triple}}\Big(
   (\mathbf{t}_{01}\!\cdot\!\mathbf{t}_{02})^2
  +(\mathbf{t}_{01}\!\cdot\!\mathbf{t}_{12})^2
  +(\mathbf{t}_{02}\!\cdot\!\mathbf{t}_{12})^2
  \Big).
  \]
  (Zero when directions are orthogonal; still works when the geometry chooses its own angles.)

- **120° model (T‑junctions with equal “tension”):** if you prefer equal separation (common in isotropic interfaces), aim for \(120^\circ\): \(\cos120^\circ=-\tfrac{1}{2}\):
  \[
  \ell^{\text{triple-dir}}_{120^\circ}
  =
  w^{\text{triple}}\sum_{pairs}
  \big(\mathbf{t}_{ij}\!\cdot\!\mathbf{t}_{ik} + \tfrac12\big)^2.
  \]
  
**Triple loss (on faces):**
\[
\boxed{\;L_{\text{triple}}=\sum_{t}\sum_{c_0<c_1<c_2}
\big(\lambda_{\text{val}}\ \ell^{\text{triple-val}} + \lambda_{\text{dir}}\ \ell^{\text{triple-dir}}\big)\;}
\]
with your chosen direction model.  
These terms are **in addition to** your edge/plane “contour alignment” and area/smoothness; they use the same \(p^\star\), \(\pi\), and \(S(\cdot)\) machinery already defined in your Ch. 4 (eqs. (4.9)–(4.25)) :contentReference[oaicite:7]{index=7}.

---

## 7) How **face‑normal differences** affect results (and how to handle them)

- **Problem:** On a curved surface the face normals change. If you compare \(\mathbf{g}_{ij}\) from adjacent faces *without* accounting for the dihedral angle, you will penalize geodesically straight boundaries that simply bend with the surface, which **over‑straightens** cuts in 3D space and can lead to non‑planar artifacts elsewhere (your experiments report convergence difficulty and non‑planar cuts on complex shapes) :contentReference[oaicite:8]{index=8} (Ch. 4 §4.6, pp. 30–31).

- **Fix 1 — Parallel transport (used above):** rotate one gradient by the **dihedral rotation** about the shared edge before comparing. This puts both directions into the **same tangent plane** and measures a **geodesic** notion of alignment.

- **Fix 2 — Curvature‑aware weights (optional):** damp alignment across very sharp dihedrals:
  \[
  w^{\text{curv}}(t,s)\;=\;\exp\!\big(-\kappa\,\big(1-\mathbf{n}_t\!\cdot\!\mathbf{n}_s\big)\big),
  \]
  and multiply the pair loss by \(w^{\text{curv}}\). This stops the loss from fighting genuine sharp features.

- **Fix 3 — Global plane support (optional synergy):** keep your **plane‑fitting** of intersection points (weighted covariance + SVD to get boundary planes) as a *global straightener* while gradient alignment keeps **local** directions coherent (Ch. 4 §4.3.1.3–4.3.1.4, eqs. (4.26)–(4.28)) :contentReference[oaicite:9]{index=9} (pp. 18–19). Using both often stabilizes training.

---

## 8) Full **step‑by‑step** procedure (drop‑in)

**Inputs:** mesh \((V,F)\), per‑vertex 6‑channel field \(F\in\mathbb{R}^{|V|\times 6}\). Precompute face areas, normals, adjacency, shared edges.

1) **Per‑face, per‑pair data.** For each face \(t\) and each of the \(15\) pairs \(i<j\):  
   - Build \(h_{ij}\) values at the face’s vertices.  
   - Compute \(\mathbf{g}_{ij}(t)\) via the 3D routine in §2; store its norm and normalized vector.  
   - Compute the three edge gates \(w^{\text{edge}}_{ij}\) and the face gate \(w^{\text{face}}_{ij}(t)\).

2) **Gradient alignment across edges.** For each adjacent pair \((t,s)\) sharing edge \(e\), for each \(i<j\):  
   - Weight: \(w^{\text{pair}}=w^{\text{edge}}_{ij}(e)\cdot w^{\text{face}}_{ij}(t)\cdot w^{\text{face}}_{ij}(s)\) (all soft).  
   - Transport \(\mathbf{g}_{ij}(t)\) to \(s\) by rotation about the edge axis by the dihedral angle.  
   - Add \(\ell_{ij}(t,s)=w^{\text{pair}}\|\hat{\mathbf{g}}_{ij}(t\!\to\! s)-\hat{\mathbf{g}}_{ij}(s)\|^2\) to \(L_{\text{grad-3D}}\).

3) **Triple‑point processing (per face).** For each face \(t\) and each triple \(c_0<c_1<c_2\):  
   - Solve eqs. (4.13)–(4.23) for \((\alpha,\beta,\gamma)\) and \(p^\star\).  
   - Compute \(w^{\text{triple}}\) via eq. (4.25).  
   - Compute \(\mathbf{g}_{01},\mathbf{g}_{02},\mathbf{g}_{12}\) and the tangent directions \(\mathbf{t}_{ij}\).  
   - Add \(\ell^{\text{triple-val}}\) and (optionally) \(\ell^{\text{triple-dir}}\) to \(L_{\text{triple}}\).

4) **Other losses (unchanged):**  
   - **Area balance** with softmax sampling (eqs. (4.30)–(4.40)) to keep regions comparable in size.  
   - **Smoothness** on the multi‑channel field (eqs. (4.41)–(4.46)) to discourage noise.  
   - (Optional) **Contour plane‑fitting** as in §4.3.1.3–4.3.1.4 for global straightening.  
   All three are already set up in your 3D chapter and integrate cleanly with this alignment term :contentReference[oaicite:10]{index=10} (Ch. 4 §4.3.2–§4.3.3, pp. 18–22; §4.3.1.3–4, pp. 18–19).

5) **Total loss and schedules.**  
   \[
   L_{\text{total}} = \lambda_{\text{grad}}\,L_{\text{grad-3D}}
   + \lambda_{\text{triple}}\,L_{\text{triple}}
   + \lambda_{\text{contour}}\,L_{\text{contour}}
   + \lambda_{\text{area}}\,L_{\text{area}}
   + \lambda_{\text{smooth}}\,L_{\text{smooth}}.
   \]
   Use the **temperature ramp** for \(\beta\) and coefficient ramps you already employ (Ch. 4 §4.4.3, eqs. (4.50)–(4.52)), and your optimizer/scheduler choices (AdamW + OneCycleLR) from §4.4.4–§4.4.5 to avoid early hard decisions and help convergence :contentReference[oaicite:11]{index=11} (pp. 23–24).

---

## 9) Practical tips & numerics

- **Vectorization:** For 6 channels, precompute all **15 pairwise** \(h_{ij}\) per face in a single batched tensor. Your “compact vectorized implementation” idea in §4.7.3 applies here too :contentReference[oaicite:12]{index=12} (p. 32).  
- **Stability:** Clamp \(\|E^\top E\|^{-1}\) solves with \(\varepsilon I\) (Tikhonov) and clip gradient norms (you already do; Ch. 4 §4.4.6) :contentReference[oaicite:13]{index=13} (p. 24).  
- **Curvature damping:** If a boundary legitimately runs along a sharp crease, reduce \(\lambda_{\text{grad}}\) locally via \(w^{\text{curv}}\) so the model can accept the kink.  
- **Diagnostics:** Visualize (i) edge crossing probabilities \(w^{\text{edge}}\), (ii) triple weights \(w^{\text{triple}}\), (iii) transported gradient angles across edges. This will immediately show whether misalignments come from gating, curvature, or noisy gradients.  
- **Synergy with your results:** On the sphere tests where you achieved clean 6 patches (Ch. 4 §4.6), this 3D gradient‑alignment term often reduces tiny wiggles left by plane‑fitting; on complex meshes where contour loss “refuses to converge,” the curvature‑aware transport prevents the loss from fighting the geometry, improving stability :contentReference[oaicite:14]{index=14} (pp. 30–31).

---

## 10) Why this works (intuitively)

- The **pairwise difference** \(h_{ij}\) is exactly the scalar whose zero line is the \(i\)–\(j\) boundary; aligning \(\nabla h_{ij}\) **direction** across adjacent faces straightens that line *geodesically* on the surface (in 2D you did the same; Fig. 3.1 demonstrates the straightening effect) :contentReference[oaicite:15]{index=15} (p. 10).  
- **Soft gates** ensure we only align where a boundary likely exists—noisy non‑boundary triangles don’t pollute the signal (edge gate eq. (4.6); face gate from smooth OR) :contentReference[oaicite:16]{index=16} (pp. 16–17).  
- **Triple‑point terms** force three boundaries to truly meet, instead of “nearly meet” with small gaps, and keep branch directions well‑behaved, using exactly the barycentric solve and soft weights you already formulated (eqs. (4.13)–(4.25)) :contentReference[oaicite:17]{index=17} (pp. 17–18).  
- **Normal‑aware comparison** (dihedral rotation) respects curvature, so “straight” means straight **on the surface**, not in the ambient \(\mathbb{R}^3\).

---

### Minimal set of equations to implement

- Per‑face gradient for any scalar \(q\):  
  \(E=[\mathbf{x}_2-\mathbf{x}_1,\ \mathbf{x}_3-\mathbf{x}_1]\), solve \((E^\top E)\mathbf{a}=\begin{bmatrix} q_2-q_1\\ q_3-q_1\end{bmatrix}\); \(\nabla q = E\mathbf{a}\).
- Edge soft gate for pair \(i,j\): \(w^{\text{edge}}_{ij}(a,b)=\sigma(-\beta\,d_a d_b)\).
- Face soft gate: \(w^{\text{face}}_{ij}=1-\prod_{k}(1-w^{\text{edge}}_{ij}(e_k))\).
- Cross‑face gradient comparison (with transport): \(\ell_{ij}(t,s)=w^{\text{edge}}_{ij}(e)\,\|\hat{\mathbf{g}}_{ij}(t\!\to\! s)-\hat{\mathbf{g}}_{ij}(s)\|^2\).
- Triple point barycentric solve (eqs. (4.13)–(4.23)) and weight (eq. (4.25)), plus \(\ell^{\text{triple-val}}\) and optional \(\ell^{\text{triple-dir}}\).
- Combine with your **Area** (eq. (4.40)) and **Smoothness** (eqs. (4.45)/(4.46)) losses and your scheduling from §4.4.3–§4.4.5. :contentReference[oaicite:18]{index=18}

---

If you want, I can draft a batched PyTorch snippet that computes \(L_{\text{grad-3D}}\) (with dihedral‑based transport) and \(L_{\text{triple}}\) using your existing tensors and indexing.
\end{verbatim}

\end{document}
