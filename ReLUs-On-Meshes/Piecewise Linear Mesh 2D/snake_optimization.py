import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

###############################################################################
# 1) Mesh-Loading and Basic Setup
###############################################################################
data = np.load("snake_mesh_filtered.npz")  # Adjust path as needed
points_np    = data["points"]   # shape (N,2)
triangles_np = data["faces"]    # shape (T,3)

print("points.shape =", points_np.shape, "  triangles.shape =", triangles_np.shape)

def triangle_area_np(p1, p2, p3):
    return 0.5 * abs(
        p1[0]*(p2[1] - p3[1]) +
        p2[0]*(p3[1] - p1[1]) +
        p3[0]*(p1[1] - p2[1])
    )

# Sum up areas of all triangles to get total mesh area (Python-side)
mesh_area = 0.0
for tri in triangles_np:
    p1, p2, p3 = points_np[tri[0]], points_np[tri[1]], points_np[tri[2]]
    mesh_area += triangle_area_np(p1, p2, p3)
print("Triangle-sum mesh_area =", mesh_area)

# Build edges for smoothness
def build_vertex_edges(triangles_np):
    edge_set = set()
    for tri in triangles_np:
        i1, i2, i3 = sorted(tri)
        edges = [(i1, i2), (i1, i3), (i2, i3)]
        for e in edges:
            edge_set.add(e)
    return np.array(list(edge_set), dtype=np.int64)

vertex_edges_np  = build_vertex_edges(triangles_np)

###############################################################################
# 2) Identify Distinct A,B, then pick a midpoint-based C
###############################################################################
def get_leftmost_vertex(points_arr):
    x_min = np.min(points_arr[:,0])
    candidates = np.where(np.isclose(points_arr[:,0], x_min))[0]
    return candidates[0]

def get_rightmost_vertex(points_arr):
    x_max = np.max(points_arr[:,0])
    candidates = np.where(np.isclose(points_arr[:,0], x_max))[0]
    return candidates[0]

A = get_leftmost_vertex(points_np)
B = get_rightmost_vertex(points_np)

# Make sure A != B
if A == B:
    raise ValueError("Leftmost and Rightmost indices ended up the same. "
                     "Mesh might be degenerate or all x-coords identical.")

pA = points_np[A]
pB = points_np[B]
print(f"[DEBUG] A index={A}, coords={pA}, B index={B}, coords={pB}")

# Simple midpoint-based C
pMid = 0.5 * (pA + pB)
d2 = np.sum((points_np - pMid)**2, axis=1)
sorted_idx = np.argsort(d2)
C = sorted_idx[0]
if C in [A,B]:
    for idx in sorted_idx:
        if idx not in [A,B]:
            C = idx
            break

print(f"[DEBUG] A={A}, B={B}, C={C}   (distinct)")

###############################################################################
# 3) Create Learnable Scalar Field, Pin f(A)=1, f(B)=-1, f(C)=0
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

points_torch     = torch.from_numpy(points_np).float().to(device)
triangles_torch  = torch.from_numpy(triangles_np).long().to(device)
vert_edges_torch = torch.from_numpy(vertex_edges_np).long().to(device)

f_param = nn.Parameter(torch.zeros(points_np.shape[0], dtype=torch.float, device=device))

with torch.no_grad():
    f_param.normal_(mean=0.0, std=0.01)
    f_param[A] = 1.0
    f_param[B] = -1.0
    f_param[C] = 0.0

pinned_indices = [A, B, C]
pinned_values  = [1.0, -1.0, 0.0]

def zero_grad_at_indices(param, idx_list):
    if param.grad is not None:
        with torch.no_grad():
            for idx in idx_list:
                param.grad[idx] = 0.0

###############################################################################
# 4) Define a 7-Point Dunavant Quadrature for Better Integration
###############################################################################
# The following barycentric coordinates + weights define a degree‐4 exact rule
# for integration on a reference triangle. Source: "Dunavant Quadrature".
dunavant7_bary = torch.tensor([
    [1/3,          1/3,          1/3         ],  # center
    [0.0597158717, 0.4701420641, 0.4701420641],
    [0.4701420641, 0.0597158717, 0.4701420641],
    [0.4701420641, 0.4701420641, 0.0597158717],
    [0.1012865073, 0.1012865073, 0.7974269853],
    [0.1012865073, 0.7974269853, 0.1012865073],
    [0.7974269853, 0.1012865073, 0.1012865073],
], dtype=torch.float)

dunavant7_w = torch.tensor([
    0.225,         # weight for the center
    0.1323941528,
    0.1323941528,
    0.1323941528,
    0.1259391805,
    0.1259391805,
    0.1259391805,
], dtype=torch.float)

###############################################################################
# 5) Define the Three Losses (Vectorized, but Using the 7-Point Rule)
###############################################################################
def area_balance_loss(points, triangles, f, beta, mesh_area_val):
    """
    Integrate sigmoid_beta(f) over the mesh, using the 7-point Dunavant rule
    for each triangle. Then compute (fraction - 0.5)^2. Returns a Python float.
    """
    tri_pts = points[triangles]  # shape (T,3,2)
    x1 = tri_pts[:,0,0]
    y1 = tri_pts[:,0,1]
    x2 = tri_pts[:,1,0]
    y2 = tri_pts[:,1,1]
    x3 = tri_pts[:,2,0]
    y3 = tri_pts[:,2,1]

    # triangle areas => shape (T,)
    area_t = 0.5 * torch.abs(
        x1*(y2 - y3) +
        x2*(y3 - y1) +
        x3*(y1 - y2)
    )

    # f-values at triangle vertices => shape (T,3)
    f_tri = f[triangles]

    # We'll evaluate f at each of the 7 Dunavant points
    #   shape (1,7,3) for barycentric
    B_expanded = dunavant7_bary.to(points.device).unsqueeze(0)
    #   shape (T,1,3) for f-values
    f_tri_exp = f_tri.unsqueeze(1)

    # shape (T,7): f at each quadrature point
    f_q = (f_tri_exp * B_expanded).sum(dim=2)

    # apply the logistic function
    s_q = 1.0 / (1.0 + torch.exp(-beta * f_q))  # shape (T,7)

    # weighting
    w7 = dunavant7_w.to(points.device)  # shape (7,)
    tri_integrals = (s_q * w7).sum(dim=1)  # (T,)

    # multiply by area and sum
    integral_val = (tri_integrals * area_t).sum()  # single scalar
    frac = integral_val.detach().cpu().item() / mesh_area_val
    return (frac - 0.5)**2  # Python float

def compute_triangle_gradients(points, triangles, f):
    """
    Vectorized version of gradient_in_triangle. For each triangle:
      M^T grad_f = b  => grad_f = solve(M^T, b).
    Returns shape (T,2).
    """
    tri_pts = points[triangles]  # (T,3,2)
    tri_vals= f[triangles]       # (T,3)

    p1 = tri_pts[:,0,:]
    p2 = tri_pts[:,1,:]
    p3 = tri_pts[:,2,:]
    f1 = tri_vals[:,0]
    f2 = tri_vals[:,1]
    f3 = tri_vals[:,2]

    M = torch.stack([p2 - p1, p3 - p1], dim=2)  # (T,2,2)
    b = torch.stack([f2 - f1, f3 - f1], dim=1)  # (T,2)

    M_t = M.permute(0,2,1)  # (T,2,2)
    b_  = b.unsqueeze(-1)   # (T,2,1)
    grad_f = torch.linalg.solve(M_t, b_)  # (T,2,1)
    return grad_f.squeeze(-1)            # (T,2)

def crosses_zero_tri(triangles, f):
    """
    Vectorized check if min(f) < 0 < max(f) in each triangle => crosses zero.
    Returns shape (T,) bool.
    """
    vals = f[triangles]  # (T,3)
    minv, _ = vals.min(dim=1)
    maxv, _ = vals.max(dim=1)
    return (minv <= 0.0) & (maxv >= 0.0)

def gradient_alignment_loss_no_adjacency(points, triangles, f):
    """
    Same O(T^2) logic, but vectorized on GPU. Compare normalized gradients
    for all pairs of zero-crossing triangles.
    """
    eps = 1e-10
    grad_cache = compute_triangle_gradients(points, triangles, f)  # (T,2)
    cross_mask = crosses_zero_tri(triangles, f)                    # (T,)
    norms = torch.norm(grad_cache, dim=1)
    keep_mask = cross_mask & (norms > eps)

    valid_g = grad_cache[keep_mask]
    valid_n = norms[keep_mask]
    M = valid_g.shape[0]
    if M < 2:
        return torch.tensor(0.0, dtype=f.dtype, device=f.device)

    hat = valid_g / (valid_n.unsqueeze(1) + eps)  # (M,2)

    # Pairwise difference => shape (M,M,2)
    diff_mat = hat.unsqueeze(1) - hat.unsqueeze(0)  # (M,M,2)
    sq_norm_mat = (diff_mat * diff_mat).sum(dim=2)  # (M,M)

    # sum only over i<j
    triu_mask = torch.triu(torch.ones(M, M, dtype=torch.bool, device=f.device), diagonal=1)
    loss_val = sq_norm_mat[triu_mask].sum()
    return loss_val

def smoothness_loss(f, vertex_edges):
    i_j = vertex_edges
    fi = f[i_j[:,0]]
    fj = f[i_j[:,1]]
    diff = fi - fj
    return torch.sum(diff**2)

def three_part_loss(f, points, triangles, vert_edges, mesh_area_val,
                    beta=3.0, lambda_grad=0.1, lambda_smooth=0.01):
    """
    Calls our new 7-point area_balance_loss for improved accuracy,
    plus the same gradient & smoothness terms as before.
    """
    L_area_float = area_balance_loss(points, triangles, f, beta, mesh_area_val)  # Python float
    L_grad       = gradient_alignment_loss_no_adjacency(points, triangles, f)    # torch scalar
    L_smooth     = smoothness_loss(f, vert_edges)                                # torch scalar

    L_area_torch = torch.tensor(L_area_float, device=f.device, dtype=f.dtype)
    total = L_area_torch + lambda_grad * L_grad + lambda_smooth * L_smooth
    return L_area_float, L_grad, L_smooth, total

###############################################################################
# 6) Optimization
###############################################################################
f_param.requires_grad = True
optimizer = optim.Adam([f_param], lr=1e-3)
max_iters = 15000

for it in range(max_iters+1):
    optimizer.zero_grad()
    L_area_val, L_grad_val, L_smooth_val, total_val = three_part_loss(
        f_param, points_torch, triangles_torch, vert_edges_torch,
        mesh_area, beta=3.0, lambda_grad=1.0, lambda_smooth=0.1
    )
    total_val.backward()

    # Zero out gradient so pinned values do NOT move
    zero_grad_at_indices(f_param, pinned_indices)

    # Update
    optimizer.step()

    # Re-pin to ensure exact values
    with torch.no_grad():
        for idx, val in zip(pinned_indices, pinned_values):
            f_param[idx] = val

    # Print debug
    if it % 500 == 0 or it == max_iters:
        la = L_area_val  # python float
        lg = L_grad_val.detach().cpu().item()
        ls = L_smooth_val.detach().cpu().item()
        tt = total_val.detach().cpu().item()
        print(f"[Iter {it}]  "
              f"L_area={la:.6f}, "
              f"L_grad={lg:.6f}, "
              f"L_smooth={ls:.6f}, "
              f"total={tt:.6f}  ||  "
              f"f[A]={f_param[A].item():.2f}, "
              f"f[B]={f_param[B].item():.2f}, "
              f"f[C]={f_param[C].item():.2f}")

###############################################################################
# 7) Plot
###############################################################################
f_final = f_param.detach().cpu().numpy()  # move back to CPU for plotting

def subdivide_mesh(points, triangles, f_values, resolution=10, beta=400.0):
    """
    Subdivide each triangle and compute c_sub = sigmoid( f_sub ).
    We pass those subdivided points to tripcolor for a smooth color plot.
    """
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-beta*x))
    
    new_points = []
    new_colors = []
    new_tris   = []
    
    pts_np = np.ascontiguousarray(points)
    f_np   = np.ascontiguousarray(f_values)
    
    for tri_id in range(triangles.shape[0]):
        i1, i2, i3 = triangles[tri_id]
        
        p1, p2, p3 = pts_np[i1], pts_np[i2], pts_np[i3]
        f1, f2, f3 = f_np[i1], f_np[i2], f_np[i3]
        
        sub_idx = np.full((resolution+1, resolution+1), -1, dtype=int)
        
        for i in range(resolution+1):
            for j in range(resolution+1 - i):
                k = resolution - i - j
                b1 = i/float(resolution)
                b2 = j/float(resolution)
                b3 = k/float(resolution)
                
                x_sub = b1*p1[0] + b2*p2[0] + b3*p3[0]
                y_sub = b1*p1[1] + b2*p2[1] + b3*p3[1]
                
                f_sub = b1*f1 + b2*f2 + b3*f3
                c_sub = sigmoid(f_sub)
                
                new_points.append([x_sub, y_sub])
                new_colors.append(c_sub)
                
                idx = len(new_points) - 1
                sub_idx[i,j] = idx
        
        for i in range(resolution):
            for j in range(resolution - i):
                v0 = sub_idx[i,   j]
                v1 = sub_idx[i+1, j]
                v2 = sub_idx[i,   j+1]
                v3 = sub_idx[i+1, j+1]
                
                new_tris.append([v0, v1, v2])
                if (j+1) <= (resolution - (i+1)):
                    new_tris.append([v1, v3, v2])
    
    new_points = np.array(new_points, dtype=np.float32)
    new_colors = np.array(new_colors, dtype=np.float32)
    new_tris   = np.array(new_tris,   dtype=np.int64)
    
    return new_points, new_tris, new_colors

points_sub, tris_sub, color_sub = subdivide_mesh(points_np, triangles_np,
                                                 f_final, resolution=15, beta=400.0)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14,6))

# Left: direct f
tc_left = ax_left.tripcolor(points_np[:,0], points_np[:,1], triangles_np,
                            f_final, shading="gouraud", cmap="coolwarm")
fig.colorbar(tc_left, ax=ax_left, fraction=0.046, pad=0.04, label="f-value")

ax_left.tricontour(points_np[:,0], points_np[:,1], triangles_np,
                   f_final, levels=[0.0], colors='k', linewidths=2.0)
ax_left.triplot(points_np[:,0], points_np[:,1], triangles_np,
                color='black', linewidth=0.5)

ax_left.plot(points_np[A,0], points_np[A,1], 'rs', markersize=8,
             label=f"A={f_final[A]:.2f}")
ax_left.plot(points_np[B,0], points_np[B,1], 'go', markersize=8,
             label=f"B={f_final[B]:.2f}")
ax_left.plot(points_np[C,0], points_np[C,1], 'm^', markersize=8,
             label=f"C={f_final[C]:.2f}")
ax_left.axis("equal")
ax_left.legend()
ax_left.set_title("Optimized f(x) - Original Triangulation (7‐pt Dunavant)")

# Right: Sigmoid( f ) on subdivided
tc_right = ax_right.tripcolor(points_sub[:,0], points_sub[:,1],
                              tris_sub, color_sub, shading="gouraud", cmap="coolwarm")
fig.colorbar(tc_right, ax=ax_right, fraction=0.046, pad=0.04, label="sigmoid(f)")

ax_right.tricontour(points_sub[:,0], points_sub[:,1], tris_sub,
                    color_sub, levels=[0.5], colors='k', linewidths=2.0)
ax_right.triplot(points_np[:,0], points_np[:,1], triangles_np,
                 color='black', linewidth=0.5, alpha=0.5)

ax_right.plot(points_np[A,0], points_np[A,1], 'rs', markersize=8,
              label=f"A=>{f_final[A]:.2f}")
ax_right.plot(points_np[B,0], points_np[B,1], 'go', markersize=8,
              label=f"B=>{f_final[B]:.2f}")
ax_right.plot(points_np[C,0], points_np[C,1], 'm^', markersize=8,
              label=f"C=>{f_final[C]:.2f}")
ax_right.axis("equal")
ax_right.legend()
ax_right.set_title("Sigmoid(f) - Subdivided (7‐pt Dunavant)")

plt.tight_layout()
plt.show()
