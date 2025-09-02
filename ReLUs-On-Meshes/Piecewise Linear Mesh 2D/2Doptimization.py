import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
from collections import defaultdict

###################################################
# 1) Generate Random 2D Mesh (NumPy)
###################################################
def generate_random_mesh(num_points=50, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.random((num_points, 2))
    tri_data = Delaunay(pts)
    return pts, tri_data.simplices

points_np, triangles_np = generate_random_mesh(num_points=60, seed=42)
num_vertices = points_np.shape[0]
num_triangles = triangles_np.shape[0]

# Compute total mesh area in NumPy (for normalization)
def triangle_area_np(p1, p2, p3):
    return 0.5 * abs(
        p1[0]*(p2[1] - p3[1]) +
        p2[0]*(p3[1] - p1[1]) +
        p3[0]*(p1[1] - p2[1])
    )

mesh_area = 0.0
for tri in triangles_np:
    p1, p2, p3 = points_np[tri[0]], points_np[tri[1]], points_np[tri[2]]
    mesh_area += triangle_area_np(p1, p2, p3)

print("Total mesh area =", mesh_area)

###################################################
# 2) Build Triangle Adjacency & Vertex Edges
###################################################
def build_triangle_adjacency(triangles_np):
    """
    Returns an array of shape (E,2) indicating pairs of triangles
    that share an edge, i.e. adjacency relationships.
    """
    edge_to_tris = defaultdict(list)
    for t_idx, tri in enumerate(triangles_np):
        i1, i2, i3 = sorted(tri)
        edges = [(i1,i2), (i1,i3), (i2,i3)]
        for e in edges:
            edge_to_tris[e].append(t_idx)
    adjacency = []
    for e, tlist in edge_to_tris.items():
        if len(tlist) == 2:
            adjacency.append(tuple(sorted(tlist)))
    return np.array(adjacency, dtype=np.int64)

def build_vertex_edges(triangles_np):
    """
    Returns an array of shape (E,2) for each distinct edge i-j
    among all triangles.
    """
    edge_set = set()
    for tri in triangles_np:
        i1, i2, i3 = sorted(tri)
        edges = [(i1,i2), (i1,i3), (i2,i3)]
        for e in edges:
            edge_set.add(e)
    return np.array(list(edge_set), dtype=np.int64)

triangle_adjacency_np = build_triangle_adjacency(triangles_np)
vertex_edges_np       = build_vertex_edges(triangles_np)

# Convert to Torch (CPU)
points      = torch.from_numpy(points_np).float()
triangles   = torch.from_numpy(triangles_np).long()
tri_adjac   = torch.from_numpy(triangle_adjacency_np)
vert_edges  = torch.from_numpy(vertex_edges_np)

##################################################################
# 3) Pick A,B,C so that C is between A and B (leftmost/rightmost)
##################################################################
A = np.argmin(points_np[:,0])  # leftmost x
B = np.argmax(points_np[:,0])  # rightmost x
pA = points_np[A]
pB = points_np[B]
midAB = 0.5*(pA + pB)

# among all vertices, pick the one closest to midAB => C
dists = np.sum((points_np - midAB)**2, axis=1)
C = np.argmin(dists)

print(f"Selected A={A} (x={pA[0]:.3f}), B={B} (x={pB[0]:.3f}), "
      f"C={C} (closest to midpoint)")

###################################################
# 4) Create Learnable Scalar Field (f)
#    Pin A=1, B=-1, C=0
###################################################
f_param = nn.Parameter(torch.zeros(num_vertices, dtype=torch.float))

with torch.no_grad():
    f_param.normal_(mean=0.0, std=0.01)
    f_param[A] = 1.0
    f_param[B] = -1.0
    f_param[C] = 0.0

##############################
# 5) Define the 3 Losses
##############################
quad_bary = torch.tensor([[0.5, 0.5, 0.0],
                          [0.5, 0.0, 0.5],
                          [0.0, 0.5, 0.5]], dtype=torch.float)
quad_w    = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float)

def sigmoid_beta(x, beta=20.0):
    return 1.0 / (1.0 + torch.exp(-beta*x))

def triangle_area_torch(p1, p2, p3):
    return 0.5 * torch.abs(
        p1[0]*(p2[1]-p3[1]) +
        p2[0]*(p3[1]-p1[1]) +
        p3[0]*(p1[1]-p2[1])
    )

def area_balance_loss(points, triangles, f, beta, mesh_area):
    """
    We integrate sigmoid_beta(f) over the entire mesh, then
    want it to be ~ 0.5 of total area.
    """
    integral_val = 0.0
    T = triangles.shape[0]
    for t_idx in range(T):
        i1, i2, i3 = triangles[t_idx]
        p1, p2, p3 = points[i1], points[i2], points[i3]
        area_t = triangle_area_torch(p1, p2, p3)
        tri_val = 0.0
        for q_idx in range(3):
            b = quad_bary[q_idx]
            w = quad_w[q_idx]
            f_q = f[i1]*b[0] + f[i2]*b[1] + f[i3]*b[2]
            tri_val += w * sigmoid_beta(f_q, beta=beta)
        integral_val += area_t * tri_val
    frac = integral_val / mesh_area
    return (frac - 0.5)**2

def gradient_in_triangle(points, tri, f):
    i1, i2, i3 = tri
    p1, p2, p3 = points[i1], points[i2], points[i3]
    f1, f2, f3 = f[i1], f[i2], f[i3]
    M = torch.stack([p2 - p1, p3 - p1], dim=1)  # shape (2,2)
    b = torch.stack([f2 - f1, f3 - f1])        # shape (2,)
    grad_f = torch.linalg.solve(M.T, b)       # shape (2,)
    return grad_f

def crosses_zero(tri, f):
    vals = torch.stack([f[tri[0]], f[tri[1]], f[tri[2]]])
    return (torch.min(vals) < 0.0) and (torch.max(vals) > 0.0)

def gradient_alignment_loss(points, triangles, f, adjacency):
    """
    For each pair of adjacent triangles that both cross zero,
    we want their gradient directions to be aligned.
    """
    eps = 1e-10
    T = triangles.shape[0]
    grad_cache = []
    cross_mask = []
    for t_idx in range(T):
        g = gradient_in_triangle(points, triangles[t_idx], f)
        grad_cache.append(g)
        cross_mask.append(crosses_zero(triangles[t_idx], f))
    grad_cache = torch.stack(grad_cache, dim=0)  # (T,2)
    cross_mask = torch.tensor(cross_mask, dtype=torch.bool)
    
    loss = 0.0
    E = adjacency.shape[0]
    for e_idx in range(E):
        t1, t2 = adjacency[e_idx]
        if cross_mask[t1] and cross_mask[t2]:
            g1 = grad_cache[t1]
            g2 = grad_cache[t2]
            n1 = torch.norm(g1)
            n2 = torch.norm(g2)
            if n1>eps and n2>eps:
                hat1 = g1/(n1+eps)
                hat2 = g2/(n2+eps)
                diff = hat1 - hat2
                loss += torch.dot(diff, diff)
    return loss

def smoothness_loss(f, vertex_edges):
    """
    Minimizes (fi - fj)^2 over edges.
    """
    i_j = vertex_edges
    fi = f[i_j[:,0]]
    fj = f[i_j[:,1]]
    diff = fi - fj
    return torch.sum(diff**2)

def three_part_loss(f, points, triangles, tri_adjac,
                    vert_edges, mesh_area,
                    beta=20.0, lambda_grad=1, lambda_smooth=1):
    L_area = area_balance_loss(points, triangles, f, beta, mesh_area)
    L_grad = gradient_alignment_loss(points, triangles, f, tri_adjac)
    L_smooth = smoothness_loss(f, vert_edges)
    total = L_area + lambda_grad*L_grad + lambda_smooth*L_smooth
    return L_area, L_grad, L_smooth, total

##############################################
# 6) Optimize with Adam for 10,000 iters
#    Print partial losses + total
##############################################
optimizer = optim.Adam([f_param], lr=1e-3)
max_iters = 10000

for it in range(max_iters+1):
    optimizer.zero_grad()

    L_area_val, L_grad_val, L_smooth_val, total_val = three_part_loss(
        f_param, points, triangles, tri_adjac, vert_edges,
        mesh_area, beta=3.0,
        lambda_grad=1.0, lambda_smooth=1
    )
    total_val.backward()
    optimizer.step()

    # Re-pin A,B,C
    with torch.no_grad():
        f_param[A] = 1.0
        f_param[B] = -1.0
        f_param[C] = 0.0

    # Print progress
    if it % 500 == 0 or it == max_iters:
        print(f"Iter {it}, "
              f"L_area={L_area_val.item():.6f}, "
              f"L_grad={L_grad_val.item():.6f}, "
              f"L_smooth={L_smooth_val.item():.6f}, "
              f"total={total_val.item():.6f}")

#########################################
# 7) Plot final result in 2 subplots
#    7a) Left:  original f
#    7b) Right: truly sigmoid(interpolate(f))
#        via triangle subdivision
#########################################
f_final = f_param.detach().numpy()  # shape (N,)

###################################################
# Subdivision routine to sample inside each triangle
# and compute sigmoid(interpolation of f).
###################################################
def subdivide_mesh(points, triangles, f_values, resolution=10, beta=400.0):
    """
    Subdivide each triangle into a finer mesh of sub-triangles.
    For each sub-vertex, do:
        f_sub = barycentric_interpolation(f)
        c_sub = sigmoid(f_sub)
    We then return the arrays:
        sub_points   [(#sub_verts, 2)]
        sub_tris     [(#sub_tris, 3)]
        sub_colors   [(#sub_verts,)]
    so that we can call:
        tripcolor(sub_points[:,0], sub_points[:,1], sub_tris, sub_colors)
    for a correct non-linear shading.
    """
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-beta*x))
    
    new_points = []
    new_colors = []
    new_tris   = []
    
    pts_np = np.ascontiguousarray(points)
    f_np   = np.ascontiguousarray(f_values)
    
    # global index for sub-vertices
    for tri_id in range(triangles.shape[0]):
        i1, i2, i3 = triangles[tri_id]
        
        p1, p2, p3 = pts_np[i1], pts_np[i2], pts_np[i3]
        f1, f2, f3 = f_np[i1], f_np[i2], f_np[i3]
        
        # We'll store sub-vertex indices in a 2D grid:
        sub_idx = np.full((resolution+1, resolution+1), -1, dtype=int)
        
        # Barycentric sampling
        for i in range(resolution+1):
            for j in range(resolution+1 - i):  # i + j <= resolution
                k = resolution - i - j
                b1 = i/float(resolution)
                b2 = j/float(resolution)
                b3 = k/float(resolution)
                
                x_sub = b1*p1[0] + b2*p2[0] + b3*p3[0]
                y_sub = b1*p1[1] + b2*p2[1] + b3*p3[1]
                
                f_sub = b1*f1 + b2*f2 + b3*f3
                c_sub = sigmoid(f_sub)
                
                # Add to global lists
                new_points.append([x_sub, y_sub])
                new_colors.append(c_sub)
                
                # index of this new sub-vertex
                idx = len(new_points) - 1
                sub_idx[i,j] = idx
        
        # Now form sub-triangles in this barycentric grid
        for i in range(resolution):
            for j in range(resolution - i):
                v0 = sub_idx[i, j]
                v1 = sub_idx[i+1, j]
                v2 = sub_idx[i, j+1]
                v3 = sub_idx[i+1, j+1]
                
                # triangle #1: (v0, v1, v2)
                new_tris.append([v0, v1, v2])
                # triangle #2: (v1, v3, v2) if j+1 <= resolution-(i+1)
                # ensures we don't go out of range
                if (j+1) <= (resolution - (i+1)):
                    new_tris.append([v1, v3, v2])
    
    # Convert to arrays
    new_points = np.array(new_points, dtype=np.float32)
    new_colors = np.array(new_colors, dtype=np.float32)
    new_tris   = np.array(new_tris,   dtype=np.int64)
    
    return new_points, new_tris, new_colors

# Subdivide for the right plot
res_for_plot = 15  # higher => smoother but more vertices
points_sub, tris_sub, color_sub = subdivide_mesh(
    points_np, triangles_np, f_final,
    resolution=res_for_plot,
    beta=400.0  # large beta => sharper transition
)

# Set up plots
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14,6))

###################################################
# Left Subplot: Plot Original f
###################################################
tc_left = ax_left.tripcolor(points_np[:,0], points_np[:,1],
                            triangles_np, f_final,
                            shading="gouraud", cmap="coolwarm")
fig.colorbar(tc_left, ax=ax_left, fraction=0.046, pad=0.04, label="f-value")

# Zero contour of f
ax_left.tricontour(points_np[:,0], points_np[:,1],
                   triangles_np, f_final,
                   levels=[0.0], colors='k', linewidths=2.0)

# Draw mesh edges
ax_left.triplot(points_np[:,0], points_np[:,1], triangles_np,
                color='black', linewidth=0.5)

# Mark pinned vertices
ax_left.plot(points_np[A,0], points_np[A,1], 'ks', markersize=8,
             label=f"A={f_final[A]:.2f} (pinned=1)")
ax_left.plot(points_np[B,0], points_np[B,1], 'ko', markersize=8,
             label=f"B={f_final[B]:.2f} (pinned=-1)")
ax_left.plot(points_np[C,0], points_np[C,1], 'k^', markersize=8,
             label=f"C={f_final[C]:.2f} (pinned=0)")

ax_left.legend()
ax_left.set_title("Optimized f(x)")

###################################################
# Right Subplot: Plot Sigmoid( Interpolated f )
# (we use the subdivided mesh)
###################################################
tc_right = ax_right.tripcolor(points_sub[:,0], points_sub[:,1],
                              tris_sub, color_sub,
                              shading="gouraud", cmap="coolwarm")
fig.colorbar(tc_right, ax=ax_right, fraction=0.046, pad=0.04, label="sigmoid(f)")

# Optional: isocontour at 0.5 in the subdivided space
ax_right.tricontour(points_sub[:,0], points_sub[:,1],
                    tris_sub, color_sub,
                    levels=[0.5], colors='k', linewidths=2.0)

# Draw original mesh edges on top (just for reference)
ax_right.triplot(points_np[:,0], points_np[:,1], triangles_np,
                 color='black', linewidth=0.5, alpha=0.5)

# Mark pinned vertices (same physical points)
ax_right.plot(points_np[A,0], points_np[A,1], 'ks', markersize=8,
              label=f"A => sigmoid({f_final[A]:.2f})")
ax_right.plot(points_np[B,0], points_np[B,1], 'ko', markersize=8,
              label=f"B => sigmoid({f_final[B]:.2f})")
ax_right.plot(points_np[C,0], points_np[C,1], 'k^', markersize=8,
              label=f"C => sigmoid({f_final[C]:.2f})")
ax_right.legend()
ax_right.set_title("Sigmoid of Interpolated f (subdivided)")

plt.tight_layout()
plt.show()
