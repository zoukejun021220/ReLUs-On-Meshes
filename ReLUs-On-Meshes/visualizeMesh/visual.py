import numpy as np
import pyvista as pv
import vtk
import numpy as np
import pyvista as pv

import numpy as np
import pyvista as pv
from typing import Optional
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


def subdivide_mesh_with_values(
    vertices: np.ndarray,    # shape (N, 3)
    faces: np.ndarray,       # shape (T, 3)
    f_values: np.ndarray,    # shape (N, C), 6-channel (or more) fields
    num_subdivisions: int = 1
):
    """
    Subdivide each triangle in the mesh 'num_subdivisions' times, returning:
      - new_vertices: shape (N', 3)
      - new_faces: shape (T', 3)
      - new_f_values: shape (N', C)
    where each newly created midpoint inherits an averaged field value.

    This allows us to visualize a finer mesh with sharper boundary edges
    once applied argmax to the interpolated field.
    """
    # Convert to lists for easier append
    current_vertices = vertices.tolist()  # each entry is [x,y,z]
    current_faces = faces.tolist()        # each entry is [i1, i2, i3]
    current_fvalues = f_values.tolist()   # each entry is [f_c0, f_c1, ..., f_cN]

    def subdivide_once(verts, facs, fvals):
        """
        Perform one round of subdivision and return updated (verts, facs, fvals).
        """
        new_verts = list(verts)      # will append as we go
        new_fvals = list(fvals)
        new_faces = []
        edge_to_mid = {}

        # Precompute edges
        for tri in facs:
            i1, i2, i3 = tri

            edges = [
                (min(i1, i2), max(i1, i2)),
                (min(i2, i3), max(i2, i3)),
                (min(i3, i1), max(i3, i1))
            ]

            # For each edge, if midpoint not yet in dictionary, create it
            for e in edges:
                if e not in edge_to_mid:
                    # midpoint in position
                    vA = np.array(verts[e[0]], dtype=np.float32)
                    vB = np.array(verts[e[1]], dtype=np.float32)
                    mid_pos = 0.5 * (vA + vB)
                    # normalize if want to keep it on a sphere of radius=1
                    # or skip normalization if it's an arbitrary mesh:
                    # mid_pos /= np.linalg.norm(mid_pos)

                    # mid f-value
                    fA = np.array(fvals[e[0]], dtype=np.float32)
                    fB = np.array(fvals[e[1]], dtype=np.float32)
                    mid_f = 0.5 * (fA + fB)

                    # index of new vertex
                    new_idx = len(new_verts)
                    new_verts.append(mid_pos.tolist())
                    new_fvals.append(mid_f.tolist())

                    edge_to_mid[e] = new_idx

        # Now create 4 new faces per old face
        for tri in facs:
            i1, i2, i3 = tri
            e1 = (min(i1, i2), max(i1, i2))
            e2 = (min(i2, i3), max(i2, i3))
            e3 = (min(i3, i1), max(i3, i1))

            a = edge_to_mid[e1]
            b = edge_to_mid[e2]
            c = edge_to_mid[e3]

            new_faces.append([i1, a, c])
            new_faces.append([i2, b, a])
            new_faces.append([i3, c, b])
            new_faces.append([a, b, c])

        return new_verts, new_faces, new_fvals

    # Repeatedly subdivide
    for _ in range(num_subdivisions):
        current_vertices, current_faces, current_fvalues = subdivide_once(
            current_vertices, current_faces, current_fvalues
        )

    # Convert back to numpy arrays
    new_vertices = np.array(current_vertices, dtype=np.float32)
    new_faces    = np.array(current_faces,    dtype=np.int32)
    new_f_values = np.array(current_fvalues,  dtype=np.float32)

    return new_vertices, new_faces, new_f_values

def visualize_segmentation(
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    f_values:   np.ndarray,
    pinned_indices: Optional[list] = None,
    region_names: Optional[list]   = None,
    subdivisions: int = 2
):
    """
    Visualize the segmentation by subdividing the mesh 'subdivisions' times,
    interpolating the multi-channel field values, and taking argmax on each
    subdivided vertex.

    Args:
      vertices_np: (N,3) original mesh vertices
      faces_np:    (T,3) original mesh faces
      f_values:    (N,C) multi-channel field (e.g., 6-channel)
      pinned_indices: optional list of pinned vertex indices in the original mesh,
                      so we can mark them in the visualization.
      region_names: optional list of labels for pinned vertices (e.g. ["Top","Bottom","Front","Back","Right","Left"])
      subdivisions: how many times to subdivide each triangle. 
                    0 => no subdivision, just argmax on the original mesh
    """
    # 1) Subdivide and interpolate the field
    if subdivisions > 0:
        sub_vertices, sub_faces, sub_fvals = subdivide_mesh_with_values(
            vertices_np, faces_np, f_values, num_subdivisions=subdivisions
        )
    else:
        # No subdivision => just use the original data
        sub_vertices = vertices_np
        sub_faces    = faces_np
        sub_fvals    = f_values

    # 2) Hardmax (argmax) for each subdivided vertex
    hard_labels = np.argmax(sub_fvals, axis=1)  # shape (N',)

    # 3) Try to render with PyVista
    try:
        import pyvista as pv
        from matplotlib.colors import ListedColormap

        # For 6-channel data, here are 6 sample RGBA colors
      
        region_colors = np.array([
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 0.0, 1.0, 1.0],  # Blue
            [0.0, 1.0, 0.0, 1.0],  # Green
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [1.0, 0.0, 1.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0, 1.0],  # Cyan
        ])
        

        region_cmap = ListedColormap(region_colors)

        # Build a PyVista PolyData from the subdivided mesh
        faces_flat = np.column_stack((np.full(len(sub_faces), 3), sub_faces)).flatten()
        vis_mesh   = pv.PolyData(sub_vertices, faces_flat)
        # Add the integer labels (1-based for nicer display)
        vis_mesh["Labels"] = hard_labels + 1

        # 4) Set up a PyVista plotter
        pv.set_plot_theme("document")
        plotter = pv.Plotter()
        plotter.add_text(
            f"Hardmax Segmentation (subdivisions={subdivisions})",
            font_size=14, position='upper_edge'
        )

        # 5) Add the mesh with labels
        plotter.add_mesh(
            vis_mesh,
            scalars="Labels",
            show_edges=False,
            cmap=region_cmap,
            interpolate_before_map=False,  # ensures crisp boundaries
            show_scalar_bar=True,
            clim=[1, region_colors.shape[0]],  # or use [1, #channels]
        )
        plotter.add_scalar_bar(
            title="Region Label",
            n_labels=region_colors.shape[0],
            fmt="%d",
            font_family="arial",
        )

        # 6) If pinned vertices are known, mark them
        if pinned_indices is not None:
            # Some default marker colors for pinned vertices
            pin_colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 0.0, 1.0],  # Blue
                [0.0, 1.0, 0.0],  # Green
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
            ]
            # Mark each pinned vertex from the ORIGINAL mesh
            # (It won't match a subdivided index unless it's a corner).
            # Just draw points in 3D space:
            for i, vidx in enumerate(pinned_indices):
                pin_pos = vertices_np[vidx]
                color   = pin_colors[i % len(pin_colors)]
                plotter.add_points(pin_pos.reshape(1,3), color=color, point_size=15)
                if region_names and i < len(region_names):
                    # Add a label
                    offset_pos = pin_pos * 1.02  # slightly offset
                    plotter.add_point_labels(
                        [offset_pos],
                        [region_names[i]],
                        font_size=10, 
                        text_color=color, 
                        shape=None
                    )

        # 7) Show and optionally screenshot
        plotter.view_isometric()
        plotter.show()

    except ImportError as e:
        print(f"PyVista not installed (Error: {e}). Falling back to Matplotlib 3D scatter.")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10,8))
        ax  = fig.add_subplot(111, projection='3d')
        
        # map each label to a color
   
        color_list = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        c_idx = np.take(color_list, hard_labels % len(color_list), axis=0)

        ax.scatter(
            sub_vertices[:,0],
            sub_vertices[:,1],
            sub_vertices[:,2],
            c=c_idx,
            s=5
        )
        ax.set_box_aspect((1,1,1))
        ax.set_title(f"Hardmax Visualization (subdiv={subdivisions}, Matplotlib fallback)")
        plt.show()


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
# Here's how to use the function:
if __name__ == "__main__":
    vertices_np, faces_np = load_volume_tet_mesh_and_extract_surface("Piecewise Linear Mesh 3D\l1-poly-dat\hex\kitty\orig.tet.vtk")
    
    data=np.load("visualizeMesh\\final_mesh_and_values1.npz")
    f_values=data["field_values"]
 
    
    visualize_segmentation(
        vertices_np=vertices_np,
        faces_np=faces_np,
        f_values=f_values,
      
        subdivisions=5,  # Adjust as needed
       
    )
    
    
    pass