from sklearn.decomposition import PCA
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry import Polygon
import numpy as np
from pathlib import Path
import ast
import json
import cv2




def fn_labels_in_square(estimated_3d_pos, mean_value, output_folder, save_image = False):

    '''
    
    
    
    '''

    # This flattens all world coordinates (assuming they are already np.arrays with shape (4,))
    #keep 3d coordinates only, without labels
    all_points_3d = np.array([entry[1][:3] for entry in estimated_3d_pos])  # shape (N, 3)
  
    #********************* PCA *********************************

    #perform PCA to get all the 3d points on X and Y axis:  #Project the 3D world points to a 2D plane using PCA
    pca = PCA(n_components=2, svd_solver='full', random_state=42)
    projected_2d = pca.fit_transform(all_points_3d)
    projected_2d_points = projected_2d[:, :2]

    #Find the smallest convex polygon that contains all 2D points, and calculate area
    hull_polygon = MultiPoint(projected_2d_points).convex_hull
    x, y = projected_2d_points[:, 0], projected_2d_points[:, 1]
    #width = x.max() - x.min()
    #height = y.max() - y.min()
    #print("Approx. area (bounding box):", width * height, "m²")

    # --- PCA axes scale correction --- a check to see if PCA transforms distances
    # Get scaling of each PCA axis (in meters per PCA unit)
    scale_x = np.linalg.norm(pca.components_[0])
    scale_y = np.linalg.norm(pca.components_[1])
    area_scale_factor = scale_x * scale_y
    
    print("scale factor area")
    print(area_scale_factor)

    # Get the total area of the convex hull (enclosing region)
    total_area_m2 = hull_polygon.area * area_scale_factor
    print(f"Total area covered by all bounding boxes: {total_area_m2:.4f} m²")

    #3d points back transformed for visualization
    reconstructed_3d = pca.inverse_transform(np.array(hull_polygon.exterior.coords))
    reconstructed_3d[:, 2] = mean_value


    #******************* SQUARE ********************************

    #project a square on polygon of size 40 x 40 cm, and 10 cm distance from left edge
    y_coords = projected_2d_points[:, 1]

    # Flip max/min to ensure left side if PCA axis is flipped
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)
    center_y = (min_y + max_y) / 2

    # Define square position (30×40 cm) at 10 cm from what we treat as "left"
    square_width = 0.4   # 40 cm
    square_height = 0.4  # 40 cm 
    square_x = x.min() + 0.1  # 10 cm from actual left
    # Start 10 cm in from the "left" (formerly max_x)
    square_bottom_y = center_y - square_height / 2

    # Define square in 2D PCA space
    square = np.array([
        [square_x,              square_bottom_y],
        [square_x + square_width, square_bottom_y],
        [square_x + square_width, square_bottom_y + square_height],
        [square_x,              square_bottom_y + square_height]
    ])

    # Find 2d points projected on PC axis that lie within the red square
    # Convert square to a Shapely polygon
    square_polygon = Polygon(square)

    # Check which PCA-projected points are inside the square
    contained_indices = [
        idx for idx, pt in enumerate(projected_2d_points)
        if square_polygon.contains(Point(pt))
    ]

    # Use those indices to get the original 3D points
    contained_3d_points = all_points_3d[contained_indices]
    contained_2d_points = projected_2d_points[contained_indices]

    labels_in_square = []
    for label_arr, coords in estimated_3d_pos:
        xyz = coords[:3]
        for contained in contained_3d_points:
            if np.allclose(xyz, contained, atol=1e-5):
                labels_in_square.append(label_arr[0])
                break


    # Project square back to 3D for visualization
    # Get the origin and PCA axes in world space
    origin = pca.mean_
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]

    # Build square manually in 3D (lying on PCA plane, with controlled Z)
    square_3d = []
    for pt2d in square:
        point3d = origin + pt2d[0] * pc1 + pt2d[1] * pc2
        point3d[2] = mean_value  # Force Z to be same as detection depth
        square_3d.append(point3d)
    square_3d = np.array(square_3d)

    #******************************************************

    # ------------------------------------------------
    # 3D Scatter Plot of All 3D Points, PCA and Square
    # ------------------------------------------------

    components = pca.components_  # Shape (2, 3)
    mean_3d = pca.mean_           # Shape (3,)
    scale = 0.8

    if save_image == True: 
        for elev in [25]: 
            for azim in [50]: 

                fig = plt.figure(figsize=(10, 5))

                # 3D view
                ax3d = fig.add_subplot(1, 2, 1, projection='3d')
                ax3d.scatter(all_points_3d[:, 0], all_points_3d[:, 1], all_points_3d[:, 2], c='blue', alpha=0.6, label='3D Points')
                ax3d.scatter(contained_3d_points[:, 0], contained_3d_points[:, 1], contained_3d_points[:, 2], color='red', s= 50, alpha = 1.0, label='Inside Square') 

                
                for i in range(2):
                    direction = components[i] * scale
                    ax3d.quiver(mean_3d[0], mean_3d[1], mean_3d[2],  # origin
                                direction[0], direction[1], direction[2],  # vector
                                color='red' if i == 0 else 'green',
                                linewidth=2,
                                label=f"PC{i+1}")

                ax3d.set_title("3D Bounding Box Points")
                ax3d.set_xlabel("X")
                ax3d.set_ylabel("Y")
                ax3d.set_zlabel("Z")
                ax3d.view_init(elev=elev, azim=azim)

                # 2D PCA Convex Hull Visualization
                ax2d = fig.add_subplot(1, 2, 2)
                x, y = projected_2d_points[:, 0], projected_2d_points[:, 1]
                hull_x, hull_y = hull_polygon.exterior.xy

                ax2d.scatter(x, y, alpha=0.6, label='PCA Projected Points')
                ax2d.scatter(contained_2d_points[:, 0], contained_2d_points[:, 1], color='skyblue', label='Inside Square') 
                ax2d.plot(hull_x, hull_y, color='green', linewidth=2, label='Convex Hull')

                #  Plot the red square here
                square_closed = np.vstack([square, square[0]])  # Ensure it's a closed loop
                ax2d.plot(square_closed[:, 0], square_closed[:, 1], color='red', linewidth=2, label="Reference Square")

                ax2d.set_title(f"PCA Ground Plane (Area = {total_area_m2:.2f} m²)")
                ax2d.set_xlabel("PCA 1")
                ax2d.set_ylabel("PCA 2")
                ax2d.axis("equal")
                ax2d.legend()

                plt.tight_layout()
                plt.savefig(output_folder / f"pca_vs_3d_debug{azim}_{elev}.png")
                plt.close()

    return labels_in_square, reconstructed_3d, square_3d



def plot_BB_cam7(r, reconstruced_3d, filtered_df, square_3d, output_directory, image_folder):

    path_calib_cam = output_directory / "poses_scaled.json"

    #*********************************************
    # Load intrinsic parameters from YAML file
    with open(path_calib_cam) as f:
        intrinsics_data = json.load(f)

    intrinsics = intrinsics_data["cam_07.png"]["intrinsics"]

    # Get focal length and principal point in real units
    fx = fy = intrinsics["focal_length"]
    cx, cy = intrinsics["principal_point"]

    # Now build intrinsic matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0, 1]
    ])

    #*********************************************
    # Load extrinsic parameters for cam_07.png
    with open(path_calib_cam) as f:
        extrinsics_data = json.load(f)

    extrinsics = extrinsics_data["cam_07.png"]["extrinsics"]
    R = np.array(extrinsics["rotation"])
    C = np.array(extrinsics["center"]).reshape(3, 1)

    def world_to_pixel(X, Y, Z, K, R, C):
                world_coords = np.array([[X], [Y], [Z]])
                X_cam = R @ (world_coords - C)  # Convert to camera coordinates
                x = K @ X_cam  # Project into pixel coordinates
                x /= x[2]      # Normalize homogeneous coordinates
                return int(round(float(x[0]))), int(round(float(x[1])))

    
    image = cv2.imread(str(image_folder / "cam_07.png"))
    overlay = image.copy()

    # --- Draw the convex hull reconstructed in 3D back on the image ---
    pts = []
    for x, y, z in reconstruced_3d:
        u, v = world_to_pixel(x, y, z, K, R, C)
        pts.append((u, v))

    # Draw the polygon (green transparent overlay)
    #cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (0, 255, 0))

    # Blend with original image
    alpha = 0.4
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    colors = [
      
     # Purple
    (0, 128, 128),   # Yellow
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Teal
    ]

    #Plot all the BB
    for _, row in r.iterrows():
        coords = row["cam_07.png"]
        num_view = row["num_observations"]


        # Only parse with ast.literal_eval if it's a string
        if isinstance(coords, str):
            coords = ast.literal_eval(coords)

        if isinstance(coords, (list, tuple)) and len(coords) == 4:
            x1, y1, x2, y2 = map(int, coords)
            color = colors[(num_view - 6) % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
            cv2.putText(image, f"{row['cluster_id']:.1f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    # Create an overlay for transparent fills
    overlay_fill = image.copy()

    for _, row in filtered_df.iterrows():
        coords = row["cam_07.png"]

        if isinstance(coords, str):
            coords = ast.literal_eval(coords)

        if isinstance(coords, (list, tuple)) and len(coords) == 4:
            x1, y1, x2, y2 = map(int, coords)

            # Draw filled grey rectangle on overlay
            cv2.rectangle(overlay_fill, (x1, y1), (x2, y2), (192, 192, 192), -1)

            # Draw border and text on original image (after blending)
            #cv2.rectangle(image, (x1, y1), (x2, y2), (192, 192, 192), 2)
            cv2.putText(image, f"{row['cluster_id']:.1f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Blend transparent fills back into the image
    alpha_fill = 0.6
    image = cv2.addWeighted(overlay_fill, alpha_fill, image, 1 - alpha_fill, 0)



    #Plot square
    square_px = [world_to_pixel(x, y, z, K, R, C) for x, y, z in square_3d]  
    #cv2.polylines(image, [np.array(square_px, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    legend_start_x = 50
    legend_start_y = 50
    box_height = 60
    box_width = 60
    spacing = 20
    font_scale = 1.2
    font_thickness = 3
    num_entries = len(colors)

    # Compute total legend height
    legend_height = num_entries * (box_height + spacing)

    # Create a copy of the image for blending
    overlay = image.copy()

    # Draw semi-transparent background rectangle
    bg_top_left = (legend_start_x - 15, legend_start_y - 15)
    bg_bottom_right = (legend_start_x + 350, legend_start_y + legend_height + 15)
    cv2.rectangle(overlay, bg_top_left, bg_bottom_right, (0, 0, 0), -1)  # black background

    # Blend with original image
    alpha = 0.7  # transparency factor
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    for i , color in enumerate(colors):
        top_left = (legend_start_x, legend_start_y + i * (box_height + spacing))
        bottom_right = (top_left[0] + box_width, top_left[1] + box_height)

        cv2.rectangle(image, top_left, bottom_right, color, -1)  # filled rectangle
        label = f"Views: {i+6}"
        cv2.putText(
            image,
            label,
            (bottom_right[0] + 5, bottom_right[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )


    cv2.imwrite(str(output_directory / "BB_image.png"), image)



def plot_BB_cam7_2colors(r, reconstruced_3d, filtered_df, square_3d, output_directory, image_folder):

    path_calib_cam = output_directory / "poses_scaled.json"

    #*********************************************
    # Load intrinsic parameters from YAML file
    with open(path_calib_cam) as f:
        intrinsics_data = json.load(f)

    intrinsics = intrinsics_data["cam_07.png"]["intrinsics"]

    # Get focal length and principal point in real units
    fx = fy = intrinsics["focal_length"]
    cx, cy = intrinsics["principal_point"]

    # Now build intrinsic matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0, 1]
    ])

    #*********************************************
    # Load extrinsic parameters for cam_07.png
    with open(path_calib_cam) as f:
        extrinsics_data = json.load(f)

    extrinsics = extrinsics_data["cam_07.png"]["extrinsics"]
    R = np.array(extrinsics["rotation"])
    C = np.array(extrinsics["center"]).reshape(3, 1)

    def world_to_pixel(X, Y, Z, K, R, C):
                world_coords = np.array([[X], [Y], [Z]])
                X_cam = R @ (world_coords - C)  # Convert to camera coordinates
                x = K @ X_cam  # Project into pixel coordinates
                x /= x[2]      # Normalize homogeneous coordinates
                return int(round(float(x[0]))), int(round(float(x[1])))

    image = cv2.imread(str(image_folder / "cam_07.png"))
    overlay = image.copy()

    # --- Draw the convex hull reconstructed in 3D back on the image ---
    pts = []
    for x, y, z in reconstruced_3d:
        u, v = world_to_pixel(x, y, z, K, R, C)
        pts.append((u, v))

    # Draw the polygon (green transparent overlay)
    #cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (0, 255, 0))

    # Blend with original image
    alpha = 0.4
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    #Plot all the BB
    for _, row in r.iterrows():
        coords = row["cam_07.png"]
        num_view = row["num_observations"]


        # Only parse with ast.literal_eval if it's a string
        if isinstance(coords, str):
            coords = ast.literal_eval(coords)

        if isinstance(coords, (list, tuple)) and len(coords) == 4:
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(image, f"{row['index']:.1f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
    #Plot the BB in square
    for _, row in filtered_df.iterrows():
        coords = row["cam_07.png"]
        
        # Only parse with ast.literal_eval if it's a string
        if isinstance(coords, str):
            coords = ast.literal_eval(coords)

        if isinstance(coords, (list, tuple)) and len(coords) == 4:
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(image, f"{row['index']:.1f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    #Plot square
    square_px = [world_to_pixel(x, y, z, K, R, C) for x, y, z in square_3d]  
    #cv2.polylines(image, [np.array(square_px, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)


    cv2.imwrite(str(output_directory / "BB_image_2colors.png"), image)