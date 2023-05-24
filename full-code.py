import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import inv
import pandas as pd


def sort_points(top_right, top_left, bot_right, bot_left):
    # Create an array of points
    points = np.array([top_right, top_left, bot_right, bot_left])

    # Sort the points based on their y-coordinates (column 1)
    sorted_points = points[np.argsort(points[:, 1])]

    # Split the sorted points into top and bottom halves
    bot_half = sorted_points[:2]
    top_half = sorted_points[2:]

    # Sort the top and bottom halves based on their x-coordinates (column 0)
    top_sorted = top_half[np.argsort(top_half[:, 0])]
    bot_sorted = bot_half[np.argsort(bot_half[:, 0])]

    # Rearrange the points as top right, top left, bot right, bot left
    Corner1 = top_sorted[1]
    Corner2 = top_sorted[0]
    Corner3 = bot_sorted[1]
    Corner4 = bot_sorted[0]

    return Corner1, Corner2, Corner3, Corner4


def f_ransac(edge_array, iter, threshold):
    max_iterations = iter  # Maximum number of iterations
    threshold = threshold  # Distance threshold to consider a point an inlier
    best_inliers = []  # Best inlier indices
    best_inlier_count = 0  # Best inlier count
    edge_array = edge_array[:, :2]

    # RANSAC iterations
    for i in range(max_iterations):
        it = i
        # Randomly sample two points from the PointCloud
        random_indices = np.random.choice(len(edge_array), size=2, replace=False)
        random_points = edge_array[random_indices]

        # Fit a line to the sampled points (implicit function)
        x1, y1 = random_points[0]
        x2, y2 = random_points[1]
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        # Calculate the distances between all points and the line
        distances = np.abs(a * edge_array[:, 0] + b * edge_array[:, 1] + c) / np.sqrt(a ** 2 + b ** 2)

        # Count the inliers based on the distance threshold
        inlier_indices = np.where(distances < threshold)[0]
        inlier_count = len(inlier_indices)

        # Check if the current model is the best so far
        if inlier_count > best_inlier_count:
            best_model = [a, b, c]
            best_inliers = inlier_indices
            best_inlier_count = inlier_count

        # Exit the loop if a satisfactory number of inliers is found
        if best_inlier_count > len(edge_array) * 0.8:
            break

    # Extract line parameters from the implicit function
    a, b, c = best_model
    slope = -a / b
    intercept = -c / b
    line_parameters = [slope, intercept]

    return best_inliers, line_parameters


def KNN(Plane_Points, neighbours, threshold):
    tree = cKDTree(Plane_Points)
    distances, _ = tree.query(Plane_Points, k=neighbours + 1)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    std_distances = np.std(distances[:, 1:], axis=1)
    outliers = np.where(mean_distances > threshold)[0]
    filtered_data = Plane_Points[outliers]
    return filtered_data

def circle_func(a, b, r, x):
    return (np.sqrt(r ** 2 - (x - a) ** 2) + b, -np.sqrt(r ** 2 - (x - a) ** 2) + b)

class RANSAC:
    def __init__(self, x_data, y_data, n):
        self.x_data = x_data
        self.y_data = y_data
        self.n = n
        self.d_min = 99999
        self.best_model = None

    def random_sampling(self):
        sample = []
        save_ran = []
        count = 0

        # get three points from data
        while True:
            ran = np.random.randint(len(self.x_data))

            if ran not in save_ran:
                sample.append((self.x_data[ran], self.y_data[ran]))
                save_ran.append(ran)
                count += 1

                if count == 3:
                    break

        return sample

    def make_model(self, sample):
        # calculate A, B, C value from three points by using matrix

        pt1 = sample[0]
        pt2 = sample[1]
        pt3 = sample[2]

        A = np.array([[pt2[0] - pt1[0], pt2[1] - pt1[1]], [pt3[0] - pt2[0], pt3[1] - pt2[1]]])
        B = np.array([[pt2[0] ** 2 - pt1[0] ** 2 + pt2[1] ** 2 - pt1[1] ** 2],
                      [pt3[0] ** 2 - pt2[0] ** 2 + pt3[1] ** 2 - pt2[1] ** 2]])
        inv_A = inv(A)

        c_x, c_y = np.dot(inv_A, B) / 2
        c_x, c_y = c_x[0], c_y[0]
        r = np.sqrt((c_x - pt1[0]) ** 2 + (c_y - pt1[1]) ** 2)
        return c_x, c_y, r

    def eval_model(self, model):
        d = 0
        c_x, c_y, r = model

        for i in range(len(self.x_data)):
            dis = np.sqrt((self.x_data[i] - c_x) ** 2 + (self.y_data[i] - c_y) ** 2)

            if dis >= r:
                d += dis - r
            else:
                d += r - dis

        return d

    def execute_ransac(self):
        # find best model
        for i in range(self.n):
            model = self.make_model(self.random_sampling())
            d_temp = self.eval_model(model)

            if self.d_min > d_temp:
                self.best_model = model
                self.d_min = d_temp



pcd = o3d.io.read_point_cloud(r"PCs\Scans\ScanData26.ply")
name = 'Data26.xlsx'
pcd_array = np.asarray(pcd.points)


print("Identifying workpiece")
DBSCAN_sub= o3d.geometry.PointCloud()
for x in range(len(pcd_array[:,2])):
    if pcd_array[x,2]>200 and pcd_array[x,2]<252:
        DBSCAN_sub.points.append(pcd_array[x])

max_label = 0

while max_label != 4:
    print("RANSAC plane creation")
    plane_model, inliers = DBSCAN_sub.segment_plane(distance_threshold=0.3,
                                                    ransac_n=3,
                                                    num_iterations=1000)
    Plane_pcd = DBSCAN_sub.select_by_index(inliers)

    print("RANSAC plane alignment and projection")
    a, b, c, d = plane_model[:4]
    plane_points = np.asarray(Plane_pcd.points)

    # Normalize the plane normal
    plane_normal = np.array([a, b, c])
    plane_normal /= np.linalg.norm(plane_normal)

    # Compute the rotation angle and axis to align the plane normal with the z-axis
    rotation_axis = np.cross(plane_normal, np.array([0, 0, 1]))
    rotation_angle = -np.arccos(np.dot(plane_normal, np.array([0, 0, 1])))

    # Create the rotation matrix using the rotation axis and angle
    rotation_matrix = Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()

    # Rotate the plane points using the same rotation matrix
    rotated_points = np.dot(plane_points, rotation_matrix.T)

    # Extract only the x and y values from the rotated points
    projected_points = rotated_points[:, :2]

    # Append a column of zeros to the projected points
    #projected_points = np.hstack((projected_points, zeros_column))

    print("K-NN Edge Segmentation")
    edges_full_array = KNN(projected_points, neighbours=10, threshold=0.14)
    edges_full = o3d.geometry.PointCloud()
    zeros_column = np.zeros((edges_full_array.shape[0], 1))
    edges_full_array = np.hstack((edges_full_array, zeros_column))
    edges_full.points = o3d.utility.Vector3dVector(edges_full_array)

    print("Edge Clustering")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            edges_full.cluster_dbscan(eps=5.5, min_points=50, print_progress=True))
    max_label = labels.max()
    print(f"Edge PC has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    edges_full.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #o3d.visualization.draw_geometries([edges_full])

print("Find cluster with most points")
unique_clusters, cluster_counts = np.unique(labels, return_counts=True)
cluster_with_most_points = unique_clusters[np.argmax(cluster_counts)]
feature_clusters = unique_clusters[(unique_clusters != cluster_with_most_points) & (unique_clusters != -1)]
feature_indices = labels[(labels != cluster_with_most_points) & (labels != -1)]

edge = o3d.geometry.PointCloud()
features = o3d.geometry.PointCloud()
for i in range(len(labels)):
    if labels[i] == cluster_with_most_points:
        edge.points.append(edges_full.points[i])
    elif labels[i] != -1:
        features.points.append(edges_full.points[i])


print("RANSAC line fitting")
lines = {}
line_param = {}
for i in range(4):
    edge_array = np.asarray(edge.points)
    line_fit, line_par = f_ransac(edge_array, iter=5000, threshold=0.5)
    variable_name = f"line{i + 1}"
    line_param[variable_name] = line_par
    lines[variable_name] = edge.select_by_index(line_fit)
    edge = edge.select_by_index(line_fit, invert=True)
def intersect(linepar1, linepar2):
    x = (linepar2[1] - linepar1[1])/(linepar1[0] - linepar2[0])
    y = linepar1[0] * x + linepar1[1]
    corner_point = np.array([x, y])
    return corner_point

corner_point1 = intersect(line_param["line1"], line_param["line3"])
corner_point2 = intersect(line_param["line2"], line_param["line3"])
corner_point3 = intersect(line_param["line1"], line_param["line4"])
corner_point4 = intersect(line_param["line2"], line_param["line4"])

print("Find COM of corner points")
centroid_x = np.average(np.array([corner_point1[0], corner_point2[0], corner_point3[0], corner_point4[0]]))
centroid_y = np.average(np.array([corner_point1[1], corner_point2[1], corner_point3[1], corner_point4[1]]))
centroid = np.array([centroid_x, centroid_y])

print("Move corner-points to origin system position")
P1 = corner_point1[0]-centroid[0], corner_point1[1]-centroid[1]
P2 = corner_point2[0]-centroid[0], corner_point2[1]-centroid[1]
P3 = corner_point3[0]-centroid[0], corner_point3[1]-centroid[1]
P4 = corner_point4[0]-centroid[0], corner_point4[1]-centroid[1]

print("Perform ICP rotational alignment")
Scan_Corners = np.array([P1, P2, P3, P4])
CAD_Corners = np.array([[25, 75], [-25, 75], [25, -75], [-25, -75]])

# Define the ICP algorithm

def icp_rotation(set1, set2, max_iterations=5000, tolerance=1e-6, distance_threshold=0.3):
    # Initialize the loop variables
    prev_error = np.inf
    for i in range(max_iterations):
        iteration_number = i
        # Find the nearest neighbors between the two sets
        nn = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(set2)
        distances, indices = nn.kneighbors(set1)

        # Compute the rotation matrix
        matched_points = set2[indices.ravel()]
        H = np.dot(set1.T, matched_points)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Apply the rotation to set1
        set1_aligned = np.dot(R, set1.T).T

        # Compute the error using manual Euclidean distance calculation
        squared_distances = np.sum((set1_aligned - matched_points) ** 2, axis=1)
        error = np.sqrt(np.mean(squared_distances))

        # Check for convergence
        if abs(prev_error - error) < tolerance and error < distance_threshold:
            break

        prev_error = error

    return set1_aligned, (iteration_number+1), R

# Align the two sets of points using ICP
Aligned_points, iterations, rot = icp_rotation(Scan_Corners, CAD_Corners)
R = np.eye(3)
R[:2, :2] = rot

Sorted_Corners = [[], [], [], []]
Sorted_Corners[0], Sorted_Corners[1], Sorted_Corners[2], Sorted_Corners[3] = sort_points(Aligned_points[0], Aligned_points[1], Aligned_points[2], Aligned_points[3])

print("Aligning features")
features_3darray = np.asarray(features.points)
features_array = features_3darray[:, :2]
features_array[:,0] = features_array[:,0] - centroid_x
features_array[:,1] = features_array[:,1] - centroid_y
aligned_features = np.dot(rot, features_array.T).T

feature_list = {}
feature_names = []
for i in range(len(feature_clusters)):
    feature_var = f"feature{feature_clusters[i]}"
    feature_names.append(feature_var)
    feature_list[feature_var] = o3d.geometry.PointCloud()

for i, label in enumerate(feature_indices):
    current_point = np.array([aligned_features[i,0], aligned_features[i,1], 0])
    feature_list[feature_names[label-1]].points.append(current_point)

feature_COMs = []
for point_cloud in feature_list.values():
    points = np.asarray(point_cloud.points)
    COMx = np.mean(points[:,0])
    COMy = np.mean(points[:,1])
    feature_COMs.append(np.array([COMx, COMy]))

# Convert to numpy array
feature_COMs = np.array(feature_COMs)

print("Defining CAD features")
class CADFeatures:
    def __init__(self):
        self.cir1 = Circle(np.array([-10, 50]), 0.1, 7.5, 0.1, 0) # top right
        self.cir2 = Circle(np.array([-10, -50]), 0.1, 7.5, 0.1, 1) # bot right
        self.rec1 = Rectangle(np.array([10, 30]), 0.1, 25, 10, 0.1, 2) # top left
        self.rec2 = Rectangle(np.array([10, -30]), 0.1, 25, 10, 0.1, 3) # bot left

class Circle:
    def __init__(self, center, pos_tol, radius, rad_tol, index):
        self.center = center
        self.pos_tol = pos_tol
        self.radius = radius
        self.rad_tol = rad_tol
        self.index = index

class Rectangle:
    def __init__(self, center, pos_tol, height, width, dim_tol, index):
        self.center = center
        self.pos_tol = pos_tol
        self.height = height
        self.width = width
        self.dim_tol = dim_tol
        self.index = index

CAD_features = CADFeatures()
CAD_COMs = np.array([CAD_features.cir1.center, CAD_features.cir2.center, CAD_features.rec1.center, CAD_features.rec2.center])

print("Identifying closest feature points")
# Find the nearest neighbors between the two sets
distances = cdist(CAD_COMs, feature_COMs)
closest_feature = np.argmin(distances, axis=1)

# closest_feature giver 4 index, som relaterer CAD_features til feature_list. fx hvis closest_feature[0] = 3
# så er den første cirkel i CAD_features tilsvarende den sidste (4.) feature i feature_list, hvilket er en cirkel.

# Extract x and y coordinates from the point lists
set1_x = [point[0] for point in CAD_Corners]
set1_y = [point[1] for point in CAD_Corners]
set2_x = [point[0] for point in Aligned_points]
set2_y = [point[1] for point in Aligned_points]
set3_x = [point[0] for point in CAD_COMs]
set3_y = [point[1] for point in CAD_COMs]
set4_x = [point[0] for point in feature_COMs]
set4_y = [point[1] for point in feature_COMs]

# Plot the points
plt.scatter(set1_x, set1_y, c='blue', label='CAD Corners')
plt.scatter(set2_x, set2_y, c='red', label='Scan Corners')
plt.scatter(set3_x, set3_y, c='green', label='CAD Feature Centers')
plt.scatter(set4_x, set4_y, c='yellow', label='Scan Feature Centers')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Show the plot
plt.show()

print("Evaluating corner tolerances")
TopRight = Sorted_Corners[0] - [25, 75]
TopLeft = Sorted_Corners[1] - [-25, 75]
BotRight = Sorted_Corners[2] - [25, -75]
BotLeft = Sorted_Corners[3] - [-25, -75]


print("Evaluating surface roughness")
a, b, c, d = plane_model[:4]

distances = []
for point in np.asarray(DBSCAN_sub.points):
    distance = abs(a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
    distances.append(distance)
rmse = np.sqrt(np.mean(np.asarray(distances) ** 2))

print("Evaluating tolerances for circle 1 (10,50)")
# make data
circle1 = np.asarray(feature_list[feature_names[closest_feature[0]]].points)
x_circle1 = circle1[:, 0]
y_circle1 = circle1[:, 1]
# n: how many times try sampling
ransac1 = RANSAC(x_circle1, y_circle1, n=50)
# execute ransac algorithm
ransac1.execute_ransac()
# get best model from ransac
a1, b1, r1 = ransac1.best_model[0], ransac1.best_model[1], ransac1.best_model[2]
# show result
circle1_tolerances = [[np.abs(CAD_features.cir1.center[0]-a1), np.abs(CAD_features.cir1.center[1]-b1)], np.abs(CAD_features.cir1.radius-r1)]

print("Evaluating tolerances for circle 2 (10,-50)")
# make data
circle2 = np.asarray(feature_list[feature_names[closest_feature[1]]].points)
x_circle2 = circle2[:, 0]
y_circle2 = circle2[:, 1]
# n: how many times try sampling
ransac2 = RANSAC(x_circle2, y_circle2, n=50)
# execute ransac algorithm
ransac2.execute_ransac()
# get best model from ransac
a2, b2, r2 = ransac2.best_model[0], ransac2.best_model[1], ransac2.best_model[2]
# show result
circle2_tolerances = [[np.abs(CAD_features.cir2.center[0]-a2), np.abs(CAD_features.cir2.center[1]-b2)], np.abs(CAD_features.cir2.radius-r2)]

print("Evaluating tolerances for rectangle 1 (-10, 30)")


lines_rec1 = {}
line_param_rec1 = {}
for i in range(4):
    edge_array = np.asarray(feature_list[feature_names[closest_feature[2]]].points)
    line_fit, line_par = f_ransac(edge_array, iter=5000, threshold=0.1)
    variable_name = f"line{i + 1}"
    line_param_rec1[variable_name] = line_par
    lines_rec1[variable_name] = feature_list[feature_names[closest_feature[2]]].select_by_index(line_fit)
    feature_list[feature_names[closest_feature[2]]] = feature_list[feature_names[closest_feature[2]]].select_by_index(line_fit, invert=True)
    #o3d.visualization.draw_geometries([lines_rec1[variable_name]])
    #o3d.visualization.draw_geometries([feature_list[feature_names[closest_feature[2]]]])

rec1_c1 = intersect(line_param_rec1["line1"], line_param_rec1["line3"]) # Top right
rec1_c2 = intersect(line_param_rec1["line2"], line_param_rec1["line3"]) # Top left
rec1_c3 = intersect(line_param_rec1["line1"], line_param_rec1["line4"]) # Bot right
rec1_c4 = intersect(line_param_rec1["line2"], line_param_rec1["line4"]) # Bot left

rec1_corner1, rec1_corner2, rec1_corner3, rec1_corner4 = sort_points(rec1_c1, rec1_c2, rec1_c3, rec1_c4)

CAD_rec1_corner1 = [(CAD_features.rec1.center[0] + (CAD_features.rec1.width / 2)), (CAD_features.rec1.center[1] + (CAD_features.rec1.height / 2))] # Top right
CAD_rec1_corner2 = [(CAD_features.rec1.center[0] - (CAD_features.rec1.width / 2)), (CAD_features.rec1.center[1] + (CAD_features.rec1.height / 2))] # Top left
CAD_rec1_corner3 = [(CAD_features.rec1.center[0] + (CAD_features.rec1.width / 2)), (CAD_features.rec1.center[1] - (CAD_features.rec1.height / 2))] # Bot right
CAD_rec1_corner4 = [(CAD_features.rec1.center[0] - (CAD_features.rec1.width / 2)), (CAD_features.rec1.center[1] - (CAD_features.rec1.height / 2))] # Bot left

# Center Position
rec1_tol_centerx = np.mean([CAD_rec1_corner1[0], CAD_rec1_corner2[0], CAD_rec1_corner3[0], CAD_rec1_corner4[0]]) - np.mean([rec1_corner1[0], rec1_corner2[0], rec1_corner3[0], rec1_corner4[0]])
rec1_tol_centery = np.mean([CAD_rec1_corner1[1], CAD_rec1_corner2[1], CAD_rec1_corner3[1], CAD_rec1_corner4[1]]) - np.mean([rec1_corner1[1], rec1_corner2[1], rec1_corner3[1], rec1_corner4[1]])
rec1_tol_center = [rec1_tol_centerx, rec1_tol_centery]

# Top position
rec1_tol_top_avg = np.mean([CAD_rec1_corner1[1], CAD_rec1_corner2[1]])-np.mean([rec1_corner1[1], rec1_corner2[1]])
rec1_tol_top_max = max([(CAD_rec1_corner1[1]-rec1_corner1[1]), (CAD_rec1_corner2[1]-rec1_corner2[1])])

# Bot position
rec1_tol_bot_avg = np.mean([CAD_rec1_corner3[1], CAD_rec1_corner4[1]])-np.mean([rec1_corner3[1], rec1_corner4[1]])
rec1_tol_bot_max = max([(CAD_rec1_corner3[1]-rec1_corner3[1]), (CAD_rec1_corner4[1]-rec1_corner4[1])])

# Right position
rec1_tol_right_avg = np.mean([CAD_rec1_corner1[0], CAD_rec1_corner3[0]])-np.mean([rec1_corner1[0], rec1_corner3[0]])
rec1_tol_right_max = max([(CAD_rec1_corner1[0]-rec1_corner1[0]), (CAD_rec1_corner3[0]-rec1_corner3[0])])

# Left position
rec1_tol_left_avg = np.mean([CAD_rec1_corner2[0], CAD_rec1_corner4[0]])-np.mean([rec1_corner2[0], rec1_corner4[0]])
rec1_tol_left_max = max([(CAD_rec1_corner2[0]-rec1_corner2[0]), (CAD_rec1_corner4[0]-rec1_corner4[0])])

rec1_tolerances = [rec1_tol_center, [rec1_tol_top_avg, rec1_tol_bot_avg, rec1_tol_right_avg, rec1_tol_left_avg], [rec1_tol_top_max, rec1_tol_bot_max, rec1_tol_right_max, rec1_tol_left_max]]



print("Evaluating tolerances for rectangle 2 (-10, -30)")
lines_rec2 = {}
line_param_rec2 = {}
for i in range(4):
    edge_array = np.asarray(feature_list[feature_names[closest_feature[3]]].points)
    line_fit, line_par = f_ransac(edge_array, iter=5000, threshold=0.1)
    variable_name = f"line{i + 1}"
    line_param_rec2[variable_name] = line_par
    lines_rec2[variable_name] = feature_list[feature_names[closest_feature[3]]].select_by_index(line_fit)
    feature_list[feature_names[closest_feature[3]]] = feature_list[feature_names[closest_feature[3]]].select_by_index(line_fit, invert=True)
    #o3d.visualization.draw_geometries([lines_rec1[variable_name]])
    #o3d.visualization.draw_geometries([feature_list[feature_names[closest_feature[2]]]])

rec2_c1 = intersect(line_param_rec2["line1"], line_param_rec2["line3"]) # Top right
rec2_c2 = intersect(line_param_rec2["line2"], line_param_rec2["line3"]) # Top left
rec2_c3 = intersect(line_param_rec2["line1"], line_param_rec2["line4"]) # Bot right
rec2_c4 = intersect(line_param_rec2["line2"], line_param_rec2["line4"]) # Bot left

rec2_corner1, rec2_corner2, rec2_corner3, rec2_corner4 = sort_points(rec2_c1, rec2_c2, rec2_c3, rec2_c4)

CAD_rec2_corner1 = [(CAD_features.rec2.center[0] + (CAD_features.rec2.width / 2)), (CAD_features.rec2.center[1] + (CAD_features.rec2.height / 2))] # Top right
CAD_rec2_corner2 = [(CAD_features.rec2.center[0] - (CAD_features.rec2.width / 2)), (CAD_features.rec2.center[1] + (CAD_features.rec2.height / 2))] # Top left
CAD_rec2_corner3 = [(CAD_features.rec2.center[0] + (CAD_features.rec2.width / 2)), (CAD_features.rec2.center[1] - (CAD_features.rec2.height / 2))] # Bot right
CAD_rec2_corner4 = [(CAD_features.rec2.center[0] - (CAD_features.rec2.width / 2)), (CAD_features.rec2.center[1] - (CAD_features.rec2.height / 2))] # Bot left

# Center Position
rec2_tol_centerx = np.mean([CAD_rec2_corner1[0], CAD_rec2_corner2[0], CAD_rec2_corner3[0], CAD_rec2_corner4[0]]) - np.mean([rec2_corner1[0], rec2_corner2[0], rec2_corner3[0], rec2_corner4[0]])
rec2_tol_centery = np.mean([CAD_rec2_corner1[1], CAD_rec2_corner2[1], CAD_rec2_corner3[1], CAD_rec2_corner4[1]]) - np.mean([rec2_corner1[1], rec2_corner2[1], rec2_corner3[1], rec2_corner4[1]])
rec2_tol_center = [rec2_tol_centerx, rec2_tol_centery]

# Top position
rec2_tol_top_avg = np.mean([CAD_rec2_corner1[1], CAD_rec2_corner2[1]])-np.mean([rec2_corner1[1], rec2_corner2[1]])
rec2_tol_top_max = max([(CAD_rec2_corner1[1]-rec2_corner1[1]), (CAD_rec2_corner2[1]-rec2_corner2[1])])

# Bot position
rec2_tol_bot_avg = np.mean([CAD_rec2_corner3[1], CAD_rec2_corner4[1]])-np.mean([rec2_corner3[1], rec2_corner4[1]])
rec2_tol_bot_max = max([(CAD_rec2_corner3[1]-rec2_corner3[1]), (CAD_rec2_corner4[1]-rec2_corner4[1])])

# Right position
rec2_tol_right_avg = np.mean([CAD_rec2_corner1[0], CAD_rec2_corner3[0]])-np.mean([rec2_corner1[0], rec2_corner3[0]])
rec2_tol_right_max = max([(CAD_rec2_corner1[0]-rec2_corner1[0]), (CAD_rec2_corner3[0]-rec2_corner3[0])])

# Left position
rec2_tol_left_avg = np.mean([CAD_rec2_corner2[0], CAD_rec2_corner4[0]])-np.mean([rec2_corner2[0], rec2_corner4[0]])
rec2_tol_left_max = max([(CAD_rec2_corner2[0]-rec2_corner2[0]), (CAD_rec2_corner4[0]-rec2_corner4[0])])

rec2_tolerances = [rec2_tol_center, [rec2_tol_top_avg, rec2_tol_bot_avg, rec2_tol_right_avg, rec2_tol_left_avg], [rec2_tol_top_max, rec2_tol_bot_max, rec2_tol_right_max, rec2_tol_left_max]]

print("Corner point locations:")
print("Top right = {},".format(TopRight)+" Top left = {},".format(TopLeft)+" Bot right = {},".format(BotRight)+" Bot left = {}".format(BotLeft))
print("Tolerances:")
print("Surface roughness (RMSE) = {}".format(rmse))
print("Circle 1:")
print("center position tolerance = {}".format(circle1_tolerances[0]))
print("radius tolerance = {}".format(circle1_tolerances[1]))
print("Circle 2:")
print("center position tolerance = {}".format(circle2_tolerances[0]))
print("radius tolerance = {}".format(circle2_tolerances[1]))
print("Rectangle 1:")
print("center position tolerance = {}".format(rec1_tolerances[0]))
print("average of sides: Top, Bot, Right, Left = {}".format(rec1_tolerances[1]))
print("maximum of sides: Top, Bot, Right, Left = {}".format(rec1_tolerances[2]))
print("Rectangle 2:")
print("center position tolerance = {}".format(rec2_tolerances[0]))
print("average of sides: Top, Bot, Right, Left = {}".format(rec2_tolerances[1]))
print("maximum of sides: Top, Bot, Right, Left = {}".format(rec2_tolerances[2]))

# Store the values in variables
circle1_pos, circle1_rad = circle1_tolerances
circle2_pos, circle2_rad = circle2_tolerances
rec1_cen, rec1_sides_avg, rec1_sides_max = rec1_tolerances
rec2_cen, rec2_sides_avg, rec2_sides_max = rec2_tolerances

# Create a list of dictionaries with the variable names and values
data = [
    {
        'TopRight': TopRight[0],
        'TopLeft': TopLeft[0],
        'BotRight': BotRight[0],
        'BotLeft': BotLeft[0],
        'RMSE': rmse,
        'Circle1Pos': circle1_pos[0],
        'Circle1Rad': circle1_rad,
        'Circle2Pos': circle2_pos[0],
        'Circle2Rad': circle2_rad,
        'Rec1Cen': rec1_cen[0],
        'Rec1SidesAvg': rec1_sides_avg[0],
        'Rec1SidesMax': rec1_sides_max[0],
        'Rec2Cen': rec2_cen[0],
        'Rec2SidesAvg': rec2_sides_avg[0],
        'Rec2SidesMax': rec2_sides_max[0]
    },
    {
        'TopRight': TopRight[1],
        'TopLeft': TopLeft[1],
        'BotRight': BotRight[1],
        'BotLeft': BotLeft[1],
        'RMSE': '',
        'Circle1Pos': circle1_pos[1],
        'Circle1Rad': '',
        'Circle2Pos': circle2_pos[1],
        'Circle2Rad': '',
        'Rec1Cen': rec1_cen[1],
        'Rec1SidesAvg': rec1_sides_avg[1],
        'Rec1SidesMax': rec1_sides_max[1],
        'Rec2Cen': rec2_cen[1],
        'Rec2SidesAvg': rec2_sides_avg[1],
        'Rec2SidesMax': rec2_sides_max[1]
    },
    {
        'TopRight': '',
        'TopLeft': '',
        'BotRight': '',
        'BotLeft': '',
        'RMSE': '',
        'Circle1Pos': '',
        'Circle1Rad': '',
        'Circle2Pos': '',
        'Circle2Rad': '',
        'Rec1Cen': '',
        'Rec1SidesAvg': rec1_sides_avg[2],
        'Rec1SidesMax': rec1_sides_max[2],
        'Rec2Cen': '',
        'Rec2SidesAvg': rec2_sides_avg[2],
        'Rec2SidesMax': rec2_sides_max[2]
    },
    {
        'TopRight': '',
        'TopLeft': '',
        'BotRight': '',
        'BotLeft': '',
        'RMSE': '',
        'Circle1Pos': '',
        'Circle1Rad': '',
        'Circle2Pos': '',
        'Circle2Rad': '',
        'Rec1Cen': '',
        'Rec1SidesAvg': rec1_sides_avg[3],
        'Rec1SidesMax': rec1_sides_max[3],
        'Rec2Cen': '',
        'Rec2SidesAvg': rec2_sides_avg[3],
        'Rec2SidesMax': rec2_sides_max[3]
    }
]



# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Create an Excel writer using pandas
writer = pd.ExcelWriter(name, engine='xlsxwriter')

# Write the DataFrame to the Excel file
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Save the Excel file
writer.save()
