import cv2
import numpy as np
import math
from collections import deque


RED = [0, 0, 255]
YELLOW = [0, 255, 255]
BLUE = [255, 0, 0]
PINK = [203, 192, 255]
BLACK = [0, 0, 0]
GREEN = [0, 255, 0]
CYAN = [255, 255, 0]
MAGENTA = [255, 0, 255]


# in BGR and not RGB
# the 3rd value represents the index of the color in the grid
colors = {
    'red': ([136, 87, 111], [180, 255, 255], -4, RED), #1, -4
    'yellow': ([0, 72, 81], [40, 255, 255], -5, YELLOW), #2, -5
    'blue': ([80, 100, 50], [120, 255, 110], -3, BLUE), #3, -3
    'pink': ([145, 100, 100], [170, 255, 255], 6, PINK), #6
    'black': ([0, 0, 0], [180, 255, 50], -2, BLACK), #11, -2
}


kernel_size = (5, 5)

anchors_color = "pink"
pastilles_colors = ["red"]
robot_color = "yellow"
obstacle_color = "black"

detected_anchors = []
grilles = None

resolution = (0.5, 0.5)
dist_anch_thresh = 50
min_area_anchors = 500
min_area_obstacle = 2000
target_radius_thresh = (10,100)
min_number_circles = 20

dist_robot_thresh = 60
min_triangle_area = 700

show_grid_colors = True
show_cropped_frame = True

anchors_found = False

finished = False

min_dist_obstacles = 10
number_detection_obstacles = 10
margin_obstacles = 200

min_area_calib = 10



def get_anchors(frame, show=False, detected_anchors=None):
    """
    Detects the anchors in the frame and updates the detected_anchors list.

    Args:
        frame (np.ndarray): Original cropped image to analyze.
        show (bool, optional): Whether to show the detection on the frame. Defaults to False.
        detected_anchors (list, optional): The list of detected anchors. Defaults to None.

    Returns:
        list : The list of detected anchors.
    """
    

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, kernel_size, 0)
    hsv = cv2.medianBlur(hsv, 5)
    kernel = np.ones(kernel_size, np.uint8)
    
    lower, upper, _, _ = colors[anchors_color]
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    
    dilation = cv2.dilate(mask, kernel, iterations=2)
    erosion = cv2.erode(dilation, kernel, iterations=2)
    
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(erosion, connectivity=8)
    
    for i in range(1, num_labels):  # ignore the background (index 0)
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        if min_area_anchors < area:
            # get the region of interest
            roi = erosion[y:y+h, x:x+w]
            roi_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in roi_contours:
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    cX, cY = int(centroids[i][0]), int(centroids[i][1])
                    norm = math.sqrt(cX**2 + cY**2)
                    
                    # Verify if the anchor is new
                    is_new_anchor = True
                    for existing_anchor in detected_anchors:
                        existing_norm = math.sqrt(existing_anchor[0]**2 + existing_anchor[1]**2)
                        if abs(norm - existing_norm) < dist_anch_thresh:
                            is_new_anchor = False
                            break
                    
                    # Add the anchor if it is new
                    if is_new_anchor:
                        detected_anchors.append((cX, cY, norm))
                    
                    if show:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 3)
                        cv2.putText(frame, "Anchor", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    # Keep only the 4 first anchors
    if len(detected_anchors) > 4:
        detected_anchors = detected_anchors[:4]
    
    return detected_anchors


def detect_and_update_circles(frame_pure, cropped_frame, grilles, anchors_positions):
    """
    Detects the largest circle of each color, traces all the points inside and its center.
    It uses the Hough Circle Transform for circle detection and Connected Components for color detection.

    Args:
        frame_pure (np.ndarray): Original cropped image to analyze.
        cropped_frame (np.ndarray): Cropped image to draw on and display.
        grilles (Grid): Instance of the grid to update the positions.
        anchors_positions (list): Positions of the anchors for mapping.

    """

    gray = cv2.cvtColor(frame_pure, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # to reduce noise

    # Hough Circle Transform for circle detection
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=target_radius_thresh[0],
        maxRadius=target_radius_thresh[1]
    )

    colors_used = get_sub_colors(colors, pastilles_colors)
    largest_circles_by_color = {}
    hsv = cv2.cvtColor(frame_pure, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, kernel_size, 0)
    hsv = cv2.medianBlur(hsv, 5)
    kernel = np.ones(kernel_size, np.uint8)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for color_name, color_values in colors_used.items():
            lower, upper, _, _ = color_values
            
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            dilation = cv2.dilate(mask, kernel, iterations=2)
            erosion = cv2.erode(dilation, kernel, iterations=2)
            
            # Apply Connected Components to find the largest circle in the color region
            _, labels, stats, _ = cv2.connectedComponentsWithStats(erosion, connectivity=8)
            max_area = 0
            largest_circle = None

            for circle in circles[0, :]:
                x, y, radius = circle
                
                # Circle outside the frame
                if not (0 <= x < anchors_positions[3][0] - anchors_positions[0][0] and 0 <= y < anchors_positions[3][1] - anchors_positions[0][1]):
                    continue 

            # Verification if the radius is correct
                if not (radius > 0 and radius < min(anchors_positions[3][0] - anchors_positions[0][0], anchors_positions[3][1] - anchors_positions[0][1]) / 2):
                    continue 

                # Verify if the circle is inside the color region
                if labels[y, x] > 0:
                    region_area = stats[labels[y, x], cv2.CC_STAT_AREA]
                    if region_area > max_area:
                        max_area = region_area
                        largest_circle = (x, y, radius)

            if largest_circle:
                largest_circles_by_color[color_name] = largest_circle
        
    got_circle = 0

    for color_name, circle in largest_circles_by_color.items():
        x1, y1, radius1 = circle

        cv2.circle(cropped_frame, (x1, y1), radius1, BLACK, 2) 
        cv2.circle(cropped_frame, (x1, y1), 2, BLACK, -1)
        
        # count the number of circles of each color
        grilles.number_circles_colors[color_name] += 1
        
        if  grilles.number_circles_colors[color_name] < min_number_circles + 1:
            grilles.stock_circles_positions[color_name].append((x1, y1, radius1))
            
        else:
            got_circle += min_number_circles
            
        
    # if we have enough circles, ie enough detections
    if got_circle == min_number_circles * len(pastilles_colors):
        grilles.enough_circles = True
        grilles.circles_frame = smooth_circles(grilles.stock_circles_positions)
        
        for color_name, vals in grilles.circles_frame.items():
            final_grid_x = int(map_range(vals[0], 0, anchors_positions[3][0] - anchors_positions[0][0], 0, grilles.bottom_right[0]))
            final_grid_y = int(map_range(vals[1], 0, anchors_positions[3][1] - anchors_positions[0][1], 0, grilles.bottom_right[1]))
            final_grid_radius = int(map_range(vals[2], 0, anchors_positions[3][0] - anchors_positions[0][0], 0, grilles.bottom_right[0]))
            
            mask_circle = np.zeros((grilles.bottom_right[1], grilles.bottom_right[0]), dtype=np.int8)

            cv2.circle(mask_circle, (final_grid_x, final_grid_y), final_grid_radius, colors_used[color_name][2], thickness=-1)
            grilles.circles_positions[color_name] = (final_grid_x, final_grid_y, final_grid_radius)
            grilles.set_grid(color_name ,cv2.bitwise_or(grilles.get_grid(color_name), cv2.transpose(mask_circle)))
    


def detect_and_update_robot(frame_pure, cropped_frame, grilles, anchors_positions):
    """
    Detects the largest triangle of each color, traces all the points inside and its center.

    Args:
        frame_pure (np.ndarray): Original cropped image to analyze.
        cropped_frame (np.ndarray): Cropped image to draw on and display.
        grilles (Grid): Instance of the grid to update the positions.
        anchors_positions (list): Positions of the anchors for mapping.

    Returns:
    dict : The smoothed position and angle of the robot.
    """

    hsv = cv2.cvtColor(frame_pure, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, kernel_size, 0)
    hsv = cv2.medianBlur(hsv, 5)
    kernel = np.ones(kernel_size, np.uint8)

    colors_used = get_sub_colors(colors, robot_color)
    
    largest_triangle = None
    largest_area = 0
    largest_triangle_info = None


    for color_name, color_values in colors_used.items():
        lower, upper, index, _ = color_values
        
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        dilation = cv2.dilate(mask, kernel, iterations=2)
        erosion = cv2.erode(dilation, kernel, iterations=2)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(erosion)

        for i in range(1, num_labels):  # ignore the background (index 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if min_triangle_area < area :
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                # get the region of interest
                roi = erosion[y:y+h, x:x+w]
                roi_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in roi_contours:
                    epsilon = 0.04 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    if len(approx) == 3 and cv2.isContourConvex(approx):  # Check for triangles
                        if area > largest_area:  # Keep the largest triangle
                            largest_area = area
                            largest_triangle = approx.copy().reshape(3, 2)
                            for i in range(0,len(largest_triangle)):
                                largest_triangle[i][0] = largest_triangle[i][0] + x
                                largest_triangle[i][1] = largest_triangle[i][1] + y
                                
                            angle, center = find_robot(largest_triangle)
                            largest_triangle_info = {
                                "color": color_name,
                                "angle": angle,
                                "center": center,
                                "index": index,
                                "area": area
                            }

    if largest_triangle is not None:
        center_x, center_y = largest_triangle_info["center"]
        center_frame = (int(center_x), int(center_y))
        color_name = largest_triangle_info["color"]
        angle = largest_triangle_info["angle"]
        
        
        # Create a mask with the filled triangle
        triangle_mask = np.zeros((grilles.bottom_right[1], grilles.bottom_right[0]), dtype=np.int8)
        triangle_points = [
            (
                int(map_range(pt[0], 0, anchors_positions[3][0] - anchors_positions[0][0], 0, grilles.bottom_right[0])),
                int(map_range(pt[1], 0, anchors_positions[3][1] - anchors_positions[0][1], 0, grilles.bottom_right[1]))
            )
            for pt in largest_triangle
        ]
        cv2.fillPoly(triangle_mask, [np.array(triangle_points, dtype=np.int32)], largest_triangle_info["index"])

        # Get the information of the triangle
        
        center_grid_x = int(map_range(center_x, 0, anchors_positions[3][0] - anchors_positions[0][0], 0, grilles.bottom_right[0]))
        center_grid_y = int(map_range(center_y, 0, anchors_positions[3][1] - anchors_positions[0][1], 0, grilles.bottom_right[1]))
        center_grid = (center_grid_x, center_grid_y)


        if not grilles.stock_robot_positions or np.linalg.norm(np.array(center_grid) - np.array(grilles.stock_robot_positions[-1][0])) <= dist_robot_thresh:
            # Update the informations
            grilles.robot_detected = True
            grilles.robot_frame_position = (center_frame, angle)
            grilles.stock_robot_positions.append((center_grid, angle))
            grilles.trajectory.append(center_frame)
            grilles.set_grid(color_name, cv2.bitwise_or(grilles.get_grid(color_name), cv2.transpose(triangle_mask)))
            grilles.count_missed = 0
            
            cv2.circle(cropped_frame, center_frame, 5, BLACK, -1)
            cv2.drawContours(cropped_frame, [largest_triangle], -1, BLACK, 2)
            print(largest_triangle_info["area"])
    else:
        # we reset the values if the robot is not detected
        grilles.robot_detected = False
        grilles.count_missed += 1
        if grilles.count_missed > 5:
            grilles.stock_robot_positions = deque(maxlen=2)
            grilles.count_missed = 0
                
    return smooth_robot(grilles.stock_robot_positions)


def detect_obstacles(frame_pure, cropped_frame, grilles, anchors_positions):
    """
    Detects the obstacles in the frame and updates the grid with the detected obstacles.

    Args:
        frame_pure (np.ndarray): Original cropped image to analyze.
        cropped_frame (np.ndarray): Cropped image to draw on and display.
        grilles (Grid): Instance of the grid to update the positions.
        anchors_positions (list): Positions of the anchors for mapping.
    """
    
    hsv = cv2.cvtColor(frame_pure, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, kernel_size, 0)
    hsv = cv2.medianBlur(hsv, 5)
    kernel = np.ones(kernel_size, np.uint8)

    lower, upper, index, _ = colors[obstacle_color]
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    
    dilation = cv2.dilate(mask, kernel, iterations=2)
    erosion = cv2.erode(dilation, kernel, iterations=2)
    
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(erosion, connectivity=8)
    
    enlarged_vertices = []
    for i in range(1, num_labels):  # ignore the background (index 0)
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        if min_area_obstacle < area:
            # get the region of interest
            roi = erosion[y:y+h, x:x+w]
            roi_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in roi_contours:
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    vertices = approx[:, 0, :] + [x, y]  # shift the vertices to the original frame
                    enlarged_vertices = enlarge_polygon(vertices)
                    
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"]) + x
                        center_y = int(M["m01"] / M["m00"]) + y
                        center = (center_x, center_y)
                    else:
                        center = (0, 0)
                    
                    # get the convex hull of the enlarged vertices
                    if len(enlarged_vertices) > 0:
                        hull = cv2.convexHull(enlarged_vertices)
                        hull_vertices = hull[:, 0, :]
                        
                        for vertex in hull_vertices:  # draw the vertices of the hull
                            cv2.circle(cropped_frame, tuple(vertex), 5, GREEN, -1)
                        cv2.drawContours(cropped_frame, [vertices], -1, RED, 2)
                        cv2.drawContours(cropped_frame, [hull_vertices], -1, YELLOW, 2)
                        cv2.circle(cropped_frame, center, 5, RED, -1)

                        grilles.update_stock_obstacles_positions(center, hull_vertices, min_dist_obstacles, number_detection_obstacles)
    
    # if enough obstacles are detected to consider the obstacles as constant
    if grilles.enough_obstacles:
        for _, data in grilles.constant_polygons.items():
            hull = cv2.convexHull(data)
            hull_vertices = hull[:, 0, :]
            mapped_hull = [
                (
                    int(map_range(pt[0], 0, anchors_positions[3][0] - anchors_positions[0][0], 0, grilles.bottom_right[0])),
                    int(map_range(pt[1], 0, anchors_positions[3][1] - anchors_positions[0][1], 0, grilles.bottom_right[1]))
                ) for pt in hull_vertices
            ]
            
            # to fill the grid with the obstacles
            hull_mask = np.zeros((grilles.bottom_right[1], grilles.bottom_right[0]), dtype=np.int8)
            cv2.fillPoly(hull_mask, [np.array(mapped_hull, dtype=np.int32)], index)
            
            grilles.set_grid(obstacle_color, cv2.bitwise_or(grilles.get_grid(obstacle_color), cv2.transpose(hull_mask)))



def map_range(value, input_min, input_max, output_min, output_max):
    """
    Map a value from one range to another.
    
    Args:
        value (float): the value to map.
        input_min (float): the minimum value of the input range.
        input_max (float): the maximum value of the input range.
        output_min (float): the minimum value of the output range.
        output_max (float): the maximum value of the output range.
    
    Returns:
        float: the mapped value.
    """
    if input_min == input_max:
        raise ValueError("The input range cannot have the same minimum and maximum values.")
    
    normalized_value = (value - input_min) / (input_max - input_min)
    mapped_value = output_min + (normalized_value * (output_max - output_min))
    
    return mapped_value


def find_robot(vertices):
    """
    Find the orientation angle of the triangle and the center of the triangle.

    Args:
        vertices (np.ndarray): Vertices of the triangle.

    Returns:
        angle (float): The orientation angle of the triangle.
        midpoint (tuple): The center of the triangle.
    """
    
    side_lengths = [
        np.linalg.norm(vertices[0] - vertices[1]),
        np.linalg.norm(vertices[1] - vertices[2]),
        np.linalg.norm(vertices[2] - vertices[0])
    ]
    
    # identify the shortest side and the back edge
    shortest_side_idx = np.argmin(side_lengths)
    if shortest_side_idx == 0:
        back_edge = (vertices[0], vertices[1])
        front_vertex = vertices[2]
    elif shortest_side_idx == 1:
        back_edge = (vertices[1], vertices[2])
        front_vertex = vertices[0]
    else:
        back_edge = (vertices[2], vertices[0])
        front_vertex = vertices[1]

    # Find the midpoint of the back edge
    mid_back = ((back_edge[0][0] + back_edge[1][0]) / 2, (back_edge[0][1] + back_edge[1][1]) / 2)

    # Calculate the angle of the line with respect to the horizontal
    dx = front_vertex[0] - mid_back[0]
    dy = front_vertex[1] - mid_back[1]
    
    # Compute the angle in degrees, with respect to the horizontal axis
    theta = np.arctan2(dy, dx) * 180 / np.pi

    # Normalize the angle in the range [0, 360], to modify if needed
    theta_N = (theta + 360) % 360
    middle = np.mean(vertices, axis=0)
    xC, yC = middle[0], middle[1]
    

    return np.round(theta_N,1), (int(xC), int(yC))


def get_sub_colors(color_ranges, sub_colors):
    """ Get the color ranges for the specified sub-colors.

    Args:
        color_ranges (dict): Dictionary containing the color ranges.
        sub_colors (list): List of sub-colors to check.

    Raises:
        ValueError: If a color name is not found in the color_ranges dictionary.

    Returns:
        dict: Dictionary containing the colors for the specified sub-colors.
    """
    
    if isinstance(sub_colors, str):
            sub_colors = [sub_colors] # depending on the use, we pass an array or a string
    
    colors_used = {}
    for color_name in sub_colors:
        if color_name not in color_ranges:
            raise ValueError(f"Color '{color_name}' not found in the 'color_ranges' dictionary.")
        else:
            colors_used[color_name] = color_ranges[color_name]
    return colors_used


def draw_anchors_lines(frame, anchors_positions):
    """ Draw lines between the anchors on the frame.

    Args:
        cropped_frame (np.ndarray): Cropped image to draw on and display.
        anchors_positions (dict): the positions of the anchors.
    """
    cv2.line(frame, anchors_positions[0][:2], anchors_positions[1][:2], BLUE, 2)
    cv2.line(frame, anchors_positions[0][:2], anchors_positions[2][:2], BLUE, 2)
    cv2.line(frame, anchors_positions[1][:2], anchors_positions[3][:2], BLUE, 2)
    cv2.line(frame, anchors_positions[2][:2], anchors_positions[3][:2], BLUE, 2)


def enlarge_polygon(vertices):
    """ Enlarge the polygon by adding a margin around it.

    Args:
        vertices (list): the vertices of the polygon.
        margin (int, optional): the margin to add around the polygon. Defaults to 15.
    Returns:
        np.array: the enlarged vertices.
    """
    enlarged_vertices = []
    num_vertices = len(vertices)

    for i in range(num_vertices):
        # for each vertex, get the previous and next vertex, then the edge vectors and the normal
        prev_idx = (i - 1) % num_vertices
        next_idx = (i + 1) % num_vertices

        edge1 = vertices[i] - vertices[prev_idx]
        edge2 = vertices[next_idx] - vertices[i]

        edge1_normalized = np.array([-edge1[1], edge1[0]], dtype=np.float64)
        edge2_normalized = np.array([-edge2[1], edge2[0]], dtype=np.float64)

        edge1_normalized /= np.linalg.norm(edge1_normalized)
        edge2_normalized /= np.linalg.norm(edge2_normalized)

        normal = 0.5*(edge1_normalized + edge2_normalized)
        enlarged_vertex = vertices[i] + margin_obstacles * normal
        enlarged_vertices.append(enlarged_vertex)

    return np.array(enlarged_vertices, dtype=np.int32)


def smooth_robot(vals):
    """ Smooth the robot positions by averaging the positions.

    Args:
        vals (dict): the positions and angles of the robot.

    Returns:
        tuple: the average position and angle of the robot.
    """
    if not vals:
        return None 
    
    avg_a = sum(val[0][0] for val in vals) // len(vals)
    avg_b = sum(val[0][1] for val in vals) // len(vals)
    avg_c = sum(val[1] for val in vals) / len(vals)
    
    return ((avg_a, avg_b), round(avg_c,1))


def smooth_circles(vals):
    """ Smooth the circles positions by averaging the positions.

    Args:
        vals (dict): the positions and radii of the circles.

    Returns:
        dict: the average positions and radii of the circles.
    """
    final = {}
    
    for color_name, data in vals.items():

        x_sum, y_sum, r_sum = map(sum, zip(*data))
        nb = len(data)
        final[color_name] = (x_sum // nb, y_sum // nb, r_sum // nb)

    return final


def found_target(robot_position, targets):
    """ Check if the robot is on a target.

    Args:
        robot_position (tuple): the position and angle of the robot.
        targets (dict): the positions and radii of the targets.

    Returns:
        bool/string: return the color of the target if found, False otherwise.
    """
    
    for color_name, target in targets.items():
        center_x, center_y, radius =  target
        center = (center_x, center_y)
        if robot_position is not None:
            center_robot = robot_position[0]
            if np.linalg.norm(np.array(center_robot) - np.array(center)) <= radius:
                return color_name

    return False




class Grid:
    def __init__(self):
        # the grids for each color
        self.grids = {}
        self.grid_size = 0


        # the parameters for the anchors
        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0
        self.anchors_positions = 0
        self.detected_anchors = []
        self.anchors_found = False
        self.show_dect_anchors = True
        
        self.top_left = (0,0)
        self.bottom_right = 0
        self.taille_crop = (0,0)
        
        # the parameters for the circles
        self.circles_positions = {}
        self.circles_frame = {}
        self.stock_circles_positions = {}
        self.number_circles_colors = {}
        self.enough_circles = False
        
        # the parameters for the obstacles
        self.obstacles_positions = {}
        self.stock_obstacles_positions = {}
        self.constant_polygons = {}
        self.enough_obstacles = False
        
        # the parameters for the robot
        self.robot_position = []
        self.robot_frame_position = []
        self.init_robot = False
        self.stock_robot_positions = deque(maxlen=2) # 2 last positions
        self.trajectory = deque(maxlen=1000)
        self.count_missed = 0
        self.robot_detected = False
        
        # the parameters for the grid
        self.grid_colors = 0
        self.final_grid = 0
        
        self.found_everything = False

        
    def reset_instance(self):
        self.grids = {}
        self.grid_size = 0


        # the parameters for the anchors
        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0
        self.anchors_positions = 0
        self.detected_anchors = []
        self.anchors_found = False
        self.show_dect_anchors = True
        
        self.top_left = (0,0)
        self.bottom_right = 0
        self.taille_crop = (0,0)
        
        # the parameters for the circles
        self.circles_positions = {}
        self.circles_frame = {}
        self.stock_circles_positions = {}
        self.number_circles_colors = {}
        self.enough_circles = False
        
        # the parameters for the obstacles
        self.obstacles_positions = {}
        self.stock_obstacles_positions = {}
        self.constant_polygons = {}
        self.enough_obstacles = False
        
        # the parameters for the robot
        self.robot_position = []
        self.robot_frame_position = []
        self.init_robot = False
        self.stock_robot_positions = deque(maxlen=2) # 2 last positions
        self.trajectory = deque(maxlen=1000)
        self.count_missed = 0
        self.robot_detected = False
        
        # the parameters for the grid
        self.grid_colors = 0
        self.final_grid = 0
        
        self.found_everything = False

    def detect_black_rectangles(self,frame):
        """
        Used for calibration purposes : conversion between the grid and the real world

        Args:
            frame (np.ndarray): Original cropped image to analyze.

        Returns:
            (np.ndarray), list: The cropped frame and the sizes of the rectangles.
        """
        
        cropped_frame = frame.copy()
        
        rectangle_sizes = []
        
        if not self.anchors_found:
            self.detected_anchors = get_anchors(frame, self.show_dect_anchors, self.detected_anchors)
            for anchor in self.detected_anchors:
                cv2.circle(cropped_frame, anchor[:2], 5, GREEN, -1)
        
        if len(self.detected_anchors) == 4: # once the anchors are detected
            
            if not self.anchors_found:
                self.show_dect_anchors = False
                self.anchors_positions = sorted(self.detected_anchors, key=lambda p: (p[2], p[0]))
                top_left = self.anchors_positions[0][:2]
                bottom_right = self.anchors_positions[3][:2]
                
                # get the coordinates
                self.x_min, self.y_min = map(int, top_left)
                self.x_max, self.y_max = map(int, bottom_right)    
                
            self.anchors_found = True
            
            # get the cropped frame
            cropped_frame = frame[self.y_min:self.y_max, self.x_min:self.x_max]
        
        
            hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.medianBlur(hsv, 5)

            lower, upper, _, _ = colors["black"]
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            
            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(mask, kernel, iterations=2)
            erosion = cv2.erode(dilation, kernel, iterations=2)

            contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rectangle_sizes = []
            
            for contour in contours:

                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                area = cv2.contourArea(contour)
                
                if len(approx) == 4 and cv2.isContourConvex(approx) and area > min_area_calib:
                    x, y, w, h = cv2.boundingRect(approx)
                    # 0.2 and 0.05 are the proportions of the calibration rectangles in meters
                    rectangle_sizes.append(((0.20/h)*resolution[0], (0.05/w)*resolution[1]))
                    cv2.drawContours(cropped_frame, [approx], -1, RED, 2)
            
        return cropped_frame, rectangle_sizes

    def is_finished(self, frame):
        """ 
            Check if the grid is finished. Ie if the anchors are detected, the target is found, the
            obstacles are found and the initial position of the robot is obtained.

        Args:
            frame (np.ndarray): Original cropped image to analyze.

        Returns:
            bool, (np.ndarray) : True if the grid is finished, False otherwise and the cropped frame.
        """
        frame_pure = frame.copy()
        cropped_frame = frame.copy()
        
        if not self.anchors_found:
            self.detected_anchors = get_anchors(frame, self.show_dect_anchors, self.detected_anchors)
            for anchor in self.detected_anchors:
                cv2.circle(cropped_frame, anchor[:2], 5, GREEN, -1)
        
        if len(self.detected_anchors) == 4: # once the anchors are detected
            
            if not self.anchors_found:
                self.show_dect_anchors = False
                self.anchors_positions = sorted(self.detected_anchors, key=lambda p: (p[2], p[0]))
                top_left = self.anchors_positions[0][:2]
                bottom_right = self.anchors_positions[3][:2]
                
                # get the coordinates
                self.x_min, self.y_min = map(int, top_left)
                self.x_max, self.y_max = map(int, bottom_right)
                
                max_anchor_X, max_anchor_Y = self.anchors_positions[3][:2]
                max_anchor_X = max_anchor_X*resolution[0]
                max_anchor_Y = max_anchor_Y*resolution[1]
                
                self.grid_size = (int(max_anchor_Y), int(max_anchor_X))
                self.bottom_right = (self.grid_size[0], self.grid_size[1]) # x then y
                
                self.add_circles_grid(pastilles_colors)
                self.add_grid(obstacle_color)
                self.add_grid(robot_color)
                
                
            self.anchors_found = True

            
            # get the cropped frame
            cropped_frame = frame[self.y_min:self.y_max, self.x_min:self.x_max]
            cropped_pure = frame_pure[self.y_min:self.y_max, self.x_min:self.x_max]
            self.taille_crop = cropped_pure.shape[:2]
            
            if not self.enough_circles:
                detect_and_update_circles(cropped_pure, cropped_frame, self, self.anchors_positions)
            else:
                for _, vals in self.circles_positions.items():
                    cv2.circle(cropped_frame, (int(vals[0]), int(vals[1])), int(vals[2]), BLACK, 2)
                    cv2.circle(cropped_frame, (int(vals[0]), int(vals[1])), 2, BLACK, -1)
            
            # to detect the obstacles
            if not self.enough_obstacles:
                detect_obstacles(cropped_pure, cropped_frame, self, self.anchors_positions)
            else:
                self.draw_constant_polygons(cropped_frame)
            
            # to detect the robot
            if not self.init_robot:
                self.robot_position = detect_and_update_robot(cropped_pure, cropped_frame, self, self.anchors_positions)
                if self.robot_position is not None:
                    self.init_robot = True
                    cv2.circle(cropped_frame, self.robot_frame_position[0], 5, YELLOW, -1)
            else:
                if self.robot_position is not None:
                    cv2.circle(cropped_frame, self.robot_frame_position[0], 5, YELLOW, -1)
                

            # if we have everything
            if self.enough_circles and self.enough_obstacles and self.init_robot:
                self.found_everything = True
            

        return self.found_everything, cropped_frame
            
    def get_finished_grid(self):
        """
        Get the final grid with the obstacles, the robot and the target.

        Returns:
            (np.ndarray), (np.ndarray): the grid with the obstacles, the robot and the target colorized and the final grid with the index values
        """
        grids_to_send = pastilles_colors + [obstacle_color]
        self.grid_colors, self.final_grid = self.display_grid_with_colors(colors, grids_to_send)

    
    def get_updated_grid(self, reset=True):
        """
        Get the updated grid with the updated robot

        Args:
            reset (bool, optional): Wheter to keep displaying the previous robot positions or not. Defaults to True.

        Returns:
            np.ndarray: the updated grid with the robot colorized.
        """

        
        colors_used = get_sub_colors(colors, [robot_color])
        mask = (cv2.transpose(self.get_grid(robot_color)) == colors_used[robot_color][2])

        color_image = self.grid_colors.copy()
        color_image[mask] = colors_used[robot_color][3]
        
        if reset:
            reset_val = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=np.int8)
            self.set_grid(robot_color, reset_val)
        
        return color_image

    def get_trajectory(self, frame, color):
        """
            Plots the trajectory of the robot on the frame.

        Args:
            frame (np.ndarray): Original cropped image to analyze.
            color (tuple): the BGR color to use for the trajectory.
        """
        for pos in self.trajectory:
            cv2.circle(frame, pos, 2, color, -1)
    
    def get_robot_position(self, frame):
        """
        Get the position and angle of the robot on the frame.

        Args:
            frame (np.ndarray): Original cropped image to analyze.

        Returns:
            (np.ndarray), tuple : the cropped frame and the position and angle of the robot.
        """
        
        frame_pure = frame.copy()
        cropped_frame = frame[self.y_min:self.y_max, self.x_min:self.x_max]
        cropped_pure = frame_pure[self.y_min:self.y_max, self.x_min:self.x_max]

        self.robot_position = detect_and_update_robot(cropped_pure, cropped_frame, self, self.anchors_positions)
        return cropped_frame

    def add_grid(self, names):
        """
        Adds a grid with the specified name to the grids dictionary.

        Args:
            names (list): the names of the grids to add.

        Raises:
            ValueError: A grid with the specified name already exists.
        """

        if isinstance(names, str):
            names = [names]  # Convert to list if a single name is passed

        for name in names:
            if name in self.grids:
                raise ValueError(f"A grid with the name: '{name}' already exists.")
            self.grids[name] = np.zeros(self.grid_size, dtype=np.int8)

    def add_circles_grid(self, names):
        """
        Adds a grid for the circles with the specified name to the grids dictionary.
        Initializes the stock_circles_positions and number_circles_colors dictionaries.

        Args:
            names (list): the names of the grids to add.

        Raises:
            ValueError: A grid with the specified name already exists.
        """
        if isinstance(names, str):
            names = [names]
            
        for name in names:
            if name in self.grids:
                raise ValueError(f"A grid with the name: '{name}' already exists.")
            self.grids[name] = np.zeros(self.grid_size, dtype=np.int8)
            self.stock_circles_positions[name] = []
            self.number_circles_colors[name] = 0

    def get_grid(self, name):
        """
        Get the grid with the specified name.

        Args:
            name (string): Name of the grid to get.

        Raises:
            KeyError: If the grid with the specified name does not exist.

        Returns:
            np.ndarray : The grid with the specified name.
        """
        
        if name not in self.grids:
            raise KeyError(f"No grid found with the name '{name}'.")
        return self.grids[name]
    

    def update_stock_obstacles_positions(self, center, vertices, min_dist=10, threshold=10):
        """
        Updates the list of detected polygon positions, marking some as constant
        and sets `self.enough_obstacles` to True if all polygons are constant.

        Args:
            center (tuple): Center of newly detected polygon.
            vertices (ndarray): Vertices of the newly detected polygon.
            min_dist (float): Minimum distance for considering a polygon to be the same.
            threshold (int): Number of detections required to mark a polygon as constant.
        """
        updated = False

        for center_stock, data in self.stock_obstacles_positions.items():
            vertices_stock, count, is_constant = data
            
            # verify if the new polygon is the same as the existing one
            if np.linalg.norm(np.array(center) - np.array(center_stock)) < min_dist:
                # update the vertices if the polygon is the same
                for vertex in vertices:
                    if all(np.linalg.norm(vertex - vertex_stock) >= min_dist for vertex_stock in vertices_stock):
                        vertices_stock = np.vstack((vertices_stock, vertex))
                
                count += 1
                if count >= threshold:
                    is_constant = True
                
                self.stock_obstacles_positions[center_stock] = (vertices_stock, count, is_constant)
                updated = True
                break
        
        if not updated:
            # if the polygon is new, add it to the stock
            self.stock_obstacles_positions[center] = (vertices, 1, False)

        # verify if all polygons are constant
        if all(is_constant for _, (_, _, is_constant) in self.stock_obstacles_positions.items()):
            self.enough_obstacles = True
            self.constant_polygons = {center: vertices for center, (vertices, _, _) in self.stock_obstacles_positions.items()}

    def draw_constant_polygons(self, frame):
        """
        Draws constant polygons on the frame.
        """
        for center, vertices in self.constant_polygons.items():
            for vertex in vertices:
                cv2.circle(frame, tuple(vertex), 5, CYAN, -1)
            cv2.circle(frame, center, 5, CYAN, -1)
            cv2.polylines(frame, [vertices.astype(int)], isClosed=True, color=CYAN, thickness=2)
            
    def draw_constant_circles(self, frame):
        """
        Draw constant circles on the frame.
        """
        for _, vals in self.circles_frame.items():
            center_x, center_y, radius = vals
            cv2.circle(frame, (center_x, center_y), radius, CYAN, 2)
            cv2.circle(frame, (center_x, center_y), 2, CYAN, -1)

    
    def set_grid(self, grid_name, new_grid):
        """
        Replace the grid with the specified name with a new grid.
        
        Args:
            grid_name (str): Name of the grid to replace.
            new_grid (np.array): New grid to replace the existing grid.
        
        Raises:
            KeyError: If the grid with the specified name does not exist.
            ValueError: If the dimensions of the new grid do not match the current grid.
        """
        if grid_name not in self.grids:
            raise KeyError(f"The grid '{grid_name}' does not exist.")
        
        if new_grid.shape != self.grids[grid_name].shape:
            raise ValueError("The dimensions of the new grid do not match the current grid.")
        
        self.grids[grid_name] = new_grid
    
    def display_grid_with_colors(self, color_ranges, sub_colors):
        """
        Displays a colored grid where each square has a color depending on the value of the square.

        Args:
            colors_rgb (dict): Dictionary of colors with their associated indices.
            sub_colors (list): List of colors to be displayed (based on indices).
        
        Returns:
            np.ndarray, np.ndarray: Image representing the colored grid and the grid with the indices.
        """
        
        grid_size = self.grid_size
        color_image = np.zeros((grid_size[0], grid_size[1], 3), dtype=np.uint8)

        colors_used = get_sub_colors(color_ranges, sub_colors)

        # Add the grids together
        combined_grid = np.zeros(grid_size, dtype=np.int8)
        for grid in self.grids.values():
            combined_grid = np.minimum(combined_grid, grid)

        # apply the colors depending on the indices
        for _, color_values in colors_used.items():
            _, _, index, rgb = color_values
            mask = (combined_grid == index)
            color_image[mask] = rgb

        return cv2.transpose(color_image), cv2.transpose(combined_grid)
    
