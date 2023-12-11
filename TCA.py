#!/usr/bin/env python
# coding: utf-8

# # Transfroming the Dicom Scans inton PNGs and collecting them in a folder

# In[6]:


import os
import shutil
import pydicom
from PIL import Image
from pydicom.errors import InvalidDicomError

def extract_number(filename):
    number_part = ''.join(filter(str.isdigit, filename))
    return int(number_part) if number_part else 0

def dicom_folder_to_png(dicom_folder):
    output_folder = "PNGS"

    # Check if the output folder exists and delete it
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Create a new empty output folder
    os.makedirs(output_folder, exist_ok=True)

    dicom_files = sorted([f for f in os.listdir(dicom_folder)], key=extract_number)

    for idx, dicom_file in enumerate(dicom_files):
        dicom_path = os.path.join(dicom_folder, dicom_file)
        print(f"Converting {dicom_path}...")

        try:
            with pydicom.dcmread(dicom_path) as ds:
                pixel_data = ds.pixel_array
                normalized_pixel_data = ((pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min()) * 255).astype('uint8')
                img = Image.fromarray(normalized_pixel_data, mode='L')
                dpi = 512
                png_file = os.path.join(output_folder, f"{idx + 1:03d}.png")
                img.save(png_file, "PNG", dpi=(dpi, dpi))
                print(f"Saved {png_file}")

        except InvalidDicomError:
            print(f"Skipped non-DICOM file: {dicom_file}")

dicom_folder = "AXIAL"
dicom_folder_to_png(dicom_folder)


# In[4]:


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def find_circles(image_path):
    # Read image.
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    brightness_factor = 2  # Adjust this value as needed.

    # Multiply the pixel values by the brightness factor.
    enhanced_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

    # Convert the enhanced image to grayscale.
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    # Binarize the grayscale image.
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Fill in the shapes (contours) in the binary image.
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours in the binary image.
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store valid circles
    valid_circles = []

    for contour in contours:
        # Calculate the area and perimeter for each contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Check if the perimeter is zero (single point) and skip this contour
        if perimeter == 0:
            continue

        # Calculate the circularity for each contour
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Define circularity and area thresholds to filter circles
        circularity_threshold = 0.8  # Adjust this threshold as needed
        min_area = 100  # Adjust this threshold as needed
        max_area = 2000

        # Check if the contour meets the circularity and area criteria
        if circularity >= circularity_threshold and max_area>= area >= min_area:
            # Fit a circle to the contour and obtain its center (a, b) and radius (r)
            (a, b), r = cv2.minEnclosingCircle(contour)
            a, b, r = int(a), int(b), int(r)
            valid_circles.append((a, b, r))
            print(f"Circle: Center ({a}, {b}), Radius {r}, Circularity {circularity}")

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (255, 100, 255), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (255, 0, 0), 2)

    if valid_circles:
        # Display the image with circles drawn
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB for displaying
        plt.show()
        return valid_circles
    else:
        print("No good-fit circles found!")
        return None

# Folder containing PNG files
pngs_folder = 'PNGS'

# List PNG files in the folder
png_files = [f for f in os.listdir(pngs_folder) if f.lower().endswith('.png')]

# Process each PNG file
for png_file in png_files:
    png_path = os.path.join(pngs_folder, png_file)
    print(f"Processing: {png_path}")
    find_circles(png_path)


# In[16]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_number(filename):
    number_part = ''.join(filter(str.isdigit, filename))
    return int(number_part) if number_part else 0

def dicom_folder_to_png(dicom_folder):
    output_folder = "PNGS"

    # Check if the output folder exists and delete it
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Create a new empty output folder
    os.makedirs(output_folder, exist_ok=True)

    dicom_files = sorted([f for f in os.listdir(dicom_folder)], key=extract_number)

    for idx, dicom_file in enumerate(dicom_files):
        dicom_path = os.path.join(dicom_folder, dicom_file)
        print(f"Converting {dicom_path}...")

        try:
            with pydicom.dcmread(dicom_path) as ds:
                pixel_data = ds.pixel_array
                normalized_pixel_data = ((pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min()) * 255).astype('uint8')
                img = Image.fromarray(normalized_pixel_data, mode='L')
                dpi = 512
                png_file = os.path.join(output_folder, f"{idx + 1:03d}.png")
                img.save(png_file, "PNG", dpi=(dpi, dpi))
                print(f"Saved {png_file}")

        except InvalidDicomError:
            print(f"Skipped non-DICOM file: {dicom_file}")

def find_circles(image_path, display=False):
    # Existing code to find circles
    # Read image.
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    brightness_factor = 2  # Adjust this value as needed.

    # Multiply the pixel values by the brightness factor.
    enhanced_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

    # Convert the enhanced image to grayscale.
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    # Binarize the grayscale image.
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Fill in the shapes (contours) in the binary image.
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours in the binary image.
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store valid circles
    valid_circles = []

    for contour in contours:
        # Calculate the area and perimeter for each contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Check if the perimeter is zero (single point) and skip this contour
        if perimeter == 0:
            continue

        # Calculate the circularity for each contour
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Define circularity and area thresholds to filter circles
        circularity_threshold = 0.8  # Adjust this threshold as needed
        min_area = 100  # Adjust this threshold as needed
        max_area = 2000

        # Check if the contour meets the circularity and area criteria
        if circularity >= circularity_threshold and max_area>= area >= min_area:
            # Fit a circle to the contour and obtain its center (a, b) and radius (r)
            (a, b), r = cv2.minEnclosingCircle(contour)
            a, b, r = int(a), int(b), int(r)
            valid_circles.append((a, b, r))
            print(f"Circle: Center ({a}, {b}), Radius {r}, Circularity {circularity}")

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (255, 100, 255), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (255, 0, 0), 2)

    if valid_circles:
        # Display the image with circles drawn
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB for displaying
        plt.show()
        return valid_circles
    else:
        print("No good-fit circles found!")
        return None


    if display:
        # Display the image with circles drawn
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB for displaying
        plt.show()

    return valid_circles

def cluster_trackers(png_files, threshold=10, display=False):
    all_circles = []

    for idx, png_file in enumerate(png_files):
        png_path = os.path.join(pngs_folder, png_file)
        circles = find_circles(png_path, display=display)

        # Handle the case when no circles are found
        if circles is not None:
            for circle in circles:
                all_circles.append((circle[0], circle[1], idx))  # x, y, z

    # Cluster circles based on proximity
    clusters = {}
    for circle in all_circles:
        added_to_cluster = False
        for cluster_id, cluster_circles in clusters.items():
            if all(np.linalg.norm(np.array(circle[:2]) - np.array(c[:2])) < threshold for c in cluster_circles):
                cluster_circles.append(circle)
                added_to_cluster = True
                break

        if not added_to_cluster:
            clusters[len(clusters)] = [circle]

    # Filter clusters by continuity in Z-axis and calculate average coordinates
    final_clusters = {}
    for cluster_id, cluster_circles in clusters.items():
        if len(cluster_circles) >= 6:  # Assuming 6 consecutive images as a threshold
            avg_x = np.mean([c[0] for c in cluster_circles])
            avg_y = np.mean([c[1] for c in cluster_circles])
            avg_z = np.mean([c[2] for c in cluster_circles])
            final_clusters[cluster_id] = (avg_x, avg_y, avg_z)

    return final_clusters

# Folder containing PNG files
pngs_folder = 'PNGS'

# List PNG files in the folder
png_files = [f for f in os.listdir(pngs_folder) if f.lower().endswith('.png')]

# Cluster the trackers
tracker_clusters = cluster_trackers(png_files, display=True)  # Set display to False to hide results

for cluster_id, coordinates in tracker_clusters.items():
    print(f"Cluster {cluster_id}: Average Coordinates: {coordinates}")


# In[17]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_number(filename):
    number_part = ''.join(filter(str.isdigit, filename))
    return int(number_part) if number_part else 0

def dicom_folder_to_png(dicom_folder):
    output_folder = "PNGS"

    # Check if the output folder exists and delete it
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Create a new empty output folder
    os.makedirs(output_folder, exist_ok=True)

    dicom_files = sorted([f for f in os.listdir(dicom_folder)], key=extract_number)

    for idx, dicom_file in enumerate(dicom_files):
        dicom_path = os.path.join(dicom_folder, dicom_file)
        print(f"Converting {dicom_path}...")

        try:
            with pydicom.dcmread(dicom_path) as ds:
                pixel_data = ds.pixel_array
                normalized_pixel_data = ((pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min()) * 255).astype('uint8')
                img = Image.fromarray(normalized_pixel_data, mode='L')
                dpi = 512
                png_file = os.path.join(output_folder, f"{idx + 1:03d}.png")
                img.save(png_file, "PNG", dpi=(dpi, dpi))
                print(f"Saved {png_file}")

        except InvalidDicomError:
            print(f"Skipped non-DICOM file: {dicom_file}")

def find_circles(image_path, display=False):
    # Existing code to find circles
    # Read image.
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    brightness_factor = 2  # Adjust this value as needed.

    # Multiply the pixel values by the brightness factor.
    enhanced_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

    # Convert the enhanced image to grayscale.
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    # Binarize the grayscale image.
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Fill in the shapes (contours) in the binary image.
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours in the binary image.
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store valid circles
    valid_circles = []

    for contour in contours:
        # Calculate the area and perimeter for each contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Check if the perimeter is zero (single point) and skip this contour
        if perimeter == 0:
            continue

        # Calculate the circularity for each contour
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Define circularity and area thresholds to filter circles
        circularity_threshold = 0.8  # Adjust this threshold as needed
        min_area = 100  # Adjust this threshold as needed
        max_area = 2000

        # Check if the contour meets the circularity and area criteria
        if circularity >= circularity_threshold and max_area>= area >= min_area:
            # Fit a circle to the contour and obtain its center (a, b) and radius (r)
            (a, b), r = cv2.minEnclosingCircle(contour)
            a, b, r = int(a), int(b), int(r)
            valid_circles.append((a, b, r))
            print(f"Circle: Center ({a}, {b}), Radius {r}, Circularity {circularity}")

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (255, 100, 255), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (255, 0, 0), 2)

    if valid_circles:
        # Display the image with circles drawn
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB for displaying
        plt.show()
        return valid_circles
    else:
        print("No good-fit circles found!")
        return None


    if display:
        # Display the image with circles drawn
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert to RGB for displaying
        plt.show()

    return valid_circles

def cluster_trackers(png_files, threshold=10, display=False):
    all_circles = []
    image_dict = {}  # Dictionary to store images with circles drawn

    for idx, png_file in enumerate(png_files):
        png_path = os.path.join(pngs_folder, png_file)
        circles = find_circles(png_path, display=display)

        # Handle the case when no circles are found
        if circles is not None:
            for circle in circles:
                all_circles.append((circle[0], circle[1], idx))  # x, y, z
            image_dict[idx] = cv2.imread(png_path, cv2.IMREAD_COLOR)  # Store the original image

    # Cluster circles based on proximity
    clusters = {}
    for circle in all_circles:
        added_to_cluster = False
        for cluster_id, cluster_circles in clusters.items():
            if all(np.linalg.norm(np.array(circle[:2]) - np.array(c[:2])) < threshold for c in cluster_circles):
                cluster_circles.append(circle)
                added_to_cluster = True
                break

        if not added_to_cluster:
            clusters[len(clusters)] = [circle]

    # Filter clusters by continuity in Z-axis and calculate average coordinates
    final_clusters = {}
    for cluster_id, cluster_circles in clusters.items():
        if len(cluster_circles) >= 6:  # Assuming 6 consecutive images as a threshold
            avg_x = np.mean([c[0] for c in cluster_circles])
            avg_y = np.mean([c[1] for c in cluster_circles])
            avg_z = np.mean([c[2] for c in cluster_circles])
            final_clusters[cluster_id] = (avg_x, avg_y, avg_z)

            # Create a folder for the cluster and save images with labeled circles
            cluster_folder = os.path.join(pngs_folder, f"Cluster_{cluster_id}")
            os.makedirs(cluster_folder, exist_ok=True)
            for x, y, z in cluster_circles:
                image = image_dict.get(z)
                if image is not None:
                    labeled_image = image.copy()
                    cv2.circle(labeled_image, (x, y), 10, (0, 255, 0), 2)  # Draw circle
                    cv2.putText(labeled_image, f"{x},{y}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label coordinate
                    cv2.imwrite(os.path.join(cluster_folder, f"image_{z}.png"), labeled_image)  # Save image

    return final_clusters

# Folder containing PNG files
pngs_folder = 'PNGS'

# List PNG files in the folder
png_files = [f for f in os.listdir(pngs_folder) if f.lower().endswith('.png')]

# Cluster the trackers
tracker_clusters = cluster_trackers(png_files, display=True)  # Set display to False to hide results

for cluster_id, coordinates in tracker_clusters.items():
    print(f"Cluster {cluster_id}: Average Coordinates: {coordinates}")


# Folder containing PNG files
pngs_folder = 'PNGS'

# List PNG files in the folder
png_files = [f for f in os.listdir(pngs_folder) if f.lower().endswith('.png')]

# Cluster the trackers
tracker_clusters = cluster_trackers(png_files, display=True)  # Set display to False to hide results

for cluster_id, coordinates in tracker_clusters.items():
    print(f"Cluster {cluster_id}: Average Coordinates: {coordinates}")


# In[ ]:




