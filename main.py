import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import re
from frame_splitter import yield_lidar_frames
import time

# Function to print the latest DataFrame to debug.csv for inspection
def debug_csv(df):
    """Save the current DataFrame to debug.csv for inspection."""
    df.to_csv('debug.csv', index=False)
    print("[debug_csv] Saved snapshot to debug.csv")

#_______________________________________________

# Loop over each LiDAR frame and visualize in sequence
# =============================================
# LiDAR Frame-by-Frame Clustering Visualization
# =============================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from frame_splitter import yield_lidar_frames
from nearest_neighbor_tracker import NearestNeighborTracker

# --- Helper Functions ---
def cluster_and_centroids(df, eps=0.7, min_samples=10):    #adjust for resolution of lidar (epsilon = distance apart, min samples = min dots to make a cluster) (default to eps=.4, min=25)
    """Cluster points and extract centroids for each cluster."""
    r = df.dist / 1000
    th = np.deg2rad(df.ang)
    xy = np.c_[r * np.cos(th), r * np.sin(th)]
    # Limit to first 10,000 points
    xy, th, r = xy[:10000], th[:10000], r[:10000]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)
    df = df.iloc[:10000].copy()
    df['cluster'] = labels
    centroids = []
    for label in set(labels):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        mean_th = np.mean(th[idx])
        mean_r = np.mean(r[idx])
        centroids.append((mean_th, mean_r))
    return th, r, labels, centroids, df

def plot_frame(th, r, labels, tracked_objs, frame_idx, show_noise=True):
    plt.clf()
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 12)  # Fix radial axis from 0 to 12 meters
    # Plot clustered points (not noise)
    cluster_mask = (labels != -1)
    if np.any(cluster_mask):
        ax.scatter(th[cluster_mask], r[cluster_mask], c=labels[cluster_mask], s=2, cmap='tab20', label='Clustered')
    # Plot noise points in gray if enabled
    if show_noise:
        noise_mask = (labels == -1)
        if np.any(noise_mask):
            ax.scatter(th[noise_mask], r[noise_mask], c='gray', s=2, label='Noise')
    # Plot and label tracked centroids
    if tracked_objs:
        centroid_th, centroid_r = zip(*[(mean_th, mean_r) for _, mean_th, mean_r in tracked_objs])
        ax.scatter(centroid_th, centroid_r, c='red', s=20, marker='o', edgecolors='black', label='Centroids')
        for obj_id, mean_th, mean_r in tracked_objs:
            # Place ID label slightly offset from centroid
            ax.text(mean_th, mean_r + 0.3, str(obj_id), color='blue', fontsize=9, ha='center', va='bottom', fontweight='bold')
        ax.legend(loc='lower right')
    plt.title(f'LiDAR Frame {frame_idx}')
    plt.pause(0.1)   #(playback speed) 0.2 seconds per frame. is 5hz rotating disk. 

# --- Initialize Tracker ---
tracker = NearestNeighborTracker(max_distance=1.5)

# --- Main Loop ---
for frame_idx, df in enumerate(yield_lidar_frames('lidar_data/walk8_raw.csv')):
    if df.empty:
        continue
    th, r, labels, centroids, df = cluster_and_centroids(df)
    # Track objects: assign IDs to centroids
    tracked_objs = tracker.update(centroids)
    print(f"Frame {frame_idx}: {len(df)} points, {len(tracked_objs)} tracked objects")
    plot_frame(th, r, labels, tracked_objs, frame_idx)
plt.show()

# --- Debug CSV: Uncomment to save a snapshot of a frame ---
# debug_csv(df)  # Save the current state of df to debug.csv

