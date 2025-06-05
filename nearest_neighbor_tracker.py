import numpy as np

class NearestNeighborTracker:
    """
    Tracks objects (centroids) across sequential frames using nearest-neighbor association.
    Assigns persistent IDs to each detected object.
    """
    def __init__(self, max_distance=0.8):   #change this parameter: 0.8 meters is considered same object. 
        self.max_distance = max_distance  # Max distance (meters) to consider same object
        self.next_id = 0
        self.objects = {}  # id: (mean_th, mean_r)
        self.active_ids = set()

    def update(self, centroids):
        """
        Associate new centroids with existing IDs, or assign new IDs.
        Args:
            centroids: list of (mean_th, mean_r) tuples for current frame
        Returns:
            List of (id, mean_th, mean_r) for current frame
        """
        assigned = set()
        results = []
        centroids_np = np.array(centroids) if centroids else np.empty((0,2))
        prev_ids = list(self.objects.keys())
        prev_centroids = np.array([self.objects[i] for i in prev_ids]) if prev_ids else np.empty((0,2))
        self.active_ids = set()
        if len(prev_centroids) > 0 and len(centroids_np) > 0:
            # Compute pairwise distances
            dists = np.linalg.norm(prev_centroids[:,None,:] - centroids_np[None,:,:], axis=2)
            for prev_idx, prev_id in enumerate(prev_ids):
                # Find closest centroid not yet assigned
                min_idx = np.argmin(dists[prev_idx])
                if dists[prev_idx, min_idx] < self.max_distance and min_idx not in assigned:
                    # Assign to this ID
                    mean_th, mean_r = centroids[min_idx]
                    results.append((prev_id, mean_th, mean_r))
                    self.objects[prev_id] = (mean_th, mean_r)
                    self.active_ids.add(prev_id)
                    assigned.add(min_idx)
        # Assign new IDs to unassigned centroids
        for idx, (mean_th, mean_r) in enumerate(centroids):
            if idx not in assigned:
                obj_id = self.next_id
                self.next_id += 1
                self.objects[obj_id] = (mean_th, mean_r)
                self.active_ids.add(obj_id)
                results.append((obj_id, mean_th, mean_r))
        # Optionally: remove lost objects (not seen in this frame)
        lost_ids = set(self.objects.keys()) - self.active_ids
        for lost_id in lost_ids:
            del self.objects[lost_id]
        return results
