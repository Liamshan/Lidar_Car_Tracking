import numpy as np

class NearestNeighborTracker:
    """
    Tracks objects (centroids) across sequential frames using nearest-neighbor association.
    Assigns persistent IDs to each detected object. Adds a persistence buffer for lost tracks.
    """
    def __init__(self, max_distance=0.8, persistence=2):
        self.max_distance = max_distance  # Max distance (meters) to consider same object
        self.persistence = persistence    # Number of frames to keep lost objects
        self.next_id = 0
        self.objects = {}  # id: (mean_th, mean_r)
        self.active_ids = set()
        self.missed_frames = {}  # id: number of missed frames
        self.lost_objects = {}   # id: (mean_th, mean_r) for recently lost

    def update(self, centroids):
        """
        Associate new centroids with existing IDs, or assign new IDs.
        Adds persistence: objects are only deleted after missing for self.persistence frames.
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
        matched_prev_ids = set()

        # 1. Match current centroids to active objects
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
                    matched_prev_ids.add(prev_id)
                    self.missed_frames[prev_id] = 0  # Reset missed frames

        # 2. Try to match unassigned centroids to recently lost objects (within buffer)
        lost_ids = [oid for oid in self.lost_objects.keys() if self.missed_frames.get(oid, 0) <= self.persistence]
        lost_centroids = np.array([self.lost_objects[oid] for oid in lost_ids]) if lost_ids else np.empty((0,2))
        if len(lost_centroids) > 0 and len(centroids_np) > 0:
            dists = np.linalg.norm(lost_centroids[:,None,:] - centroids_np[None,:,:], axis=2)
            for lost_idx, lost_id in enumerate(lost_ids):
                min_idx = np.argmin(dists[lost_idx])
                if dists[lost_idx, min_idx] < self.max_distance and min_idx not in assigned:
                    mean_th, mean_r = centroids[min_idx]
                    results.append((lost_id, mean_th, mean_r))
                    self.objects[lost_id] = (mean_th, mean_r)
                    self.active_ids.add(lost_id)
                    assigned.add(min_idx)
                    self.missed_frames[lost_id] = 0
                    if lost_id in self.lost_objects:
                        del self.lost_objects[lost_id]

        # 3. Assign new IDs to remaining unassigned centroids
        for idx, (mean_th, mean_r) in enumerate(centroids):
            if idx not in assigned:
                obj_id = self.next_id
                self.next_id += 1
                self.objects[obj_id] = (mean_th, mean_r)
                self.active_ids.add(obj_id)
                results.append((obj_id, mean_th, mean_r))
                self.missed_frames[obj_id] = 0

        # 4. Update missed frame counts and clean up lost objects
        to_remove = []
        for obj_id in list(self.objects.keys()):
            if obj_id not in self.active_ids:
                self.missed_frames[obj_id] = self.missed_frames.get(obj_id, 0) + 1
                if self.missed_frames[obj_id] > self.persistence:
                    # Remove after buffer exceeded
                    to_remove.append(obj_id)
                    if obj_id in self.lost_objects:
                        del self.lost_objects[obj_id]
                else:
                    # Keep as lost object
                    self.lost_objects[obj_id] = self.objects[obj_id]
            else:
                self.missed_frames[obj_id] = 0
                if obj_id in self.lost_objects:
                    del self.lost_objects[obj_id]
        for obj_id in to_remove:
            if obj_id in self.objects:
                del self.objects[obj_id]
            if obj_id in self.missed_frames:
                del self.missed_frames[obj_id]
        return results
