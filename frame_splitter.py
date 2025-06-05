import re
import pandas as pd

def yield_lidar_frames(filename):
    """
    Generator that yields a pandas DataFrame for each LiDAR scan (frame).
    Each frame is split by the presence of an 's' marker at the start of a line.
    Each DataFrame has columns: ang, dist, qual (all floats/ints).
    """
    frame_lines = []
    pattern = re.compile(r'theta:\s*([\d.]+)\s*Dist:\s*([\d.]+)\s*Q:\s*(\d+)')
    with open(filename, 'r') as f:
        for line in f:
            # Check for 's' marker (start of new frame)
            if line.strip().startswith('S') or line.strip().startswith('s'):
                # If we have accumulated points, yield the previous frame
                if frame_lines:
                    df = _parse_frame_lines(frame_lines, pattern)
                    if not df.empty:
                        yield df
                    frame_lines = []
                # Always include the 's' line in the new frame
                frame_lines.append(line)
            else:
                frame_lines.append(line)
        # Yield the last frame if any
        if frame_lines:
            df = _parse_frame_lines(frame_lines, pattern)
            if not df.empty:
                yield df

def _parse_frame_lines(lines, pattern):
    """Helper to parse lines into a DataFrame of ang, dist, qual."""
    data = []
    for line in lines:
        match = pattern.search(line)
        if match:
            ang, dist, qual = match.groups()
            data.append([float(ang), float(dist), int(qual)])
    return pd.DataFrame(data, columns=['ang', 'dist', 'qual'])
