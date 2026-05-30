def perturb_normalized_bbox(bbox, jitter_level, rng, min_size=1e-4):
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {bbox!r}")
    xmin, ymin, xmax, ymax = [float(value) for value in bbox]
    if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
        raise ValueError(f"Expected valid normalized bbox, got {bbox!r}")
    if jitter_level < 0:
        raise ValueError(f"jitter_level must be non-negative, got {jitter_level!r}")
    if jitter_level == 0:
        return list(bbox)

    jitter = float(jitter_level) / 100.0
    width = xmax - xmin
    height = ymax - ymin
    center_x = (xmin + xmax) / 2.0 + rng.uniform(-jitter, jitter) * width
    center_y = (ymin + ymax) / 2.0 + rng.uniform(-jitter, jitter) * height
    new_width = max(width * (1.0 + rng.uniform(-jitter, jitter)), min_size)
    new_height = max(height * (1.0 + rng.uniform(-jitter, jitter)), min_size)

    def clip_interval(center, size):
        low = max(0.0, center - size / 2.0)
        high = min(1.0, center + size / 2.0)
        if high - low >= min_size:
            return low, high
        if low <= 0.0:
            return 0.0, min(1.0, min_size)
        if high >= 1.0:
            return max(0.0, 1.0 - min_size), 1.0
        half = min_size / 2.0
        return max(0.0, center - half), min(1.0, center + half)

    new_xmin, new_xmax = clip_interval(center_x, new_width)
    new_ymin, new_ymax = clip_interval(center_y, new_height)
    return [new_xmin, new_ymin, new_xmax, new_ymax]
