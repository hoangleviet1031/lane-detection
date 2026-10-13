"""Tinh lane-center deviation: xe dang lech bao nhieu so voi tam 2 vach lane, quy doi ra met."""

LANE_WIDTH_METERS = 3.7  # do rong lane chuan (dung de uoc luong ty le pixel -> met)


def _x_at_bottom(points):
    """Lay x cua diem gan camera nhat (y lon nhat) trong 1 lane - vi tri dai dien de tinh deviation."""
    if not points:
        return None
    return max(points, key=lambda p: p[1])[0]


def compute_deviation(left_points, right_points, img_width):
    """Tra ve (offset_m, direction) hoac None neu thieu du lieu.
    offset_m duong = xe lech sang TRAI tam lane, am = lech sang PHAI (quy uoc: nhin tu tren xe xuong).
    direction: 'LEFT' | 'RIGHT' | 'CENTER'."""
    left_x = _x_at_bottom(left_points) if left_points else None
    right_x = _x_at_bottom(right_points) if right_points else None
    if left_x is None or right_x is None or right_x <= left_x:
        return None

    lane_center = (left_x + right_x) / 2
    img_center = img_width / 2
    lane_width_px = right_x - left_x
    meters_per_pixel = LANE_WIDTH_METERS / lane_width_px

    offset_m = (img_center - lane_center) * meters_per_pixel

    if abs(offset_m) < 0.05:
        direction = "CENTER"
    elif offset_m > 0:
        direction = "LEFT"
    else:
        direction = "RIGHT"
    return offset_m, direction
