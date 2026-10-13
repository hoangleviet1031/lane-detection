"""Parse annotation CULane (.lines.txt) thanh format diem lane dung chung cho project.

Cau truc CULane goc:
    driver_23_30frame/05151649_0422.MP4/00000.jpg
    driver_23_30frame/05151649_0422.MP4/00000.lines.txt
Moi dong trong .lines.txt la 1 lane: "x1 y1 x2 y2 x3 y3 ... xn yn" (cach nhau bang space).
So dong = so lane co trong anh (toi da 4, thuong la 2 lane gan xe nhat quan trong nhat).
"""


def parse_lines_txt(path):
    """Doc 1 file .lines.txt, tra ve list cac lane, moi lane la list (x, y)."""
    lanes = []
    with open(path, "r") as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            if len(values) < 4:
                continue  # lane qua it diem, bo qua
            points = list(zip(values[0::2], values[1::2]))
            lanes.append(points)
    return lanes


def load_sample(image_path, annotation_path=None):
    """Tra ve (image_path, lanes). Neu khong co annotation_path, suy ra tu image_path
    bang cach doi duoi .jpg -> .lines.txt (dung convention cua CULane)."""
    if annotation_path is None:
        annotation_path = image_path.rsplit(".", 1)[0] + ".lines.txt"
    lanes = parse_lines_txt(annotation_path)
    return image_path, lanes


def load_list_file(list_path, root_dir=""):
    """Doc 1 file list cua CULane (vd list/test_split/test8_night.txt), moi dong la
    1 duong dan anh (co the co leading '/'). Tra ve list full path toi anh."""
    paths = []
    with open(list_path, "r") as f:
        for line in f:
            rel = line.strip().split()[0].lstrip("/")
            if rel:
                paths.append(root_dir.rstrip("/") + "/" + rel if root_dir else rel)
    return paths
