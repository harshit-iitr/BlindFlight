import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import heapq
import json
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "Final Model Gridded.pth"
MAPPING_PATH = "class_mapping (1).json"
TEST_IMG_DIR = Path("../../input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/test/images")
OUTPUT_CSV = "submission.csv"


CELL_SIZE = 64
MAX_ROWS = 20
MAX_COLS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COST_TABLES = {
    "Desert": {"Road": 1.2, "Start": 1.2, "End": 2.2, "Hazard": 3.7, "Cacti": 999.0, "Rocks": 999.0, "Obstacle": 999.0, "Unknown": 8.0},
    "Forest": {"Road": 1.5, "Start": 1.5, "End": 2.5, "Hazard": 2.8, "Tree": 999.0, "Obstacle": 999.0, "Unknown": 8.0},
    "Lab": {"Road": 1.0, "Start": 1.0, "End": 2.0, "Hazard": 3.0, "Wall": 999.0, "Plasma": 999.0, "Obstacle": 999.0, "Unknown": 8.0}
}

# ==========================================
# 2. VISION & GRID HELPERS
# ==========================================
def preprocess_for_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    no_texture = cv2.medianBlur(gray, 7)
    thresh = cv2.adaptiveThreshold(no_texture, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)
    return thresh

def get_intersections(img):
    thresh = preprocess_for_grid(img)
    scale = 25
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))
    mask_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    mask_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
    intersections = cv2.bitwise_and(mask_h, mask_v)
    intersections = cv2.dilate(intersections, np.ones((5,5)))
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(intersections)
    points = [centroids[i] for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 10]
    if not points: return np.array([])
    clustering = DBSCAN(eps=20, min_samples=1).fit(points)
    points = np.array(points)
    clean_points = [np.mean(points[clustering.labels_ == label], axis=0) for label in set(clustering.labels_)]
    return np.array(clean_points)

def sort_points_robust(points):
    y_clustering = DBSCAN(eps=25, min_samples=3).fit(points[:, 1].reshape(-1, 1))
    rows_dict = {}
    for pt, label in zip(points, y_clustering.labels_):
        if label == -1: continue
        if label not in rows_dict: rows_dict[label] = []
        rows_dict[label].append(pt)
    sorted_keys = sorted(rows_dict.keys(), key=lambda k: np.mean([p[1] for p in rows_dict[k]]))
    return [np.array(sorted(rows_dict[k], key=lambda p: p[0])) for k in sorted_keys]

def get_global_column_grid(grid_rows):
    all_x = [pt[0] for row in grid_rows for pt in row]
    if not all_x: return np.array([])
    all_x = np.array(all_x).reshape(-1, 1)
    clustering = DBSCAN(eps=15, min_samples=1).fit(all_x)
    col_centers = [np.mean(all_x[clustering.labels_ == label]) for label in set(clustering.labels_) if label != -1]
    return np.array(sorted(col_centers))

def extract_grid_data(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return None, None, "Load Error"

    grid_tensor = np.zeros((MAX_ROWS, MAX_COLS, CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8)
    mask_tensor = np.zeros((MAX_ROWS, MAX_COLS), dtype=np.uint8)

    try:
        points = get_intersections(img)
        if len(points) < 40: return grid_tensor, mask_tensor, "Fallback: Low Pts"
        grid_rows = sort_points_robust(points)
        if len(grid_rows) < 4: return grid_tensor, mask_tensor, "Fallback: Few Rows"
        col_centers = get_global_column_grid(grid_rows)
        if len(col_centers) < 2: return grid_tensor, mask_tensor, "Fallback: No Cols"

        for r in range(min(len(grid_rows) - 1, MAX_ROWS)):
            row_top = grid_rows[r]
            row_btm = grid_rows[r+1]
            for pt_top in row_top:
                neighbors_tr = [p for p in row_top if p[0] > pt_top[0]]
                if not neighbors_tr: continue
                pt_tr = min(neighbors_tr, key=lambda p: p[0])
                if abs(pt_tr[0] - pt_top[0]) > 100: continue
                candidates_bl = [p for p in row_btm if abs(p[0] - pt_top[0]) < 70]
                if not candidates_bl: continue
                pt_bl = min(candidates_bl, key=lambda p: abs(p[0] - pt_top[0]))
                candidates_br = [p for p in row_btm if abs(p[0] - pt_tr[0]) < 70]
                if not candidates_br: continue
                pt_br = min(candidates_br, key=lambda p: abs(p[0] - pt_tr[0]))

                src = np.array([pt_top, pt_tr, pt_br, pt_bl], dtype="float32")
                dst = np.array([[0,0], [CELL_SIZE,0], [CELL_SIZE,CELL_SIZE], [0,CELL_SIZE]], dtype="float32")
                M = cv2.getPerspectiveTransform(src, dst)
                warped = cv2.warpPerspective(img, M, (CELL_SIZE, CELL_SIZE))

                diffs = np.abs(col_centers - pt_top[0])
                c_idx = np.argmin(diffs)
                if c_idx < MAX_COLS:
                    grid_tensor[r, c_idx] = warped
                    mask_tensor[r, c_idx] = 1
        return grid_tensor, mask_tensor, "Active: Gridded"
    except: return grid_tensor, mask_tensor, "Error"

# ==========================================
# 3. MODEL LOADING (FIXED FOR STANDARD RESNET)
# ==========================================
with open(MAPPING_PATH, 'r') as f:
    idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

print("🚀 Loading V13 Model (Standard Architecture)...")

# 1. Base ResNet
model = models.resnet18(pretrained=False)

# --- FIX: DISABLED HACK TO MATCH SAVED FILE ---
# The saved checkpoint has a 7x7 conv1, so we must use standard definition.
# model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# model.maxpool = nn.Identity()
# ----------------------------------------------

model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))

try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
except RuntimeError as e:
    print("\n❌ LOAD ERROR! Model mismatch.")
    print(f"Error details: {e}\n")
    raise e

if torch.cuda.device_count() > 1:
    print(f"🔥 Twin-Turbo Active: {torch.cuda.device_count()} GPUs for Inference")
    model = nn.DataParallel(model)

model = model.to(device)
model.eval()

# We still feed 128x128 images. ResNet will just downsample them normally.
transform_pipe = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. PATH FORMATTER (LOWERCASE LDRU)
# ==========================================
def path_coords_to_string(path_list):
    if not path_list or len(path_list) < 2:
        return None

    dirs = []
    for i in range(len(path_list) - 1):
        r1, c1 = path_list[i]
        r2, c2 = path_list[i+1]

        if r2 > r1: dirs.append("d")
        elif r2 < r1: dirs.append("u")
        elif c2 > c1: dirs.append("r")
        elif c2 < c1: dirs.append("l")

    return "".join(dirs)

def clean_image_id(filename):
    """ '0001.png' -> 1 """
    s = str(filename).lower()
    s = s.replace(".png", "").replace(".jpg", "")
    try:
        return int(s)
    except:
        return s

# ==========================================
# 5. SOLVER
# ==========================================
def solve_image(image_path):
    # 1. Slice
    grid_tensor, mask_tensor, status = extract_grid_data(image_path)
    if "Active" not in status: return None

    rows, cols = mask_tensor.shape
    batch_tensors = []
    coords = []

    # 2. Batch
    for r in range(rows):
        for c in range(cols):
            if mask_tensor[r, c] == 1:
                rgb_cell = cv2.cvtColor(grid_tensor[r, c], cv2.COLOR_BGR2RGB)
                batch_tensors.append(transform_pipe(rgb_cell))
                coords.append((r, c))

    if not batch_tensors: return None

    # 3. Predict
    batch_stack = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        outputs = model(batch_stack)
        _, preds = torch.max(outputs, 1)

    # 4. Map
    terrain_map = np.full((rows, cols), "Unknown", dtype=object)
    counts = {"Desert": 0, "Forest": 0, "Lab": 0}

    for i, (r, c) in enumerate(coords):
        class_name = idx_to_class[preds[i].item()]
        terrain_map[r, c] = class_name

        biome = class_name.split("_")[0] if "_" in class_name else "Lab"
        weight = 3 if any(x in class_name for x in ["Cacti", "Tree", "Plasma", "Wall", "Rocks"]) else 1
        if "Start" in class_name or "End" in class_name: weight = 5
        counts[biome] = counts.get(biome, 0) + weight

    dominant_biome = max(counts, key=counts.get) if counts else "Lab"
    costs = COST_TABLES.get(dominant_biome, COST_TABLES["Lab"])

    # 5. A*
    start_pos, end_pos = None, None
    for r in range(rows):
        for c in range(cols):
            if "Start" in terrain_map[r, c]: start_pos = (r, c)
            if "End" in terrain_map[r, c]: end_pos = (r, c)

    if not start_pos or not end_pos: return None

    pq = [(0, start_pos)]
    cost_so_far = {start_pos: 0}
    came_from = {}

    while pq:
        curr_cost, curr = heapq.heappop(pq)
        if curr == end_pos:
            path = []
            while curr != start_pos:
                path.append(curr)
                curr = came_from[curr]
            path.append(start_pos)
            path.reverse()
            return path_coords_to_string(path)

        r, c = curr
        for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if 0 <= nr < rows and 0 <= nc < cols:
                label = terrain_map[nr, nc]
                cell_type = label.split("_")[1] if "_" in label else label
                step_cost = costs.get(cell_type, 999.0)

                new_cost = cost_so_far[curr] + step_cost
                if step_cost < 100:
                    if (nr, nc) not in cost_so_far or new_cost < cost_so_far[(nr, nc)]:
                        cost_so_far[(nr, nc)] = new_cost
                        priority = new_cost + abs(nr-end_pos[0]) + abs(nc-end_pos[1])
                        heapq.heappush(pq, (priority, (nr, nc)))
                        came_from[(nr, nc)] = curr

    return None

# ==========================================
# 6. RUN
# ==========================================
test_files = sorted(list(TEST_IMG_DIR.glob("*.png")))
data = []

print(f"Generating Submission for {len(test_files)} images...")

for img_path in tqdm(test_files):
    path_str = solve_image(img_path)
    data.append({
        "image_id": clean_image_id(img_path.name),
        "path": path_str
    })
    

df = pd.DataFrame(data)
df = df.sort_values(by="image_id")
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Submission saved to {OUTPUT_CSV}")
print(df.head())







