"""Render camera, temporal occupancy and trajectory assets without Open3D.

The input occupancy file is the native ``multi_gpu_test_temporal`` output
with shape [1, 4, 200, 200, 16].  Camera paths and frame identity are read
from a metadata JSON.  An optional NPZ adds predicted and ground-truth ego
trajectories.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


LABELS = [
    "other", "barrier", "bicycle", "bus", "car",
    "construction vehicle", "motorcycle", "pedestrian", "traffic cone",
    "trailer", "truck", "driveable surface", "other flat", "sidewalk",
    "terrain", "manmade", "vegetation", "free",
]

# Same semantic colors as tools/visualization/vis_tool.py, expressed in RGB.
PALETTE_RGB = np.array([
    [140, 70, 70], [100, 158, 40], [0, 0, 230], [20, 85, 125],
    [220, 20, 60], [60, 60, 255], [255, 140, 0], [200, 150, 70],
    [255, 61, 99], [60, 155, 40], [222, 184, 135], [100, 100, 100],
    [165, 42, 42], [60, 60, 60], [75, 0, 75], [255, 158, 0],
    [0, 175, 0], [245, 247, 250],
], dtype=np.uint8)

CAMERA_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--trajectory")
    parser.add_argument("--output-dir", default="docs/assets")
    parser.add_argument("--prefix", default="model_prediction")
    return parser.parse_args()


def resolve_project_path(project_root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    text = str(path)
    if text.startswith("./"):
        text = text[2:]
    return project_root / text


def top_projection(volume):
    """Return the highest occupied label for each horizontal cell."""
    occupied = (volume != 17) & (volume != 255)
    any_occupied = occupied.any(axis=2)
    highest_from_top = np.argmax(occupied[:, :, ::-1], axis=2)
    z_index = volume.shape[2] - 1 - highest_from_top
    projected = np.full(volume.shape[:2], 17, dtype=np.uint8)
    xs, ys = np.where(any_occupied)
    projected[xs, ys] = volume[xs, ys, z_index[xs, ys]].astype(np.uint8)
    # Ego x points upward and positive y points to the left in the image.
    return projected[::-1, ::-1]


def draw_ego(image, center, scale=1.0):
    cx, cy = center
    half_w = max(5, int(9 * scale))
    half_h = max(9, int(18 * scale))
    body = np.array([
        [cx - half_w, cy + half_h], [cx + half_w, cy + half_h],
        [cx + half_w, cy - half_h + 5], [cx + half_w - 4, cy - half_h],
        [cx - half_w + 4, cy - half_h], [cx - half_w, cy - half_h + 5],
    ], np.int32)
    cv2.fillConvexPoly(image, body, (38, 45, 55), cv2.LINE_AA)
    cv2.polylines(image, [body], True, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(image, (cx, cy - half_h - 2), (cx, cy - half_h - 13),
             (38, 45, 55), 3, cv2.LINE_AA)


def occupancy_panel(volume, title, panel_size=560):
    projected = top_projection(volume)
    rgb = PALETTE_RGB[np.clip(projected, 0, len(PALETTE_RGB) - 1)]
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST)
    for meter in range(-30, 40, 10):
        pixel = int(round((40 - meter) / 80 * panel_size))
        if 0 <= pixel < panel_size:
            cv2.line(image, (0, pixel), (panel_size, pixel), (210, 217, 225), 1)
            cv2.line(image, (pixel, 0), (pixel, panel_size), (210, 217, 225), 1)
    draw_ego(image, (panel_size // 2, panel_size // 2), scale=1.2)
    header = np.full((74, panel_size, 3), (247, 249, 252), np.uint8)
    cv2.putText(header, title, (18, 47), cv2.FONT_HERSHEY_DUPLEX,
                1.0, (30, 40, 52), 2, cv2.LINE_AA)
    return np.vstack([header, image])


def add_legend(canvas):
    present = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
    legend = np.full((170, canvas.shape[1], 3), (247, 249, 252), np.uint8)
    columns = 5
    cell_width = canvas.shape[1] // columns
    for index, label_id in enumerate(present):
        row, col = divmod(index, columns)
        x, y = col * cell_width + 18, row * 48 + 24
        color = tuple(int(v) for v in PALETTE_RGB[label_id][::-1])
        cv2.rectangle(legend, (x, y), (x + 24, y + 24), color, -1)
        cv2.putText(legend, LABELS[label_id], (x + 34, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (48, 58, 70), 1,
                    cv2.LINE_AA)
    return np.vstack([canvas, legend])


def render_occupancy(prediction, output_path, metadata):
    data = np.load(prediction)
    if data.ndim == 6 and data.shape[0] == 1:
        data = data[0]
    if data.ndim == 5 and data.shape[0] == 1:
        data = data[0]
    if data.shape[:2] != (4, 200) or data.shape[2:] != (200, 16):
        raise ValueError(f"unexpected occupancy shape: {data.shape}")
    titles = ["Current (0 s)", "Future (+2 s)", "Future (+4 s)", "Future (+6 s)"]
    panels = [occupancy_panel(data[i], titles[i]) for i in range(4)]
    gap = np.full((panels[0].shape[0], 16, 3), (247, 249, 252), np.uint8)
    canvas = panels[0]
    for panel in panels[1:]:
        canvas = np.hstack([canvas, gap, panel])
    title = np.full((115, canvas.shape[1], 3), (247, 249, 252), np.uint8)
    heading = "TEMPORAL SEMANTIC OCCUPANCY PREDICTION"
    subtitle = f"epoch 4 | {metadata['scene_name']} frame {metadata['frame_idx']} | 80 m x 80 m"
    cv2.putText(title, heading, (22, 48), cv2.FONT_HERSHEY_DUPLEX,
                1.15, (26, 38, 52), 2, cv2.LINE_AA)
    cv2.putText(title, subtitle, (24, 86), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (90, 103, 117), 2, cv2.LINE_AA)
    canvas = add_legend(np.vstack([title, canvas]))
    cv2.imwrite(str(output_path), canvas)


def crop_to_ratio(image, width, height):
    target = width / height
    h, w = image.shape[:2]
    if w / h > target:
        new_w = int(round(h * target))
        start = (w - new_w) // 2
        image = image[:, start:start + new_w]
    else:
        new_h = int(round(w / target))
        start = (h - new_h) // 2
        image = image[start:start + new_h]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def render_cameras(project_root, metadata, output_path):
    panel_w, panel_h, gap = 640, 360, 14
    panels = []
    for camera in CAMERA_ORDER:
        path = resolve_project_path(project_root, metadata["camera_paths"][camera])
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        panel = crop_to_ratio(image, panel_w, panel_h)
        cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1),
                      (238, 240, 243), 4)
        cv2.rectangle(panel, (14, 14), (300, 60), (22, 29, 39), -1)
        cv2.putText(panel, camera, (28, 47), cv2.FONT_HERSHEY_DUPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    spacer = np.full((panel_h, gap, 3), (247, 249, 252), np.uint8)
    rows = []
    for start in (0, 3):
        rows.append(np.hstack([panels[start], spacer, panels[start + 1],
                               spacer, panels[start + 2]]))
    row_gap = np.full((gap, rows[0].shape[1], 3), (247, 249, 252), np.uint8)
    body = np.vstack([rows[0], row_gap, rows[1]])
    title = np.full((105, body.shape[1], 3), (247, 249, 252), np.uint8)
    cv2.putText(title, "SIX-CAMERA INPUT", (20, 46), cv2.FONT_HERSHEY_DUPLEX,
                1.15, (26, 38, 52), 2, cv2.LINE_AA)
    cv2.putText(title,
                f"{metadata['scene_name']} frame {metadata['frame_idx']} | synchronized nuScenes images",
                (22, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (90, 103, 117), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), np.vstack([title, body]))


def render_trajectory(trajectory_path, output_path, metadata):
    data = np.load(trajectory_path)
    pred = np.asarray(data["pred_trajectory"])[0]
    gt = np.asarray(data["gt_trajectory"])[:, :2]
    pred = np.vstack([np.zeros((1, 2), dtype=pred.dtype), pred])
    all_points = np.vstack([pred, gt])
    x_pad = max(3.0, float(np.ptp(all_points[:, 0])) * 0.8)
    x_min, x_max = float(all_points[:, 0].min() - x_pad), float(all_points[:, 0].max() + x_pad)
    y_min = min(-2.0, float(all_points[:, 1].min() - 2.0))
    y_max = float(all_points[:, 1].max() + 4.0)
    width, height = 900, 1000
    left, right, top, bottom = 115, 55, 125, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    canvas = np.full((height, width, 3), (247, 249, 252), np.uint8)

    def pixel(point):
        x, y = float(point[0]), float(point[1])
        px = left + int(round((x - x_min) / (x_max - x_min) * plot_w))
        py = top + plot_h - int(round((y - y_min) / (y_max - y_min) * plot_h))
        return px, py

    for y in np.arange(np.ceil(y_min / 5) * 5, y_max + 0.1, 5):
        py = pixel((x_min, y))[1]
        cv2.line(canvas, (left, py), (left + plot_w, py), (220, 226, 232), 1)
        cv2.putText(canvas, f"{y:.0f}", (58, py + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (91, 103, 116), 1, cv2.LINE_AA)
    for x in np.arange(np.floor(x_min), x_max + 0.1, 1):
        px = pixel((x, y_min))[0]
        cv2.line(canvas, (px, top), (px, top + plot_h), (230, 234, 239), 1)
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h),
                  (100, 112, 124), 2)

    gt_color, pred_color = (70, 168, 77), (51, 104, 220)
    gt_pixels = np.array([pixel(p) for p in gt], np.int32)
    pred_pixels = np.array([pixel(p) for p in pred], np.int32)
    cv2.polylines(canvas, [gt_pixels], False, gt_color, 6, cv2.LINE_AA)
    cv2.polylines(canvas, [pred_pixels], False, pred_color, 6, cv2.LINE_AA)
    for index, point in enumerate(gt_pixels):
        cv2.circle(canvas, tuple(point), 8, gt_color, -1, cv2.LINE_AA)
        if index:
            cv2.putText(canvas, f"{index * 0.5:.1f}s", (point[0] + 10, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, gt_color, 1, cv2.LINE_AA)
    for point in pred_pixels:
        cv2.circle(canvas, tuple(point), 7, pred_color, -1, cv2.LINE_AA)
    draw_ego(canvas, pixel((0, 0)), scale=1.0)

    cv2.putText(canvas, "EGO TRAJECTORY PREDICTION", (32, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (26, 38, 52), 2, cv2.LINE_AA)
    cv2.putText(canvas,
                f"epoch 4 | {metadata['scene_name']} frame {metadata['frame_idx']} | 0.5 s intervals",
                (34, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (90, 103, 117), 2, cv2.LINE_AA)
    cv2.line(canvas, (155, 955), (210, 955), pred_color, 6, cv2.LINE_AA)
    cv2.putText(canvas, "Prediction", (225, 962), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (45, 55, 67), 2, cv2.LINE_AA)
    cv2.line(canvas, (445, 955), (500, 955), gt_color, 6, cv2.LINE_AA)
    cv2.putText(canvas, "Ground truth", (515, 962), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (45, 55, 67), 2, cv2.LINE_AA)
    cv2.putText(canvas, "longitudinal y (m)", (8, 560),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (72, 83, 95), 1, cv2.LINE_AA)
    cv2.putText(canvas, "lateral x (m)", (365, 910),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (72, 83, 95), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_dir = resolve_project_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = resolve_project_path(project_root, args.metadata)
    metadata = json.loads(metadata_path.read_text())
    prediction = resolve_project_path(project_root, args.prediction)
    occupancy_path = output_dir / f"{args.prefix}_occupancy.png"
    cameras_path = output_dir / f"{args.prefix}_six_cameras.png"
    render_occupancy(prediction, occupancy_path, metadata)
    render_cameras(project_root, metadata, cameras_path)
    outputs = [occupancy_path, cameras_path]
    if args.trajectory:
        trajectory_path = resolve_project_path(project_root, args.trajectory)
        output_path = output_dir / f"{args.prefix}_trajectory.png"
        render_trajectory(trajectory_path, output_path, metadata)
        outputs.append(output_path)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
