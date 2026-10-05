"""Render the Kitchen mesh using the supplied ICGui camera (OpenCV axes).

Run from the repository root:
    conda run -n nerficgnew python scripts/render_kitchen_mesh.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


C2W = np.array([
    [-0.1068617, -0.02285028, -0.99401129, 2.73235082],
    [0.99424375, 0.0053284, -0.10700918, 1.12787107],
    [0.00774168, -0.9997247, 0.02214934, 1.46325637],
    [0.0, 0.0, 0.0, 1.0],
])
WIDTH, HEIGHT = 2560, 1440
FX, FY, CX, CY = 1247.0, 1243.804, 1280.0, 720.0
NEAR, FAR = 0.01, 100.0


def clip_depth(polygon, plane, keep_greater):
    """Clip camera-space XYZ/RGB vertices, retaining interpolated colors."""
    result = []
    previous = polygon[-1]
    previous_inside = (previous[2] >= plane) if keep_greater else (previous[2] <= plane)
    for current in polygon:
        inside = (current[2] >= plane) if keep_greater else (current[2] <= plane)
        if inside != previous_inside:
            t = (plane - previous[2]) / (current[2] - previous[2])
            result.append(previous + t * (current - previous))
        if inside:
            result.append(current)
        previous, previous_inside = current, inside
    return result


def rasterize(triangle, rgb, depth):
    """Two-sided z-buffer rendering with perspective-correct vertex colors."""
    z = triangle[:, 2]
    x = FX * triangle[:, 0] / z + CX
    y = FY * triangle[:, 1] / z + CY
    xmin = max(0, int(np.ceil(x.min() - 0.5)))
    xmax = min(WIDTH - 1, int(np.floor(x.max() - 0.5)))
    ymin = max(0, int(np.ceil(y.min() - 0.5)))
    ymax = min(HEIGHT - 1, int(np.floor(y.max() - 0.5)))
    if xmin > xmax or ymin > ymax:
        return
    denominator = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
    if abs(denominator) < 1e-12:
        return
    # ICGui rays pass through pixel centers at (column + .5, row + .5).
    px = np.arange(xmin, xmax + 1)[None, :] + 0.5
    py = np.arange(ymin, ymax + 1)[:, None] + 0.5
    a = ((y[1] - y[2]) * (px - x[2]) + (x[2] - x[1]) * (py - y[2])) / denominator
    b = ((y[2] - y[0]) * (px - x[2]) + (x[0] - x[2]) * (py - y[2])) / denominator
    c = 1 - a - b
    inverse_z = a / z[0] + b / z[1] + c / z[2]
    inside = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9) & (inverse_z > 0)
    candidate_depth = np.full_like(inverse_z, np.inf)
    np.divide(1.0, inverse_z, out=candidate_depth, where=inside)
    patch_depth = depth[ymin:ymax + 1, xmin:xmax + 1]
    update = inside & (candidate_depth < patch_depth)
    if not update.any():
        return
    weights = np.stack((a[update] / z[0], b[update] / z[1], c[update] / z[2]), axis=1)
    color = (weights @ triangle[:, 3:]) / inverse_z[update, None]
    rgb[ymin:ymax + 1, xmin:xmax + 1][update] = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    patch_depth[update] = candidate_depth[update]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, default=Path('/home/chen/dataset/scannet++/0c5385e84b/scans/mesh_aligned_0.05.ply'))
    parser.add_argument('--output', type=Path, default=Path('GSPAR/outputs/scannetpp/Kitchen/mesh_camera_view.png'))
    args = parser.parse_args()
    mesh = PlyData.read(str(args.mesh))
    vertices = mesh['vertex']
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    colors = np.column_stack([vertices[k] for k in ('red', 'green', 'blue')])
    faces = np.stack(mesh['face']['vertex_indices'])
    if faces.shape[1] != 3:
        raise ValueError('Expected a triangular mesh.')
    w2c = np.linalg.inv(C2W)
    camera_xyz = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    attributes = np.column_stack((camera_xyz, colors))
    # Reject triangles wholly outside any camera frustum plane before rasterizing.
    x, y, z = camera_xyz.T
    planes = np.column_stack((z - NEAR, FAR - z, FX*x + CX*z,
                              (WIDTH-CX)*z - FX*x, FY*y + CY*z,
                              (HEIGHT-CY)*z - FY*y))
    visible = np.ones(len(faces), dtype=bool)
    for distances in planes.T:
        visible &= np.any(distances[faces] >= 0, axis=1)
    faces = faces[visible]
    print(f'Rendering {len(faces)} candidate triangles at {WIDTH} x {HEIGHT}', flush=True)
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    depth = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float64)
    for face in faces:
        polygon = attributes[face]
        if np.any(polygon[:, 2] < NEAR):
            polygon = clip_depth(polygon, NEAR, True)
        if len(polygon) and np.any(np.asarray(polygon)[:, 2] > FAR):
            polygon = clip_depth(polygon, FAR, False)
        for i in range(1, len(polygon) - 1):
            rasterize(np.array([polygon[0], polygon[i], polygon[i + 1]]), rgb, depth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output)
    metadata = dict(mesh=str(args.mesh.resolve()), c2w=C2W.tolist(), width=WIDTH,
                    height=HEIGHT, fx=FX, fy=FY, cx=CX, cy=CY, near=NEAR, far=FAR,
                    camera_axes='x right, y down, z forward',
                    rendering='unlit perspective-correct vertex RGB; black background',
                    covered_fraction=float(np.isfinite(depth).mean()))
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {args.output}; mesh coverage: {metadata["covered_fraction"]:.2%}', flush=True)


if __name__ == '__main__':
    main()
