import torch
import math
import numpy as np
def fibonacci_sphere(n,shell_radius=1.0,scale=0.3,device="cuda") -> torch.Tensor:
    """Fibonacci spiral points on a unit sphere. Shape: (n, 3)."""
    phi = (1 + 5**0.5) / 2
    i = torch.arange(n, dtype=torch.float32)
    theta = 2 * torch.pi * i / phi
    #z = -i / (n - 1)  # z ranges from 0 to -1
    z = 1 - 2 * i / (n - 1)  # z ranges from 1 to -1
    r = torch.sqrt(torch.clamp(1 - z * z, min=0.0))
    x, y = r * torch.cos(theta), r * torch.sin(theta)
    points = torch.stack([x, y, z], dim=1) * shell_radius
    scales = torch.full((n, 3), scale, device=device, dtype=torch.float32)
    return points,scales


def fibonacci_sphere_with_latitude_scales(
    n: int,
    shell_radius: float = 1.0,
    min_scale: float = 0.005,
    max_scale: float = 0.06,
    device: str = "cuda",
):
    """
    Fibonacci hemisphere sampling 
    Importance samples z to be denser near the equator; returns points on z∈[-1, 0]
    and per-axis scales for a spherical shell of radius `shell_radius`.
    """
    phi = (1 + 5**0.5) / 2
    k = torch.arange(n, dtype=torch.float32, device=device)

    # z importance sampling (denser near |z|≈0); hemisphere: z ∈ [-1, 0]                               
    z_grid = torch.linspace(-1.0, 0.0, 2048, device=device)
    scale_grid = min_scale + (max_scale - min_scale) * torch.abs(z_grid)
    w = (scale_grid + 1e-8).pow(-1.5)                      
    cdf = torch.cumsum(w, dim=0)
    cdf = cdf / cdf[-1]
    u = (k + 0.5) / n
    idx = torch.searchsorted(cdf, u, right=True).clamp(1, 2048 - 1)
    c0, c1 = cdf[idx - 1], cdf[idx]
    z0, z1 = z_grid[idx - 1], z_grid[idx]
    t = (u - c0) / torch.clamp(c1 - c0, min=1e-12)
    z = z0 + t * (z1 - z0)

    # Angular Fibonacci phase for low-discrepancy on the ring at each z
    theta = 2 * torch.pi * k / phi
    r = torch.sqrt(torch.clamp(1 - z * z, min=0.0))
    x, y = r * torch.cos(theta), r * torch.sin(theta)
    points = torch.stack([x, y, z], dim=1) * shell_radius

    # Latitude-dependent scale (smaller near equator → higher density)
    s = min_scale + (max_scale - min_scale) * torch.abs(z)
    scales = s.unsqueeze(1).repeat(1, 3)
    return points, scales


def compute_max_visible_hits(
    points: torch.Tensor,
    camera,
    alpha_2d: torch.Tensor,
    pixel_pad: int = 1,
    tau: float = 1e-3,) -> torch.Tensor:
    """Per-Gaussian visibility (bool): inside image and any neighbor has (1 - alpha) > tau."""
    """
    Visibility test for background Gaussians.

    Projects world-space points into the current camera and marks a point as visible if:
      1) it lies in front of the camera,
      2) its projection falls inside image bounds,
      3) within a (2*pixel_pad+1)^2 neighborhood around the rounded projection,
         any pixel satisfies (1 - alpha) > tau (i.e., sufficient background openness).

    Returns: Bool tensor [N], True indicates the point receives supervision in this view.
    Notes: Applies a Z-axis flip on w2c to match the rasterizer convention; intrinsics use a 0.5-pixel center.
    """
    device = points.device
    H, W = alpha_2d.shape[-2], alpha_2d.shape[-1]

    # world -> camera using the current View convention (camera looks along +Z)
    c2w = camera.c2w.to(device)
    w2c = torch.linalg.inv(c2w)

    pts_h = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)  # [N,4]
    cam_h = (w2c @ pts_h.T).T
    Xc, Yc, Zc, _ = cam_h.unbind(dim=1)

    in_front = Zc > 0

    # intrinsics
    fx = camera.camera.focal_x
    fy = camera.camera.focal_y
    cx = camera.camera.center_x - 0.5
    cy = camera.camera.center_y - 0.5

    # project to pixels
    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy

    in_w = (u >= 0) & (u < (W - 1))
    in_h = (v >= 0) & (v < (H - 1))
    in_img = in_front & in_w & in_h

    u_i = u.clamp(0, W - 1).round().long()
    v_i = v.clamp(0, H - 1).round().long()

    hits = torch.zeros(points.shape[0], dtype=torch.bool, device=device)
    if in_img.any():
        uu = u_i[in_img]
        vv = v_i[in_img]
        k = 2 * pixel_pad + 1

        # neighborhood (k x k) using meshgrid, then linear gather
        offs = torch.arange(-pixel_pad, pixel_pad + 1, device=device)
        du, dv = torch.meshgrid(offs, offs, indexing="ij")  # [k,k]
        u_nb = (uu[:, None, None] + du).clamp(0, W - 1).reshape(-1, k * k)
        v_nb = (vv[:, None, None] + dv).clamp(0, H - 1).reshape(-1, k * k)

        lin_idx = v_nb * W + u_nb  # [N_in, k*k]
        flat_bg = (1.0 - alpha_2d).reshape(-1)
        local_max = flat_bg[lin_idx].amax(dim=1)

        tmp = torch.zeros_like(hits)
        tmp_idx = in_img.nonzero(as_tuple=False).squeeze(1)
        tmp[tmp_idx] = local_max > tau
        hits = tmp

    return hits



@torch.no_grad()
def compute_max_visible_hits_pano(
    points: torch.Tensor,     
    camera,                   
    alpha_2d: torch.Tensor,   
    pixel_pad: int = 1,
    tau: float = 1e-3,
) -> torch.Tensor:
    """
    Visibility test for background Gaussians under a panoramic camera (equirectangular).

    Marks a point as visible if:
      1) it transforms into the current camera,
      2) its direction maps to valid equirectangular pixel coords,
      3) within a (2*pixel_pad+1)^2 neighborhood, any pixel has (1 - alpha) > tau.

    Returns: Bool tensor [N], True indicates the point receives supervision in this view.
    """
    device = points.device
    H, W = alpha_2d.shape[-2], alpha_2d.shape[-1]


    c2w = camera.c2w.to(device)
    w2c = torch.linalg.inv(c2w)
    pts_h = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)  
    cam_pts = (w2c @ pts_h.T).T[:, :3] 

 
    cam_dirs = torch.nn.functional.normalize(cam_pts, dim=1)  
    x, y, z = cam_dirs.unbind(dim=1)


    azimuth = torch.atan2(x, -z)  
    inclination = torch.asin(y.clamp(-1,1))  

    # map to equirectangular pixel coords
    u = (azimuth + math.pi) / (2 * math.pi) * W  
    v = (inclination + math.pi/2) / math.pi * H  

  
    in_w = (u >= 0) & (u < (W - 1))
    in_h = (v >= 0) & (v < (H - 1))
    in_img = in_w & in_h

    u_i = u.clamp(0, W - 1).round().long()
    v_i = v.clamp(0, H - 1).round().long()

    hits = torch.zeros(points.shape[0], dtype=torch.bool, device=device)
    if in_img.any():
        uu = u_i[in_img]
        vv = v_i[in_img]
        k = 2 * pixel_pad + 1

        # neighborhood (k x k) using meshgrid, then linear gather
        offs = torch.arange(-pixel_pad, pixel_pad + 1, device=device)
        du, dv = torch.meshgrid(offs, offs, indexing="ij")  # [k,k]
        u_nb = (uu[:, None, None] + du).clamp(0, W - 1).reshape(-1, k * k)
        v_nb = (vv[:, None, None] + dv).clamp(0, H - 1).reshape(-1, k * k)

        lin_idx = v_nb * W + u_nb  # [N_in, k*k]
        flat_bg = (1.0 - alpha_2d).reshape(-1)
        local_max = flat_bg[lin_idx].amax(dim=1)

        tmp = torch.zeros_like(hits)
        tmp_idx = in_img.nonzero(as_tuple=False).squeeze(1)
        tmp[tmp_idx] = local_max > tau
        hits = tmp

    return hits


def propagate_by_overlay(
    points: torch.Tensor,
    sh0: torch.Tensor,
    overlay_indices: torch.Tensor | None,
    knn_k: int = 2048,
    max_layer: int = 10,
    paint_knn: int = 1,) :
    """Layer-driven color diffusion via KNN; update only where no GT supervision."""
    if overlay_indices is None or overlay_indices.numel() == 0:
        return sh0

    device, N = points.device, points.shape[0]

    # layering from seed indices
    overlay_layers = layer_propagation_knn(
        points,
        overlay_indices.to(device),
        knn_k=knn_k,
        max_layer=max_layer,
    )

    # diffuse colors along layers
    base = sh0.detach().clone()
    propagated = propagate_color_by_knn_layer(
        points,
        base,
        overlay_layers,
        knn_k=paint_knn
    )
    # write back only where no GT (layer > 0)
    out = sh0.clone()
    mask = overlay_layers > 0
    out[mask] = propagated[mask]
    return out

def layer_propagation_knn(points: torch.Tensor, overlay_indices: torch.Tensor, knn_k: int = 8, max_layer: int = 200):
    device = points.device
    N = points.shape[0]

    # Initialize all layers as unknown (-1)
    layers = torch.full((N,), -1, dtype=torch.long, device=device)

    # Points with GT supervision are layer 0
    with_gt = torch.ones(N, dtype=torch.bool, device=device)
    with_gt[overlay_indices] = False
    layers[with_gt] = 0

    # Current frontier (previous layer)
    last = torch.where(layers == 0)[0]  
    cur = 1

    # Chunked min-distance (avoids full NxM cdist)
    def streaming_min_dist(prev_pts, cand_pts,
                        q_chunk=4096, r_chunk=32768,
                        max_matrix_mb=200):

        device = cand_pts.device

        # If no pts, return +inf distances
        if prev_pts.numel() == 0 or cand_pts.numel() == 0:
            return torch.full((cand_pts.shape[0],), float("inf"),
                            device=device, dtype=torch.float32)

        C = cand_pts.shape[0]

        out = torch.full((C,), float("inf"), device=device, dtype=torch.float32)

        qh = cand_pts.half()
        rh = prev_pts.half()

        # Limit max cdist matrix size to avoid OOM
        max_elems = (max_matrix_mb * 1024 * 1024) // 2

        for qs in range(0, C, q_chunk):
            qe = min(C, qs + q_chunk)
            q = qh[qs:qe]
            qs_size = q.shape[0]

            local_min = torch.full((qs_size,), float("inf"), device=device)

            R = rh.shape[0]
            rs = 0

            # Process ref points in chunks, dynamically shrinking if needed
            while rs < R:
                max_r = max_elems // max(qs_size, 1)
                dynamic_r = min(r_chunk, max_r)
                if dynamic_r < 1024:
                    dynamic_r = 1024   

                re = min(R, rs + dynamic_r)

                # Local pairwise distances
                d = torch.cdist(q, rh[rs:re]).half()
                local_min = torch.minimum(local_min,
                                        d.min(dim=1).values.to(torch.float32))

                del d
                rs = re

            out[qs:qe] = local_min

        return out

    # BFS-style layer expansion
    while cur <= max_layer:
        cand = torch.where(layers == -1)[0]
        if cand.numel() == 0 or last.numel() == 0:
            break

        prev_pts = points[last]
        cand_pts = points[cand]

        # Compute min-dist for all candidate points
        md = streaming_min_dist(prev_pts, cand_pts)

        # Compute min-dist for all candidate points
        thr = torch.quantile(md.detach().cpu(), 0.2).to(device)
        this_layer = cand[md < thr]

        if this_layer.numel() == 0:
            break

        layers[this_layer] = cur
        last = this_layer
        cur += 1

    layers[layers == -1] = max_layer + 1
    return layers


def propagate_color_by_knn_layer(points: torch.Tensor, colors: torch.Tensor, layers: torch.Tensor, knn_k: int = 8):
    
    out = colors.clone()
    max_layer = int(layers.max().item())

    # Chunked KNN color propagation
    def streaming_knn(query_pts, ref_pts, ref_vals, k,
                    q_chunk=4096, r_chunk=32768,
                    max_matrix_mb=200):

        if query_pts.numel() == 0 or ref_pts.numel() == 0:
            return query_pts.new_empty((0,) + ref_vals.shape[1:])

        device = query_pts.device
        k = max(1, min(k, ref_pts.shape[0]))

        Q = query_pts.shape[0]
        out_vals = torch.empty((Q,) + ref_vals.shape[1:],
                            device=device, dtype=ref_vals.dtype)

        qh = query_pts.half()
        rh = ref_pts.half()

        max_elems = (max_matrix_mb * 1024 * 1024) // 2

        for qs in range(0, Q, q_chunk):
            qe = min(Q, qs + q_chunk)
            q = qh[qs:qe]                 
            qs_size = q.shape[0]

            # Maintain running top-k distances & indices
            best_d = torch.full((qs_size, k), float("inf"), device=device, dtype=torch.float16)
            best_i = torch.full((qs_size, k), -1, dtype=torch.long, device=device)

            R = rh.shape[0]
            rs = 0

            # Sweep over reference points in safe chunks
            while rs < R:
                max_r = max_elems // max(qs_size, 1)      
                dynamic_r = min(r_chunk, max_r)
                if dynamic_r < 1024:
                    dynamic_r = 1024                    
                re = min(R, rs + dynamic_r)

                d = torch.cdist(q, rh[rs:re]).half()    

                # Merge with previous best-k
                cand_d = torch.cat([best_d, d], dim=1)    
                idx_block = torch.arange(rs, re, device=device).view(1, -1).expand(qs_size, -1)
                cand_i = torch.cat([best_i, idx_block], dim=1)

                # Keep only k smallest
                best_d, sel = torch.topk(cand_d, k, dim=1, largest=False)
                best_i = cand_i.gather(1, sel)

                del d, cand_d, cand_i, idx_block
                rs = re

            # Normalize inverse-distance weights
            w = 1.0 / (best_d.float() + 1e-4)
            w = w / w.sum(dim=1, keepdim=True)

            # Weighted sum of reference colors
            g = ref_vals[best_i]             
            ww = w
            while ww.ndim < g.ndim:
                ww = ww.unsqueeze(-1)

            out_vals[qs:qe] = (g * ww).sum(dim=1)

        return out_vals

    # Process layers from low to high
    for l in range(1, max_layer + 1):
        idx = torch.where(layers == l)[0]
        if idx.numel() == 0:
            continue

        prev_idx = torch.where(layers < l)[0]
        if prev_idx.numel() == 0:
            continue

        q = points[idx]
        r = points[prev_idx]
        rv = out[prev_idx]

        # KNN-based color diffusion for this layer
        out[idx] = streaming_knn(q, r, rv, k=knn_k)

    return out


@torch.no_grad()
def knn_fix_opacity(
    points: torch.Tensor,
    opacity: torch.Tensor,       
    mask_low: torch.Tensor,      # bad points (to fix)
    mask_good: torch.Tensor,     # good reference points
    knn_k: int = 8,
) -> torch.Tensor:
    """
    Smooth opacity of 'bad' points using KNN from nearby 'good' points.
    Returns new probabilities; untouched entries keep original values.
    """
    N = points.shape[0]
    result = opacity.clone()

    if mask_low.sum() == 0 or mask_good.sum() == 0:
        return result

    bad_idx = torch.where(mask_low)[0]
    good_idx = torch.where(mask_good)[0]
    bad_pts = points[bad_idx]   
    good_pts = points[good_idx] 
    good_opacity = opacity[good_idx]   

    good_opacity = good_opacity.squeeze(1)   

   
    k_eff = min(knn_k, good_pts.shape[0])
    knn_dists, knn_indices = batched_cdist_knn(bad_pts, good_pts, k_eff, batch_size=4096)

    expanded_good = good_opacity.unsqueeze(0).expand(knn_indices.shape[0], -1)   # (Nb, Ng)
    gathered_opacity = torch.gather(expanded_good, 1, knn_indices)  # (Nb, k)

    weights = 1.0 / (knn_dists + 1e-4)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    new_opacity = (weights * gathered_opacity).sum(dim=1)  # (Nb,)

    if result.dim() == 2 and result.shape[1] == 1:
        result[bad_idx, 0] = new_opacity
    else:
        result[bad_idx] = new_opacity
    return result


def batched_cdist_knn(
    bad_pts, good_pts, k, batch_size=4096
):
    """
    Compute k-NN distances/indices from bad_pts to good_pts using batched cdist to save memory.
    """
    device = bad_pts.device
    Nb = bad_pts.shape[0]
    Ng = good_pts.shape[0]
    d_k = []
    i_k = []
    for i in range(0, Nb, batch_size):
        sub_bad = bad_pts[i:i+batch_size]
        dist = torch.cdist(sub_bad, good_pts)  
        k_eff = min(k, Ng)
        dd, ii = dist.topk(k_eff, largest=False)
        d_k.append(dd)
        i_k.append(ii)
        del dist
    d_k = torch.cat(d_k, dim=0)
    i_k = torch.cat(i_k, dim=0)
    return d_k, i_k



def compute_lambda(complexity):
    """
    Empirical ranges from experiments
    Mapping from sky complexity → color regularization weight.
    Sigmoid curve clipped to [0.01, 0.1].
    """
    a = 20  
    k = 0.37  
    sig = 1 / (1 + np.exp(-a * (k - complexity)))
    return 0.01 + 0.09 * sig



def interp_HTGS(x, xmin, xmax, ymin, ymax, k=4.0):
    """
    Softly-bounded interpolation.
    - Inside [xmin, xmax]: almost linear.
    - Outside: flattens, avoiding extreme explosion.
    k controls sharpness of flattening.
    """
    # normalize to [0,1]
    t = (x - xmin) / (xmax - xmin)
    # sigmoid smooth clamp
    t = 1 / (1 + np.exp(-k*(t - 0.5)))
    return ymin + (ymax - ymin) * t

def interp_SPaGS(R: float, C: float):
    """
    Empirical ranges from experiments
    Scene-adaptive presets (empirical): map scene radius R (meters) and sky
    complexity C to (BASE_GAUSSIANS, SCALE_RATIO).
    - BASE_GAUSSIANS grows smoothly (log) with radius.
    - SCALE_RATIO slightly increases with radius and decreases with complexity, bounded to [0.02, 0.04].
    """
    R_new = max(R / 1000.0, 0.001)
    N = 8000 + (23000 - 8000) * (math.log1p(R_new) / math.log1p(5.0))

    S = 0.04 + 0.01 * (R_new / (R_new + 2.0)) - 0.004 * (C - 0.3)

    # limited at [0.02, 0.04]
    S = max(0.02, min(0.04, S))

    return int(round(N)),S
