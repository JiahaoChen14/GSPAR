import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from collections import Counter

# cache the segmentation model to avoid reloading
_SEGFORMER_PROCESSOR = None
_SEGFORMER_MODEL = None
SKY_CACHE_VERSION = "2"

def _get_segformer():
    """Lazy-load SegFormer once and reuse."""
    global _SEGFORMER_PROCESSOR, _SEGFORMER_MODEL
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError(
            'Sky-mask generation requires the optional "transformers" package.'
        ) from exc
    if _SEGFORMER_PROCESSOR is None:
        _SEGFORMER_PROCESSOR = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    if _SEGFORMER_MODEL is None:
        _SEGFORMER_MODEL = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        ).to("cuda").eval()
    return _SEGFORMER_PROCESSOR, _SEGFORMER_MODEL


def resolve_sky_cache_dir(dataset_path: str | Path, dataset_root: str | Path, output_root: str | Path) -> Path:
    """Maps a dataset directory to its method-owned sky-cache directory."""
    dataset_path = Path(dataset_path)
    dataset_root = Path(dataset_root).resolve()
    if not dataset_path.is_absolute():
        dataset_path = dataset_root.parent / dataset_path
    dataset_path = dataset_path.resolve()

    try:
        relative_path = dataset_path.relative_to(dataset_root)
    except ValueError:
        # Preserve the useful suffix for external paths such as
        # /data/project/dataset/tanks_and_temples/train.
        dataset_parts = dataset_path.parts
        if "dataset" in dataset_parts:
            dataset_index = len(dataset_parts) - 1 - dataset_parts[::-1].index("dataset")
            relative_parts = dataset_parts[dataset_index + 1:]
            relative_path = Path(*relative_parts) if relative_parts else Path(dataset_path.name)
        else:
            relative_path = Path(dataset_path.name)

    return Path(output_root) / "mask" / relative_path


def get_view_image_name(view) -> str:
    """Returns an RGB filename through the public GSPAR 2.0 View API."""
    load_parameters, _ = view.get_parallel_load_helpers("rgb")
    return Path(load_parameters["path"]).name


def is_sky_cache_current(save_dir: str | Path) -> bool:
    """Checks whether the cache was generated with the current split mapping."""
    try:
        return (Path(save_dir) / "cache_version.txt").read_text().strip() == SKY_CACHE_VERSION
    except OSError:
        return False


def load_cached_sky_masks_and_color(imgs, names, save_dir: str | Path):
    """Loads and validates cached masks in the same order as the given views."""
    save_dir = Path(save_dir)
    try:
        target_color = np.load(save_dir / "sky_global_color.npy", allow_pickle=False)
    except (OSError, ValueError):
        return None
    if target_color.shape != (3,):
        return None

    masks = []
    for img, name in zip(imgs, names, strict=True):
        mask_path = save_dir / "sky_masks" / f"{Path(name).stem}_mask.png"
        try:
            with Image.open(mask_path) as mask_image:
                mask_array = np.array(mask_image.convert("L"), dtype=np.uint8, copy=True)
        except (OSError, ValueError):
            return None
        if mask_array.shape != tuple(img.shape[-2:]):
            return None
        masks.append(torch.from_numpy(mask_array > 0))
    return target_color, masks


def load_cached_sky_complexity(save_dir: str | Path) -> float | None:
    """Loads cached global sky complexity when available."""
    try:
        with np.load(Path(save_dir) / "sky_hist" / "global_hist.npz", allow_pickle=False) as data:
            return float(np.asarray(data["complexity"]).item())
    except (OSError, KeyError, ValueError):
        return None


def write_sky_cache_version(save_dir: str | Path) -> None:
    """Marks a fully generated sky cache as current."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "cache_version.txt").write_text(f"{SKY_CACHE_VERSION}\n")




def compute_sky_mode_color_and_masks(imgs, names, save_dir: str):
    """
    Compute sky masks and global sky color mode using SegFormer segmentation.
    Each mask isolates sky regions; the mode color is estimated from top sky pixels.
    """
    os.makedirs(save_dir, exist_ok=True)
    mask_dir = os.path.join(save_dir, "sky_masks")
    os.makedirs(mask_dir, exist_ok=True)

    processor, model = _get_segformer()
    target_idx = next(i for i, lbl in model.config.id2label.items() if lbl.lower() == "sky")

    masks, colors = [], []
    with torch.no_grad():
        for img, name in zip(imgs, names):
            img_u8 = (img * 255).round().byte() if img.max() <= 1.0 else img.byte()
            img_np = img_u8.cpu().permute(1, 2, 0).numpy()
            H, W = img_np.shape[:2]

            logits = model(**processor(images=Image.fromarray(img_np), return_tensors="pt").to("cuda")).logits
            class_mask = (logits.argmax(1)[0].cpu().numpy() == target_idx)

            if class_mask.shape != (H, W):
                class_mask = np.array(
                    Image.fromarray(class_mask.astype(np.uint8) * 255).resize((W, H), resample=Image.NEAREST)
                ) > 128


            if class_mask.sum() == 0:
                mask_t = torch.zeros((H, W), dtype=torch.bool)
                mode_color = None
            else:
                mask_t = torch.from_numpy(class_mask.copy()).to(torch.bool)
                # Use the top 5% region to find the most frequent sky color (mode)
                top_rows = max(1, int(np.ceil(H * 0.05)))
                top_pixels = img_np[:top_rows][class_mask[:top_rows]]
                if top_pixels.size:
                    px, _ = Counter(map(tuple, top_pixels)).most_common(1)[0]
                    mode_color = np.array(px, dtype=np.float32)
                else:
                    mode_color = None  


            masks.append(mask_t)

            base = Path(name).stem
            mask_path = os.path.join(mask_dir, f"{base}_mask.png")
            Image.fromarray(mask_t.cpu().numpy().astype(np.uint8) * 255).save(mask_path)
            if mode_color is not None:
                colors.append(mode_color)

    if len(colors) > 0:
        mean_color = np.mean(colors, axis=0)
    else:
        mean_color = np.array([0, 0, 0], dtype=np.float32)
    mean_color = mean_color.round().astype(np.uint8)
    np.save(os.path.join(save_dir, "sky_global_color.npy"), mean_color)

    return mean_color, masks





def _rgb_to_hsv_np(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Vectorized RGB→HSV conversion (0–255 input → [0,1] output).
    Returns [H, S, V] with hue normalized to [0,1].
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc == 0, 0.0, delta / (maxc + 1e-12))

    rc = (maxc - r) / (delta + 1e-12)
    gc = (maxc - g) / (delta + 1e-12)
    bc = (maxc - b) / (delta + 1e-12)

    h = np.zeros_like(maxc)

    mask = (maxc == r)
    h[mask] = (bc - gc)[mask]
    mask = (maxc == g)
    h[mask] = 2.0 + (rc - bc)[mask]
    mask = (maxc == b)
    h[mask] = 4.0 + (gc - rc)[mask]

    h = (h / 6.0) % 1.0
    h = np.where(delta < 1e-12, 0.0, h)

    hsv = np.stack([h, s, v], axis=-1).astype(np.float32)
    return hsv



def compute_sky_hist_and_complexity(
    imgs, names, masks, save_dir: str,
    hue_bins: int = 36,          # number of hue histogram bins (10° per bin)
    sat_thresh: float = 0.2,     # below this S, pixel is considered near-colorless
    val_white: float = 0.85,     # brightness threshold for white
    sv_bins: int = 32,                      # bins for S/V entropy
    w_balance: float = 0.4,                 # weight for white–colored balance
    w_hue: float = 0.4,                     # weight for hue entropy
    w_val: float = 0.35,                    # weight for value entropy
    w_sat: float = 0.25                     # weight for saturation entropy
):
    """
    Compute global sky color histogram and complexity statistics using sky masks.
    Returns a dictionary of global stats including color distribution, entropy, and overall complexity.
    """



    hist_dir = os.path.join(save_dir, "sky_hist")
    os.makedirs(hist_dir, exist_ok=True)


    global_hue_hist = np.zeros(hue_bins, dtype=np.int64)
    global_white = 0
    global_colored = 0   
    global_other = 0

    all_S, all_V = [], []

    hue_bins_edges = np.linspace(0.0, 1.0, hue_bins + 1, dtype=np.float32)

    for img, name, mask_t in zip(imgs, names, masks):
        img_u8 = (img * 255).round().byte() if img.max() <= 1.0 else img.byte()
        img_np = img_u8.cpu().permute(1, 2, 0).numpy()     # [H,W,3]
        mask = mask_t.cpu().numpy().astype(bool)

        if mask.sum() == 0:
            continue

        hsv = _rgb_to_hsv_np(img_np)
        h = hsv[..., 0][mask]
        s = hsv[..., 1][mask]
        v = hsv[..., 2][mask]

        # White: near-colorless and bright
        is_white = (s < sat_thresh) & (v >= val_white)

        # Colored: anything non-white with enough saturation
        is_colored = (~is_white) & (s >= sat_thresh)

        # Other: non-white but low saturation (e.g., dark gray clouds/haze)
        is_other = (~is_white) & (~is_colored)

        global_white   += int(is_white.sum())
        global_colored += int(is_colored.sum())
        global_other   += int(is_other.sum())

        # Hue histogram over colored pixels only
        if is_colored.any():
            hue_hist, _ = np.histogram(h[is_colored], bins=hue_bins_edges)
            global_hue_hist += hue_hist

        all_S.append(s.astype(np.float32))
        all_V.append(v.astype(np.float32))

    if len(all_S) == 0:
        all_S = np.empty((0,), dtype=np.float32)
        all_V = np.empty((0,), dtype=np.float32)
    else:
        all_S = np.concatenate(all_S, axis=0)
        all_V = np.concatenate(all_V, axis=0)

    sky_total = global_white + global_colored + global_other
    if sky_total == 0:
        global_stats = {
            "hue_hist": global_hue_hist.astype(np.int64),
            "hue_hist_norm": global_hue_hist.astype(np.float32),
            "white_count": 0, "colored_count": 0, "other_count": 0,  
            "p_white": 0.0, "p_color": 0.0, "p_other": 0.0,        
            "balance": 0.0,
            "H_hue": 0.0, "H_sat": 0.0, "H_val": 0.0, "H_mix": 0.0,
            "complexity": 0.0
        }
        np.savez_compressed(os.path.join(hist_dir, "global_hist.npz"), **global_stats)
        return global_stats

    p_white   = global_white / sky_total
    p_colored = global_colored / sky_total
    p_other   = global_other / sky_total

    # Balance between white and colored 
    balance = 1.0 - abs(p_colored - p_white)

    def _norm_entropy_from_hist(hist: np.ndarray) -> float:
        eps = 1e-12
        total = hist.sum()
        if total <= 0:
            return 0.0
        p = hist.astype(np.float64) / float(total)
        H = -np.sum(p * np.log(p + eps))
        return float(H / np.log(len(hist) + eps))

    def _norm_entropy_from_values(vals: np.ndarray, bins: int) -> float:
        if vals.size == 0:
            return 0.0
        hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0))
        return _norm_entropy_from_hist(hist)

    H_hue = _norm_entropy_from_hist(global_hue_hist)  
    H_sat = _norm_entropy_from_values(all_S, sv_bins) 
    H_val = _norm_entropy_from_values(all_V, sv_bins) 

    wh_sum = max(w_hue + w_val + w_sat, 1e-12)
    w_h = w_hue / wh_sum
    w_v = w_val / wh_sum
    w_s = w_sat / wh_sum

    H_mix = w_h * H_hue + w_v * H_val + w_s * H_sat

    w_balance = float(w_balance)
    w_content = max(1.0 - w_balance, 0.0)
    complexity = w_balance * balance + w_content * H_mix

    hue_total = global_hue_hist.sum()
    hue_hist_norm = (global_hue_hist / hue_total).astype(np.float32) if hue_total > 0 else global_hue_hist.astype(np.float32)

    global_stats = {
        "hue_hist": global_hue_hist.astype(np.int64),
        "hue_hist_norm": hue_hist_norm,
        "white_count": int(global_white),
        "colored_count": int(global_colored),       
        "other_count": int(global_other),
        "p_white": float(p_white),
        "p_colored": float(p_colored),              
        "p_other": float(p_other),
        "balance": float(balance),
        "H_hue": float(H_hue),
        "H_sat": float(H_sat),
        "H_val": float(H_val),
        "H_mix": float(H_mix),
        "complexity": float(complexity)
    }
    np.savez_compressed(os.path.join(hist_dir, "global_hist.npz"), **global_stats)
    return global_stats



