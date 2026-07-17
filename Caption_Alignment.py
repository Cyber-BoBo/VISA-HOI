import argparse
import os

import numpy as np
import zarr


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def cosine_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = _l2_normalize(x.astype(np.float32, copy=False))
    y = _l2_normalize(y.astype(np.float32, copy=False))
    sim = x @ y.T
    sim = np.clip(sim, -1.0, 1.0)
    return (1.0 - sim).astype(np.float32, copy=False)


def dtw_path(cost: np.ndarray):
    ns, nt = cost.shape
    dp = np.full((ns + 1, nt + 1), np.inf, dtype=np.float32)
    tb = np.full((ns + 1, nt + 1), -1, dtype=np.int8)
    dp[0, 0] = 0.0
    for i in range(1, ns + 1):
        for j in range(1, nt + 1):
            c = cost[i - 1, j - 1]
            a = dp[i - 1, j]
            b = dp[i, j - 1]
            d = dp[i - 1, j - 1]
            if d <= a and d <= b:
                dp[i, j] = c + d
                tb[i, j] = 2
            elif a <= b:
                dp[i, j] = c + a
                tb[i, j] = 0
            else:
                dp[i, j] = c + b
                tb[i, j] = 1

    i, j = ns, nt
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        move = tb[i, j]
        if move == 2:
            i -= 1
            j -= 1
        elif move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        else:
            break
    path.reverse()
    return dp[ns, nt], path


def align_by_dtw(src: np.ndarray, tgt: np.ndarray):
    cost = cosine_distance_matrix(src, tgt)
    total_cost, path = dtw_path(cost)
    nt = tgt.shape[0]
    buckets = [[] for _ in range(nt)]
    for i, j in path:
        buckets[j].append(i)
    mapping = np.zeros(nt, dtype=np.int64)
    last_i = 0
    for j in range(nt):
        if buckets[j]:
            i = int(np.round(np.mean(buckets[j])))
            last_i = i
        else:
            i = last_i
        mapping[j] = i
    mapping = np.clip(mapping, 0, src.shape[0] - 1)
    aligned = src[mapping].astype(np.float32, copy=False)
    return aligned, cost, path, total_cost, mapping


def visualize_dtw(cost: np.ndarray, path, out_path: str, title: str):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path_i = [p[0] for p in path]
    path_j = [p[1] for p in path]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    im = ax.imshow(cost, aspect="auto", origin="lower", cmap="viridis")
    ax.plot(path_j, path_i, color="red", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("GT index")
    ax.set_ylabel("Downsampled index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def open_zarr(path: str, mode: str = "r"):
    return zarr.open(path, mode=mode)


def choose_writable_dir(preferred_dir: str, fallback_dir: str):
    for d in [preferred_dir, fallback_dir]:
        try:
            os.makedirs(d, exist_ok=True)
            test_path = os.path.join(d, ".write_test")
            with open(test_path, "w") as f:
                f.write("ok")
            os.remove(test_path)
            return d
        except Exception:
            continue
    raise RuntimeError(f"Neither {preferred_dir} nor {fallback_dir} is writable.")


def align_single(video_id: str, src_root, gt_root, vis_dir: str):
    src = src_root[video_id]["clip_text"][:]
    tgt = gt_root[video_id]["clip_text"][:]
    aligned, cost, path, total_cost, _ = align_by_dtw(src, tgt)
    vis_path = os.path.join(vis_dir, f"{video_id}_dtw.png")
    visualize_dtw(cost, path, vis_path, f"{video_id} DTW cost={total_cost:.4f} src={len(src)} tgt={len(tgt)}")
    return aligned


def align_all(src_root, gt_root, out_root, vis_dir: str, visualize_first: bool = True):
    video_ids = sorted(set(gt_root.keys()).intersection(set(src_root.keys())))
    if not video_ids:
        raise RuntimeError("No common video ids between caption_clip_text.zarr and caption_clip_text_gt.zarr")
    if visualize_first:
        align_single(video_ids[0], src_root, gt_root, vis_dir)
    for k, video_id in enumerate(video_ids):
        src = src_root[video_id]["clip_text"][:]
        tgt = gt_root[video_id]["clip_text"][:]
        aligned, _, _, _, _ = align_by_dtw(src, tgt)
        g = out_root.require_group(video_id)
        if "clip_text" in g:
            del g["clip_text"]
        chunk_t = min(256, aligned.shape[0])
        g.create_dataset(
            "clip_text",
            data=aligned,
            shape=aligned.shape,
            chunks=(chunk_t, aligned.shape[1]),
            dtype=np.float32,
        )
        if (k + 1) % 10 == 0 or (k + 1) == len(video_ids):
            print(f"[{k + 1}/{len(video_ids)}] aligned {video_id} -> {aligned.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/BimanualInteractions/bimanual_derived_features",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--video_id", type=str, default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--no_single_vis", action="store_true")
    args = parser.parse_args()

    src_path = os.path.join(args.base_dir, "caption_clip_text.zarr")
    gt_path = os.path.join(args.base_dir, "caption_clip_text_gt.zarr")
    requested_out_dir = args.base_dir if args.out_dir is None else args.out_dir
    out_dir = choose_writable_dir(requested_out_dir, os.path.join(os.getcwd(), "caption_alignment_outputs"))
    out_path = os.path.join(out_dir, "caption_clip_text_aligned.zarr")
    vis_dir = os.path.join(out_dir, "caption_alignment_vis")

    src_root = open_zarr(src_path, mode="r")
    gt_root = open_zarr(gt_path, mode="r")
    out_root = open_zarr(out_path, mode="w")

    if args.video_id is None:
        common = sorted(set(src_root.keys()).intersection(set(gt_root.keys())))
        if not common:
            raise RuntimeError("No common video ids between caption_clip_text.zarr and caption_clip_text_gt.zarr")
        video_id = common[0]
    else:
        video_id = args.video_id

    if not args.no_single_vis:
        aligned = align_single(video_id, src_root, gt_root, vis_dir)
        g = out_root.require_group(video_id)
        g.create_dataset(
            "clip_text",
            data=aligned,
            shape=aligned.shape,
            chunks=(min(256, aligned.shape[0]), aligned.shape[1]),
            dtype=np.float32,
        )
        print(f"single_aligned_saved {video_id} -> {aligned.shape} (visualization in {vis_dir})")

    if args.batch:
        align_all(src_root, gt_root, out_root, vis_dir, visualize_first=False)


if __name__ == "__main__":
    main()
