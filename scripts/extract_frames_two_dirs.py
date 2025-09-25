import argparse, os, cv2, pathlib, random, glob, shutil

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mpg", ".mpeg", ".webm", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".ppm", ".tif", ".tiff", ".webp"}

def list_videos(root):
    return sorted([p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                   if os.path.isfile(p) and os.path.splitext(p)[1].lower() in VIDEO_EXTS])

def list_image_dirs_any(root):
    result = []
    for dirpath, _, filenames in os.walk(root):
        imgs = []
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            ext = os.path.splitext(fpath)[1].lower()
            if ext in IMAGE_EXTS or ext == "" or ext not in VIDEO_EXTS:
                im = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if im is not None:
                    imgs.append(fpath)
        if imgs:
            imgs.sort()
            result.append((dirpath, imgs))
    return result

def sample_idxs(n, k):
    if n <= 0 or k <= 0: return []
    return [int(i * n / (k + 1)) for i in range(1, k + 1)]

def split(items, val_ratio, seed=42):
    random.seed(seed); arr = items[:]; random.shuffle(arr)
    n_val = max(1, int(len(arr) * val_ratio)) if arr else 0
    return arr[n_val:], arr[:n_val]

def extract_from_videos(video_paths, out_dir, per_video=5):
    os.makedirs(out_dir, exist_ok=True)
    for vpath in video_paths:
        name = pathlib.Path(vpath).stem
        cap = cv2.VideoCapture(vpath)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for j, idx in enumerate(sample_idxs(n, per_video)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(os.path.join(out_dir, f"{name}_{j}.jpg"), frame)
        cap.release()

def extract_from_image_dirs(img_dirs, out_dir, per_seq=5):
    os.makedirs(out_dir, exist_ok=True)
    for d, imgs in img_dirs:
        name = pathlib.Path(d).name
        n = len(imgs)
        for j, idx in enumerate(sample_idxs(n, per_seq)):
            src = imgs[idx]
            im = cv2.imread(src, cv2.IMREAD_COLOR)
            if im is None: continue
            dst = os.path.join(out_dir, f"{name}_{j}.jpg")
            cv2.imwrite(dst, im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def extract_all(real_dir, fake_dir, out_root, per_video=5, val_ratio=0.1, seed=42):
    os.makedirs(out_root, exist_ok=True)
    real_videos = list_videos(real_dir)
    if real_videos:
        train, val = split(real_videos, val_ratio, seed)
        extract_from_videos(train, os.path.join(out_root, "train", "0_real"), per_video)
        extract_from_videos(val,   os.path.join(out_root, "val",   "0_real"), per_video)
    else:
        real_dirs = list_image_dirs_any(real_dir)
        train, val = split(real_dirs, val_ratio, seed)
        extract_from_image_dirs(train, os.path.join(out_root, "train", "0_real"), per_video)
        extract_from_image_dirs(val,   os.path.join(out_root, "val",   "0_real"), per_video)
    fake_videos = list_videos(fake_dir)
    if fake_videos:
        train, val = split(fake_videos, val_ratio, seed)
        extract_from_videos(train, os.path.join(out_root, "train", "1_fake"), per_video)
        extract_from_videos(val,   os.path.join(out_root, "val",   "1_fake"), per_video)
    else:
        fake_dirs = list_image_dirs_any(fake_dir)
        train, val = split(fake_dirs, val_ratio, seed)
        extract_from_image_dirs(train, os.path.join(out_root, "train", "1_fake"), per_video)
        extract_from_image_dirs(val,   os.path.join(out_root, "val",   "1_fake"), per_video)
    print("Wrote frames under:", os.path.abspath(out_root))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--per_video", type=int, default=5)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    extract_all(args.real_dir, args.fake_dir, args.out, args.per_video, args.val_ratio, args.seed)
