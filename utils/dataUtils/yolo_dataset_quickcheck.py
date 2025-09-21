
"""
YOLO Dataset QuickCheck
- Validates YOLO label files (values in range, class ids in set)
- Summarizes class counts, box size distribution
- Optionally renders N sample overlays to verify alignment
Usage:
    python yolo_dataset_quickcheck.py /path/to/dataset --nc 3 --names crack delamination efflorescence --render 20
"""
import argparse, os, random
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str, help="root with train/val[/test]/{images,labels}")
    ap.add_argument("--splits", nargs="+", default=["train","val"], help="splits to scan")
    ap.add_argument("--nc", type=int, required=True, help="number of classes")
    ap.add_argument("--names", nargs="+", default=None, help="class names (optional)")
    ap.add_argument("--render", type=int, default=0, help="render N random samples per split to 'quickcheck_out'")
    return ap.parse_args()

def load_labels(txt_path: Path) -> List[Tuple[int,float,float,float,float]]:
    items = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            parts = line.split()
            if len(parts) != 5: 
                raise ValueError(f"Bad line in {txt_path}: {line}")
            c = int(parts[0])
            x,y,w,h = map(float, parts[1:])
            items.append((c,x,y,w,h))
    return items

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def render_overlay(img_path: Path, labels: List[Tuple[int,float,float,float,float]], out_path: Path, names=None):
    img = Image.open(img_path).convert("RGB")
    W,H = img.size
    draw = ImageDraw.Draw(img)
    for (c,x,y,w,h) in labels:
        cx,cy,bw,bh = x*W, y*H, w*W, h*H
        x1 = clamp(cx - bw/2, 0, W-1)
        y1 = clamp(cy - bh/2, 0, H-1)
        x2 = clamp(cx + bw/2, 0, W-1)
        y2 = clamp(cy + bh/2, 0, H-1)
        draw.rectangle([x1,y1,x2,y2], outline=255, width=2)
        label = str(c) if names is None or c>=len(names) else names[c]
        draw.text((x1+3, y1+3), label, fill=255)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

def main():
    args = parse_args()
    dataset = Path(args.dataset)
    names = args.names

    for split in args.splits:
        img_dir = dataset / split / "images"
        lbl_dir = dataset / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"[{split}] missing folders: {img_dir.exists()} {lbl_dir.exists()}")
            continue

        imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".tif",".tiff"]])
        lbls = sorted([p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"])
        img_names = {p.stem for p in imgs}
        lbl_names = {p.stem for p in lbls}

        missing_lbl = img_names - lbl_names
        missing_img = lbl_names - img_names
        print(f"\n== {split} ==")
        print(f"images: {len(imgs)}, labels: {len(lbls)}")
        if missing_lbl: print(f"  - missing labels for {len(missing_lbl)} images (e.g., {list(sorted(missing_lbl))[:5]})")
        if missing_img: print(f"  - missing images for {len(missing_img)} labels (e.g., {list(sorted(missing_img))[:5]})")

        # Stats
        class_counts = [0]*args.nc
        bad_range = 0
        zero_area = 0
        out_samples = []

        for lbl in lbls:
            items = load_labels(lbl)
            for (c,x,y,w,h) in items:
                if not (0 <= c < args.nc):
                    raise ValueError(f"Class id {c} out of range in {lbl}")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    bad_range += 1
                if w*h <= 0:
                    zero_area += 1
                class_counts[c] += 1
            if args.render and random.random() < (args.render / max(1,len(lbls))):
                out_samples.append(lbl)

        print(f"class counts: {class_counts}")
        print(f"bad normalized boxes: {bad_range}, zero-area boxes: {zero_area}")

        # Render samples
        if args.render:
            out_dir = dataset / "quickcheck_out" / split
            chosen = random.sample(lbls, min(args.render, len(lbls))) if len(lbls) else []
            for lbl in chosen:
                stem = lbl.stem
                img_path = None
                for ext in [".jpg",".jpeg",".png",".tif",".tiff"]:
                    p = img_dir / f"{stem}{ext}"
                    if p.exists():
                        img_path = p; break
                if img_path is None:
                    continue
                items = load_labels(lbl)
                render_overlay(img_path, items, out_dir / f"{stem}.jpg", names=names)
            print(f"rendered {len(chosen)} samples to {out_dir}")

if __name__ == "__main__":
    main()
