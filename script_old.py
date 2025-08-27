import requests
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")  # hides UserWarning/DeprecationWarning

from transformers import OwlViTProcessor, OwlViTForObjectDetection


# ---------------------------
# Utilities: IoU + per-class NMS
# ---------------------------
def iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = a1 + a2 - inter + 1e-9
    return inter / union


def nms_per_class(boxes, scores, labels, iou_thresh=0.7):
    """Return kept indices after per-class NMS."""
    keep_idx = []
    by_cls = {}
    for i, lab in enumerate(labels):
        by_cls.setdefault(lab, []).append(i)

    for lab, idxs in by_cls.items():
        idxs = sorted(idxs, key=lambda i: float(scores[i]), reverse=True)
        kept = []
        while idxs:
            i = idxs.pop(0)
            kept.append(i)
            idxs = [j for j in idxs if iou_xyxy(boxes[i], boxes[j]) < iou_thresh]
        keep_idx.extend(kept)
    return sorted(keep_idx)


# ---------------------------
# Simple English phrasing helpers
# ---------------------------
EN_NAMES = {
    "chair": ("chair", "chairs"),
    "table": ("table", "tables"),
    "floor": ("floor", "floors"),
    "office": ("office", "offices"),
    "monitor": ("monitor", "monitors"),
    "keyboard": ("keyboard", "keyboards"),
    "speaker": ("speaker", "speakers"),
    "mouse": ("mouse", "mice"),
    "lamp": ("lamp", "lamps"),
    "plant": ("plant", "plants"),
    "shelf": ("shelf", "shelves"),
}

def count_phrase_en(n, singular, plural):
    if n == 0:
        return ""
    if n == 1:
        return f"one {singular}"
    return f"{n} {plural}"


def describe_scene(result, image_size_hw, score_thresh=0.05, iou_thresh=0.7
):
    H, W = image_size_hw
    boxes = [b.tolist() for b in result["boxes"]]
    scores = [float(s) for s in result["scores"]]
    labels = [str(l) for l in result["text_labels"]]

    # Threshold
    idx = [i for i, s in enumerate(scores) if s >= score_thresh]
    boxes = [boxes[i] for i in idx]
    scores = [scores[i] for i in idx]
    labels = [labels[i] for i in idx]

    # Per-class NMS
    keep = nms_per_class(boxes, scores, labels, iou_thresh=iou_thresh)
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]
    labels = [labels[i] for i in keep]

    # Aggregate counts
    per_cls = {}
    for lab, sc, box in zip(labels, scores, boxes):
        per_cls.setdefault(lab, {"count": 0, "scores": [], "boxes": []})
        per_cls[lab]["count"] += 1
        per_cls[lab]["scores"].append(sc)
        per_cls[lab]["boxes"].append(box)

    # Scene hints
    scene_bits = []
    if "office" in per_cls and per_cls["office"]["count"] >= 1:
        scene_bits.append("It looks like an office.")
    if "floor" in per_cls and per_cls["floor"]["count"] >= 1:
        areas = []
        for b in per_cls["floor"]["boxes"]:
            areas.append(max(0, b[2]-b[0]) * max(0, b[3]-b[1]))
        if areas:
            cover = 100.0 * (max(areas) / (W * H))
            scene_bits.append(f"The floor covers roughly {cover:.0f}% of the image.")

    # Main object counts (ordered for readability)
    core = []
    for k in ["table", "chair", "monitor", "keyboard", "mouse", "speaker", "lamp", "plant", "shelf", "floor"]:
        if k in per_cls:
            n = per_cls[k]["count"]
            sing, plur = EN_NAMES.get(k, (k, k + "s"))
            phrase = count_phrase_en(n, sing, plur)
            if phrase:
                core.append(phrase)

    # Confidence snippets (optional)
    conf_bits = []
    for k in ["table", "chair", "monitor", "keyboard", "mouse", "speaker", "lamp", "plant", "shelf", "floor"]:
        if k in per_cls:
            mean_c = sum(per_cls[k]["scores"]) / len(per_cls[k]["scores"])
            conf_bits.append(f"avg confidence {k}: {mean_c:.2f}")

    # Build text
    parts = []
    if core:
        opener = "In the office there are " if "office" in per_cls else "Detected "
        if len(core) == 1:
            parts.append(opener + core[0] + ".")
        elif len(core) == 2:
            parts.append(opener + f"{core[0]} and {core[1]}.")
        else:
            parts.append(opener + f"{', '.join(core[:-1])}, and {core[-1]}.")
    if scene_bits:
        parts.append(" ".join(scene_bits))
    if conf_bits:
        parts.append(" | " + " ; ".join(conf_bits))

    description = " ".join(parts) if parts else "No objects above the confidence threshold."
    return description, per_cls, (boxes, scores, labels)




# ---------------------------
# Run OWL-ViT and visualize
# ---------------------------
def main():
    # Model & processor
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    # Image
    url = "https://www.itoki.jp/resources/column/article/office-workspace/assets/img/office-workspace_KV.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    #path = "/home/shirb/transformers/images/4.png"

    #image = Image.open(path).convert("RGB")

    # Prompts 
    text_labels = [[
        "office", "floor", "chair", "table",
        "screen",
        "keyboard",
        "mouse",
        "speaker",
        "lamp",
        "plant",
        "shelf"
    ]]
    # Forward
    inputs = processor(text=text_labels, images=image, return_tensors="pt")

    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.tensor([(image.height, image.width)])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
    )
    result = results[0]
    rows = []
    for box, score, text_label in zip(result["boxes"], result["scores"], result["text_labels"]):
        box = [round(i, 1) for i in box.tolist()]
        rows.append([text_label, f"{score:.3f}", box])

    print("\n=== Raw Detections ===")
    print(tabulate(rows, headers=["Label", "Confidence", "BBox [x1,y1,x2,y2]"], tablefmt="pretty"))
   
    # Build description + get filtered detections
    description, per_cls, (boxes, scores, labels) = describe_scene(
        results[0],
        image_size_hw=(image.height, image.width),
        score_thresh=0.05,
        iou_thresh=0.7
    )
    print("\n=== Scene Summary ===")

    print(description)

    with open("outputs/scene_summary.txt", "w") as f:
        f.write(description + "\n")

    # ---------------------------
    # Draw detections (color per class)
    # ---------------------------
    # stable order of unique labels for color mapping
    unique_labels = sorted(set(labels))
    cmap = cm.get_cmap("tab10", len(unique_labels))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")

    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = [float(x) for x in box]
        color = label_to_color[label]

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            xmin,
            max(ymin - 6, 6),
            f"{label} ({score:.2f})",
            fontsize=11,
            bbox=dict(facecolor=color, alpha=0.5, edgecolor="none")
        )

    plt.tight_layout()
    out_path = "outputs/owlvit_annotated.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()



