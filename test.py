import requests
from PIL import Image
import torch
import cv2
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.pyplot as plt
import numpy as np
import math
import os

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
video_path = "Multifunctional Creative Workspace – Office Tour 2023_short.mp4"

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = 30  # Process every 30th frame

text_labels = [["computer", "desk", "chair", "monitor", 
                "keyboard", "mouse", "person", "phone", 
                "notebook", "pen", "shelf", "cabinet", 
                "plant", "bottle", "cup", "person",
                "laptop", "bag", "table", "window", "door",
                "building", "tree", "car", "road", "sky"]]

# Assign a unique color to each label
label_colors = {}
np.random.seed(42)
for label in text_labels[0]:
    label_colors[label] = tuple(np.random.randint(0, 255, 3).tolist())

# Prepare video writer
assert cap.isOpened(), f"Failed to open {video_path}"

fps_in = cap.get(cv2.CAP_PROP_FPS) or 0.0
if fps_in <= 1 or math.isnan(fps_in):
    fps_in = 30.0  # fallback
fps_out = fps_in / frame_interval
# OpenCV/MP4 doesn’t like < 1 fps — clamp:
fps_out = max(1.0, float(fps_out))

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# many codecs want even dimensions:
width  -= width  % 2
height -= height % 2
out_path = '/home/shirb/transformers/OWL-ViT/output_annotated.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))
if not out.isOpened():
    # fallback to AVI+XVID if your OpenCV build’s MP4/ffmpeg is finicky
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = '/home/shirb/transformers/OWL-ViT/output_annotated.avi'
    out = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))
    assert out.isOpened(), "VideoWriter failed to open (mp4v and XVID)"
plt.ion()  # Interactive mode

out_1 = cv2.VideoWriter('test_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
for _ in range(10):
    out_1.write(255*np.ones((height, width, 3), dtype=np.uint8))
out_1.release()

for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    if i % frame_interval != 0:
        continue

    if i >= 1_000:
        break  # Limit to first 10,000 frames for testing

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=text_labels, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)])
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
    )
    result = results[0]
    boxes, scores, text_labels_out = result["boxes"], result["scores"], result["text_labels"]

    frame_disp = frame.copy()
    for box, score, text_label in zip(boxes, scores, text_labels_out):
        box = [int(j) for j in box.tolist()]
        color = label_colors.get(text_label, (0, 255, 0))
        label_str = f"{text_label}: {score:.2f}"
        cv2.rectangle(frame_disp, (box[0], box[1]), (box[2], box[3]), color, 2)  
        cv2.putText(frame_disp, label_str, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    plt.clf()
    plt.imshow(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
    plt.title(f"Frame {i}")
    plt.axis('off')
    plt.pause(0.01)
    try:
        if frame_disp.shape[1] != width or frame_disp.shape[0] != height:
            print("Resizing frame from", frame_disp.shape, "to", (width, height))
            frame_disp = cv2.resize(frame_disp, (width, height))
        else:
            print("Frame size is correct:", frame_disp.shape)
        out.write(frame_disp)
        print(f"Processed frame {i}")
    except Exception as e:
        print(f"Error writing frame {i}: {e}")

plt.ioff()
cap.release()
out.release()
print("Saved annotated video to 'output_annotated.mp4'")

