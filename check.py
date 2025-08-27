
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from generate_labels import GenerateLabels

def draw_bounding_boxes_matplotlib(image, results, texts, threshold=0.3):
    """Draw bounding boxes using matplotlib"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Define colors for different classes
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'orange', 'pink']

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score >= threshold:
            # Get box coordinates
            x1, y1, x2, y2 = box.tolist()

            # Calculate width and height
            width = x2 - x1
            height = y2 - y1

            # Get label text and color
            label_text = f"{texts[0][label]}: {score:.2f}"
            color = colors[label % len(colors)]

            # Create rectangle patch
            rect = patches.Rectangle((x1, y1), width, height,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Add label
            ax.text(x1, y1 - 5, label_text, color='white', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))

    ax.axis('off')
    plt.title('Object Detection Results')
    plt.tight_layout()
    return fig


def main():
    # Load model & processor
    print("Loading model and processor...")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    # Load image - CHANGE THIS PATH TO YOUR IMAGE
    image_path =  "/home/user1/OWL-ViT/images/IMG_5723.JPG"  # Update this with your image path

    image = Image.open(image_path)

    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Define text queries (object descriptions)
    # You can modify this list to detect different objects
    #texts = [["screen","keyboard","touchpad","webcam","logo","port"]]
    #texts = [["a glass of water","water","glass","bottle","telephone","laptop","computer"]]
    #gen_label = GenerateLabels()
    #texts = gen_label.get_labels()
    #texts = [['laptop', 'laptop[keyboard[enter key]]', 'laptop[keyboard]', 'laptop[touchpad]', 'desk[cable grommet]', 'desk[drawer]', 'desk[leg]', 'keyboard[enter key]', 'keyboard[keycap]', 'keyboard[spacebar]', 'laptop[keyboard]']]
    texts = [['a laptop', 'a laptop[a keyboard[an enter key]]', 'a laptop[a keyboard]', 'a keyboard[enter key]', 'a laptop[a computer mouse]']]
    #texts = [[ 'a laptop', 'a keyboard', 'a enter key','a computer mouse']]

    print(f"Looking for: {texts[0]}")

    # Prepare inputs
    print("Processing inputs...")
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocess
    target_sizes = torch.tensor([image.size[::-1]])
    threshold = 0.2  # Adjust this value to filter detections
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold
    )[0]

    # Print results
    print("\n=== Detection Results ===")
    detection_count = 0
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score >= threshold:
            detection_count += 1
            print(f">>> {texts[0][label]}: {round(score.item(), 3)} at {[round(b) for b in box.tolist()]}")

    if detection_count == 0:
        print("No objects detected above threshold. Try lowering the threshold value.")
    else:
        print(f"\nTotal objects detected: {detection_count}")

    # Draw bounding boxes using matplotlib and display
    fig = draw_bounding_boxes_matplotlib(image, results, texts, threshold)
    output_path_plt = "output_with_boxes_matplotlib.png"
    fig.savefig(output_path_plt, dpi=100, bbox_inches='tight')
    print(f"Matplotlib visualization saved to: {output_path_plt}")

    # Show the plot (comment out if running in non-interactive environment)
    plt.show()

    return results


if __name__ == "__main__":
    # Run the detection
    results, image_with_boxes = main()