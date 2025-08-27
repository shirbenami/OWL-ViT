[# OWL-ViT

This project demonstrates how to use **OWL-ViT** (Open-Vocabulary Vision Transformer) for 
open-vocabulary object detection with text on **images and videos**.

OWL-ViT is a transformer-based model that can detect objects not only from a fixed set 
of categories but also from **arbitrary text queries** (e.g., `"laptop"`, `"coffee mug"`, `"chair"`), 
and even from **example images** (one-shot detection).

---

## Features
- Run OWL-ViT on **single images** or **video streams**.
- Support for **custom text prompts** (open-vocabulary queries).
- Support for **image-conditioned queries** (detect objects similar to a given crop).
- Automatic **bounding box visualization** with class labels and confidence scores.
- Works on local images/videos or webcam input.

---

##  Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/owlvit-demo.git
   cd owlvit-demo```
   
2. create a venv:
  ```bash
  python3 -m venv .env
  source .env/bin/activate```

3. Install dependencies:

  ```pip install -r requirements.txt```


📚 References
OWL-ViT: Simple Open-Vocabulary Object Detection with Vision Transformers - https://huggingface.co/docs/transformers/model_doc/owlvit
](https://huggingface.co/docs/transformers/model_doc/owlvit)
