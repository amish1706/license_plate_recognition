# License Plate Recognition

### Setting Up

---

Download the weights for Yolov5 and Low-Light Image Enhancement model from [here](https://drive.google.com/drive/folders/1DjcsQ3I0oPs1bIMyb5rX4W8N4skG_gXm?usp=sharing).

and create a folder “weights” and put them inside.

```python
git clone https://github.com/amish1706/license_plate_recognition
```

# Inference Script

```python
python pipeline.py <image_path>
```

### Required Arguments:

- `<image_path>`

### Options:

- `--save` : Save the annotated image in output directory, i.e. outputs
- `--show`: View the annotated image
- `--llie` : Apply low-light image enhancement (if your image is shot in low light, set this to True)
- `--out_dir`: Output directory for saving results
- `--model`: TrOCR model used for OCR (default : trocr-small-printed) Check [available](https://huggingface.co/models?search=troc) models

---
# Gradio Local Deployment

```python
python deploy.py
```
