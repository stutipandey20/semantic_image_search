# utils.py
# Helper functions (embed image, embed text, similarity)

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the model and processor once
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

def load_image(image_path):
    """Open an image file."""
    return Image.open(image_path).convert("RGB")

def get_image_embedding(image):
    """Return normalized image embedding."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

def get_text_embeddings(texts):
    """Return normalized embeddings for list of texts."""
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

def find_best_caption(image_vector, text_vectors, captions):
    """Return caption with highest cosine similarity to the image."""
    scores = cosine_similarity(image_vector, text_vectors)
    best_idx = scores[0].argmax()
    return captions[best_idx], scores[0][best_idx]