# app.py
# 🧠 Project Goal Reminder
# Upload or load an image + candidate captions → 
# find the best-matching caption using CLIP (image + text embeddings + cosine similarity).

# Main logic (load image, captions, print best match)

from utils import load_image, get_image_embedding, get_text_embeddings, find_best_caption

# === Configuration ===
image_path = "assets/dogs.jpg"  # Replace with your test image path
captions = [
    "A dog playing in the grass",
    "A cat on a couch",
    "A child flying a kite",
    "A car parked on the street",
    "Two dogs are playing together",
    "Two puppies are playing together",
]

# === Process ===
image = load_image(image_path)
image_vector = get_image_embedding(image)
text_vectors = get_text_embeddings(captions)
best_caption, score = find_best_caption(image_vector, text_vectors, captions)

# === Result ===
print(f"Best Matching Caption:\n→ {best_caption} (Similarity Score: {score:.3f})")