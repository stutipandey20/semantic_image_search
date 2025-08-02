# 🎯 CLIP Caption Matcher

A mini project that uses OpenAI's CLIP model to match images with the most relevant captions. This is a local, API-free project using open-source models from Hugging Face.

---

## 🚀 Features

- Upload an image and input multiple candidate captions
- Uses CLIP to encode both image and text in the same vector space
- Computes cosine similarity to find the best matching caption
- Lightweight and runs entirely on your local machine
- Extendable to do reverse image search, image tagging, or zero-shot classification

---

## 🛠️ Tech Stack

- Python 3.x
- [Transformers](https://huggingface.co/docs/transformers/index) (Hugging Face)
- PyTorch
- PIL (Pillow)
- scikit-learn

---

## 🖼️ Example

Image: test.jpg
Captions:

A dog playing in the park
A cat on a couch
A child flying a kite

🔍 Output:
Best Matching Caption → A dog playing in the park

---

## ✅ Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/clip_caption_matcher.git
cd clip_caption_matcher

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```
---

## 📦 Dependencies
To regenerate requirements.txt:

pip freeze > requirements.txt

---

## 📌 Notes
This project uses the openai/clip-vit-base-patch32 model from Hugging Face.

No API keys or internet access required after initial model download.

Ideal for learning about embeddings, cosine similarity, and multi-modal AI.

---

## 📄 License
MIT License. Feel free to use and modify.

---