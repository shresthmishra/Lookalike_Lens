# Lookalike Lens - A Visual Product Matcher

Lookalike Lens is a full-stack web application that allows users to find visually similar products by uploading an image or providing an image URL.

**Live Demo:** [**lookalike-lens.vercel.app**](https://lookalike-lens.vercel.app)

---

## ✨ Features

* **Image Upload:** Supports both direct file uploads and image URLs as input.
* **Visual Search:** Utilizes a Vision Transformer (ViT) AI model to generate vector embeddings and find similar products.
* **Product Results:** Displays results in a clean, responsive grid with similarity scores.
* **Dynamic Filtering:** Allows users to filter the results in real-time based on a similarity threshold.
* **Mobile Responsive Design:** A modern UI that works beautifully on both desktop and mobile devices.

## 🛠️ Tech Stack

* **Frontend:** Vite, React, Material-UI (MUI)
* **Backend:** Python, FastAPI
* **Databases:**
    * **PostgreSQL (Render):** For storing product metadata.
    * **Milvus (Zilliz Cloud):** For high-speed vector similarity search.
* **AI Model:** `google/vit-base-patch32-224-in21k`
* **Deployment:**
    * Frontend on Vercel
    * Backend on Hugging Face Spaces

## 🚀 Running Locally

### Backend

1.  Navigate to the `backend` directory.
2.  Create and activate a virtual environment: `python -m venv venv` and `source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows).
3.  Install dependencies: `pip install -r requirements.txt`.
4.  Run the server: `uvicorn main:app --reload`.

### Frontend

1.  Navigate to the `frontend` directory.
2.  Install dependencies: `npm install`.
3.  Run the development server: `npm run dev`.

