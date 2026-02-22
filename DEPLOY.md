# AgriVision AI — Deployment Guide

## Before You Deploy

1. **Add `best_model.pth`**  
   Place your trained model at `model/best_model.pth`.  
   The app will not run without it.

2. **Confirm class order**  
   The 17 classes in `data/disease_data.py` must match training order.

---

## Streamlit Cloud Deployment

1. **Push to GitHub**
   - Create a new repo
   - Push this project (including `best_model.pth`)

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - New app → choose your repo
   - Main file: `app.py`
   - Save

3. **Wait**  
   First run can take a few minutes (PyTorch installs).

---

## Local Run

```bash
cd Agri-Vision_AI
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## If `best_model.pth` is Too Large for GitHub

- GitHub limit: 100 MB per file
- Options:
  1. Use [Git LFS](https://git-lfs.github.com) for the `.pth` file
  2. Host the file elsewhere and download it at app startup (e.g. via `streamlit run` or an init script)
