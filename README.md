


# 🔢 DigitVision — Handwritten Digit Recognition Web App

DigitVision is a **full-stack deep learning web application** that recognizes handwritten digits (0–9) in real time using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.

It combines the power of **TensorFlow/Keras** for digit recognition with a modern **Flask + HTML/CSS/JS** frontend for an interactive experience.

---

## 🌐 Live Demo

🚀 Try it now: https://huggingface.co/spaces/debjani31/DigitVision

*(Draw any digit from 0–9 and see real-time predictions!)*

---

## 📖 Overview

DigitVision demonstrates how **deep learning** and **web development** can merge into a powerful AI application.  
It uses a CNN model trained with **~99% accuracy** on the MNIST dataset and deploys it through **Flask**, wrapped in a visually engaging user interface.

---

## ✨ Features

✅ Real-time digit prediction through canvas drawing  
✅ Upload your own digit images for recognition  
✅ Interactive tools (pencil, eraser, clear canvas)  
✅ Confidence score display for predictions  
✅ Clean neon-themed UI with animations  
✅ User Sign-in / Sign-up pages (UI ready for backend integration)

---

## 🧠 Model Details

- **Dataset:** [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- **Input Shape:** `28 × 28 × 1` grayscale images  
- **Architecture:**
  - Conv2D (32) → Conv2D (32) → MaxPooling2D → Dropout  
  - Conv2D (64) → Conv2D (64) → MaxPooling2D → Dropout  
  - Flatten → Dense(256) → Dropout → Dense(10, Softmax)
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Accuracy:** ~99% (validation)  

Model trained on Kaggle and saved as `mnist_cnn_model.h5` for deployment.

---

## 🧩 Tech Stack

| Layer | Technology |
|--------|-------------|
| **Frontend** | HTML, CSS, JavaScript, Bootstrap |
| **Backend** | Flask (Python) |
| **Deep Learning** | TensorFlow, Keras |
| **Image Processing** | OpenCV, Pillow |
| **Deployment** | Hugging Face Spaces (Docker + Python) |

---

## 🖥️ Installation (for local use)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/debjani31/DigitVision.git
cd DigitVision
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App

```bash
python app.py
```

Access it at → [http://localhost:5000](http://localhost:5000)

---

## 🧾 How to Use

1. Launch the app or visit the [live Hugging Face demo](https://huggingface.co/spaces/debjani31/DigitVision)
2. **Sign in or Sign up** (UI available)
3. Use the **canvas** to draw a digit or **upload** one
4. Click **Predict Digit**
5. View predicted number + confidence score instantly 🎯

---

## 📂 Project Structure

```
DigitVision/
│
├── app.py                   # Flask backend & model routes
├── mnist_cnn_model.h5       # Trained CNN model
├── requirements.txt          # Dependencies
├── website/                  # Frontend HTML templates
│   ├── layout.html
│   ├── home.html
│   ├── sign.html
│   └── static/ (CSS, JS, images)
└── README.md
```

---

## 🧠 Model Pipeline

1. User draws or uploads a digit
2. Image is sent to Flask backend
3. Preprocessed with OpenCV:

   * Grayscale conversion
   * Inversion (white digit on black background)
   * Cropping and centering
   * Resizing to 28×28
   * Normalization (0–1)
4. Model predicts the digit and confidence
5. Result displayed in the frontend instantly

---


---

## 🔮 Future Improvements

* 🧩 Add Firebase or SQL-based authentication
* 📱 Make fully responsive for mobile devices
* 🧠 Train on EMNIST for letters + digits
* ☁️ Auto-deploy with CI/CD (GitHub Actions)

---

## 👩‍💻 Author

**Debjani Mandal**
🎓 Machine Learning & Web Developer Enthusiast
🌐 [Live App on Hugging Face](https://huggingface.co/spaces/debjani31/DigitVision)
💻 Project: *DigitVision – Handwritten Digit Recognition using CNN*

---

## 🪄 License

This project is licensed under the **MIT License**.
Feel free to fork, modify, and build upon it for your own learning or research.

