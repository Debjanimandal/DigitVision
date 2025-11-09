from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import io
from PIL import Image
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # Force TensorFlow to use legacy Keras
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
# Use TensorFlow's built-in Keras instead of standalone Keras
from tensorflow import keras
load_model = keras.models.load_model

app = Flask(__name__, template_folder='website', static_folder='website')
CORS(app)

# Rebuild model architecture from scratch and load weights
def load_legacy_model(model_path):
    """Rebuild model architecture and load weights from H5 file"""
    try:
        # Rebuild the exact model architecture
        model = keras.Sequential([
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(1, 1)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load weights from H5 file
        model.load_weights(model_path)
        return model
    except Exception as e:
        print(f"Failed to rebuild and load model: {e}")
        return None

# Load the trained model
base_dir = os.path.dirname(__file__)
candidate_names = [
    'mnist_cnn_model.h5',      # prefer .h5 if the user re-saved the model
    'mnist_cnn_model.keras'    # fallback to .keras
]

# Pick the first existing model path
MODEL_PATH = None
for name in candidate_names:
    path = os.path.join(base_dir, name)
    if os.path.isfile(path):
        MODEL_PATH = path
        break
if MODEL_PATH is None:
    # Default to .h5 path even if missing, to keep logs consistent
    MODEL_PATH = os.path.join(base_dir, candidate_names[0])

print(f"Looking for model. Resolved path: {MODEL_PATH}")
print(f"Model file exists: {os.path.isfile(MODEL_PATH)}")

try:
    model = load_legacy_model(MODEL_PATH)
    if model:
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
    else:
        print("✗ Failed to load model with compatibility loader")
        model = None
except Exception as e:
    print(f"✗ Error loading model from {MODEL_PATH}")
    print(f"✗ Error details: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    model = None

def preprocess_image(image):
    """
    Preprocess image for MNIST model prediction
    - Handles both uploaded images and canvas drawings
    - Converts to grayscale
    - Resizes to 28x28
    - Normalizes to 0-1
    - Ensures white digit on black background (MNIST format)
    """
    try:
        img = image.copy()
        print(f"DEBUG PREPROCESS: Input shape={img.shape}, dtype={img.dtype}")

        # Handle RGBA (canvas drawings come as RGBA with transparent background)
        if len(img.shape) == 3 and img.shape[2] == 4:
            # Extract alpha channel to find drawn areas
            alpha = img[:, :, 3]
            # Convert RGB channels to grayscale
            rgb = img[:, :, :3]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            
            # Create mask: where alpha > 0 (drawn) AND pixel is dark (black drawing)
            # Canvas has BLACK digit on TRANSPARENT background
            drawn_mask = alpha > 0
            dark_mask = gray < 128
            digit_mask = drawn_mask & dark_mask
            
            # Create output: white (255) where digit is drawn, black (0) elsewhere
            img = np.zeros_like(gray)
            img[digit_mask] = 255
            
            print(f"DEBUG PREPROCESS: Drawn pixels found: {np.sum(digit_mask)}")
        else:
            # If RGB -> grayscale
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # For uploaded images: determine if we need to invert
            # Check if the digit is darker or lighter than background
            # Apply Otsu's thresholding to separate foreground/background
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Count white vs black pixels
            white_pixels = np.sum(binary == 255)
            black_pixels = np.sum(binary == 0)
            
            # If more white pixels, the background is white (digit is black) - no inversion needed
            # If more black pixels, the background is black (digit is white) - already good
            # But we want WHITE digit on BLACK background for MNIST
            if white_pixels > black_pixels:
                # Background is white, digit is black - need to invert
                img = 255 - img
            
            print(f"DEBUG PREPROCESS: white_pixels={white_pixels}, black_pixels={black_pixels}, inverted={white_pixels > black_pixels}")

        print(f"DEBUG PREPROCESS: After grayscale - min={img.min()}, max={img.max()}, mean={img.mean()}")
        
        # Apply threshold to get clean binary image with automatic threshold
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        print(f"DEBUG PREPROCESS: After threshold - min={thresh.min()}, max={thresh.max()}, mean={thresh.mean()}")

        # Apply morphological closing to remove small noise
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find the bounding box of the digit (white pixels)
        coords = cv2.findNonZero(thresh)
        if coords is not None and len(coords) > 10:
            x, y, w, h = cv2.boundingRect(coords)
            digit = thresh[y:y+h, x:x+w]
            print(f"DEBUG PREPROCESS: Found digit at ({x},{y}) size {w}x{h}")
            
            # If the bounding box is too large (>80% of image), the digit is likely small and lost
            # Try to find a tighter bounding box by using a higher threshold
            img_h, img_w = thresh.shape
            if w > img_w * 0.8 or h > img_h * 0.8:
                print(f"DEBUG PREPROCESS: Bounding box too large ({w}x{h} vs {img_w}x{img_h}), trying adaptive threshold")
                # Use adaptive threshold to better separate small digit from background
                adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                coords2 = cv2.findNonZero(adaptive_thresh)
                if coords2 is not None and len(coords2) > 10:
                    x, y, w, h = cv2.boundingRect(coords2)
                    digit = adaptive_thresh[y:y+h, x:x+w]
                    print(f"DEBUG PREPROCESS: Found tighter bounding box at ({x},{y}) size {w}x{h}")
            
            # Apply morphological operations to thicken thin strokes
            kernel = np.ones((2,2), np.uint8)
            digit = cv2.dilate(digit, kernel, iterations=1)
            
            # Enhance contrast - stretch to full range [0, 255]
            digit_min = digit.min()
            digit_max = digit.max()
            if digit_max > digit_min:
                digit = ((digit - digit_min) / (digit_max - digit_min) * 255).astype(np.uint8)
                print(f"DEBUG PREPROCESS: Enhanced contrast - old range [{digit_min},{digit_max}], new range [0,255]")
            else:
                print(f"DEBUG PREPROCESS: WARNING - No contrast to enhance (all pixels same value: {digit_min})")
        else:
            print("DEBUG PREPROCESS: No digit found in image")
            # Return empty canvas if no digit found
            canvas = np.zeros((28, 28, 1), dtype=np.float32)
            return canvas

        # Resize the cropped digit to fit in 20x20 box (with aspect ratio preserved)
        h_d, w_d = digit.shape
        if h_d > w_d:
            new_h = 20
            new_w = max(1, int(round((w_d * 20) / h_d)))
        else:
            new_w = 20
            new_h = max(1, int(round((h_d * 20) / w_d)))
        
        digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center the digit in a 28x28 black canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

        # Normalize to [0, 1]
        canvas = canvas.astype("float32") / 255.0

        # Reshape to (28, 28, 1)
        canvas = canvas.reshape(28, 28, 1)

        # Save debug image (for visual check)
        cv2.imwrite("debug_preprocessed.png", (canvas * 255).astype(np.uint8))
        print(f"DEBUG PREPROCESS: Final - min={float(canvas.min()):.3f}, max={float(canvas.max()):.3f}, mean={float(canvas.mean()):.3f}")

        return canvas

    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        import traceback
        traceback.print_exc()
        canvas = np.zeros((28, 28, 1), dtype=np.float32)
        return canvas
@app.route('/')
def index():
    """Serve the layout page"""
    return render_template('layout.html')

@app.route('/layout.html')
def layout():
    """Serve the layout page"""
    return render_template('layout.html')

@app.route('/sign.html')
def sign():
    """Serve the sign in/up page"""
    return render_template('sign.html')

@app.route('/home.html')
def home():
    """Serve the home page"""
    return render_template('home.html')

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    """
    Handle image upload prediction
    Expects: multipart/form-data with 'image' file
    Returns: JSON with prediction
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get the uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # Preprocess
        processed_image = preprocess_image(image_array)
        
        # Reshape for model (add batch dimension)
        input_data = processed_image.reshape(1, 28, 28, 1)
        
        # Predict
        prediction = model.predict(input_data, verbose=0)
        print("DEBUG INPUT SHAPE:", input_data.shape)
        print("DEBUG INPUT STATS: min=", float(input_data.min()), "max=", float(input_data.max()), "mean=", float(input_data.mean()))
        print("DEBUG MODEL OUTPUT:", prediction)

        predicted_digit = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100
        
        return jsonify({
            'prediction': int(predicted_digit),
            'confidence': round(confidence, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    """
    Handle canvas drawing prediction
    Expects: JSON with 'image' as base64 data URL
    Returns: JSON with prediction
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode and convert to numpy array
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        print(f"DEBUG CANVAS: Original shape={image_array.shape}, dtype={image_array.dtype}")
        print(f"DEBUG CANVAS: Image stats - min={image_array.min()}, max={image_array.max()}, mean={image_array.mean()}")
        
        # DON'T convert RGBA to RGB - let preprocess_image handle it
        # The preprocessing function needs the alpha channel to detect drawn areas
        
        # Preprocess
        processed_image = preprocess_image(image_array)
        
        print(f"DEBUG CANVAS: After preprocess - shape={processed_image.shape}")
        print(f"DEBUG CANVAS: After preprocess - min={processed_image.min()}, max={processed_image.max()}, mean={processed_image.mean()}")
        
        # Reshape for model (add batch dimension)
        input_data = processed_image.reshape(1, 28, 28, 1)
        
        # Predict
        prediction = model.predict(input_data, verbose=0)
        print(f"DEBUG CANVAS: Model prediction={prediction}")
        
        predicted_digit = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100
        
        print(f"DEBUG CANVAS: Final prediction={predicted_digit}, confidence={confidence}")
        
        return jsonify({
            'prediction': int(predicted_digit),
            'confidence': round(confidence, 2)
        })
    
    except Exception as e:
        print(f"ERROR in predict_canvas: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/<path:filename>')
def serve_files(filename):
    """Serve static files (images, css, js, etc.) from website folder"""
    import os
    file_path = os.path.join('website', filename)
    if os.path.isfile(file_path):
        from flask import send_file
        return send_file(file_path)
    return "File not found", 404

if __name__ == '__main__':
    if model is None:
        print("⚠ Warning: Model could not be loaded. Predictions will fail.")
    print("🚀 Starting DigitVision server on http://localhost:5000")
    app.run(debug=True, port=5000)
