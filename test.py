"""
GPU-Accelerated Animal Detection using YOLOv8
This uses a pre-trained deep learning model that runs on your GPU for fast, accurate detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path

# <-- Choose your input method -->
image_url = "https://static.boredpanda.com/blog/wp-content/uploads/2015/07/da-vinci-horse-pattern-north-yorkshire-coverimage.jpg"
# Option 2: Use a local file path (uncomment to use)
# image_path = "path/to/your/image.jpg"

# Configuration
USE_URL = True  # Set to False to use local file path
USE_GPU = True  # Set to False to use CPU instead

# Detection settings
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to show detection (0.0-1.0)
IOU_THRESHOLD = 0.45  # Overlap threshold for removing duplicate detections

# Model selection: 'yolov8n' (fastest), 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x' (most accurate)
MODEL_SIZE = 'yolov8x'  # Use 'x' for best accuracy, 'n' for speed

def check_gpu_availability():
    """Check if GPU is available for PyTorch"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU detected: {gpu_name}")
            print(f"  Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠ No GPU detected. Will use CPU (slower)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed. Install with: pip install torch torchvision")
        return False

def load_image_from_path(path):
    """Load image from local file path"""
    try:
        image = cv2.imread(path)
        if image is None:
            print(f"Failed to load image from: {path}")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def url_to_image(url):
    """Download the image, convert it to a NumPy array, and decode"""
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        resp = urllib.request.urlopen(req)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def load_yolo_model(model_size='yolov8n', use_gpu=True):
    """Load YOLOv8 model"""
    try:
        from ultralytics import YOLO
        
        print(f"\n📦 Loading YOLOv8 model ({model_size})...")
        print("   (First run will download the model - ~6MB to 138MB depending on size)")
        
        # Load model
        model = YOLO(f'{model_size}.pt')
        
        # Set device
        if use_gpu:
            device = 'cuda:0' if check_gpu_availability() else 'cpu'
        else:
            device = 'cpu'
            print("   Using CPU as requested")
        
        model.to(device)
        print(f"✓ Model loaded on {device.upper()}")
        
        return model, device
        
    except ImportError:
        print("\n❌ Ultralytics YOLOv8 not installed!")
        print("\nInstall with:")
        print("  pip install ultralytics")
        print("\nFor GPU support, also install:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def detect_animals(image, model, confidence_threshold=0.25, iou_threshold=0.45):
    """Detect animals in image using YOLO"""
    
    # Run inference
    print("\n🔍 Running detection...")
    results = model(image, conf=confidence_threshold, iou=iou_threshold, verbose=False)
    
    # Get the first result (single image)
    result = results[0]
    
    # Extract detections
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
    
    # Get class names
    class_names = result.names
    
    # Filter for animals only (COCO dataset animal classes)
    animal_classes = {
        0: 'person', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 
        18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 
        23: 'giraffe', 24: 'backpack'
    }
    
    # Actually, let's keep all living things for broader detection
    # COCO has limited animal classes, so we'll show all detections
    
    detections = []
    for i in range(len(boxes)):
        class_id = class_ids[i]
        class_name = class_names[class_id]
        confidence = confidences[i]
        box = boxes[i]
        
        detections.append({
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'box': box
        })
    
    return detections, result

def create_segmentation_mask(image, detections):
    """Create a segmentation mask from detections"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for det in detections:
        x1, y1, x2, y2 = det['box'].astype(int)
        # Fill the bounding box area
        mask[y1:y2, x1:x2] = 255
    
    # Clean up mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask

def draw_detections(image, detections):
    """Draw bounding boxes and labels on image"""
    annotated = image.copy()
    
    # Generate colors for different classes
    np.random.seed(42)
    colors = {}
    
    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']
        x1, y1, x2, y2 = det['box'].astype(int)
        
        # Get or create color for this class
        if class_name not in colors:
            colors[class_name] = tuple(np.random.randint(0, 255, 3).tolist())
        color = colors[class_name]
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Create label
        label = f"{class_name}: {confidence:.2f}"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(annotated, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(annotated, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated

def create_outlined_version(image, mask):
    """Create version with green outline"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = image.copy()
    cv2.drawContours(outlined, contours, -1, (0, 255, 0), 3)
    return outlined

def create_visualizations(image, detections):
    """Create all visualization outputs"""
    
    # Annotated image with bounding boxes
    annotated = draw_detections(image, detections)
    
    # Segmentation mask
    mask = create_segmentation_mask(image, detections)
    
    # Outlined version
    outlined = create_outlined_version(image, mask)
    
    # Overlay
    overlay = image.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 200, 0]) * 0.5
    
    # Segmented (black background)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    # White background
    white_bg = np.ones_like(image) * 255
    white_bg[mask > 0] = image[mask > 0]
    
    return annotated, mask, outlined, overlay, segmented, white_bg

def main():
    print("="*70)
    print("🤖 GPU-Accelerated Animal Detection using YOLOv8")
    print("="*70)
    
    # Check GPU
    print("\n1. Checking hardware...")
    has_gpu = check_gpu_availability()
    if not has_gpu and USE_GPU:
        print("   Continuing with CPU...")
    
    # Load model
    print("\n2. Loading AI model...")
    model, device = load_yolo_model(MODEL_SIZE, USE_GPU)
    
    if model is None:
        print("\n❌ Failed to load model. Please install required packages.")
        return
    
    # Load image
    print("\n3. Loading image...")
    if USE_URL:
        print("   Source: URL")
        image = url_to_image(image_url)
    else:
        print("   Source: Local file")
        image = load_image_from_path(image_path)
    
    if image is None:
        print("❌ Failed to load image")
        return
    
    print(f"   ✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Detect animals
    detections, result = detect_animals(image, model, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
    
    print(f"\n✓ Detection complete!")
    print(f"   Found {len(detections)} object(s)")
    
    if len(detections) == 0:
        print("\n⚠ No objects detected with current confidence threshold.")
        print(f"   Try lowering CONFIDENCE_THRESHOLD (current: {CONFIDENCE_THRESHOLD})")
        return
    
    # Print detections
    print("\n📋 Detected objects:")
    for i, det in enumerate(detections, 1):
        print(f"   {i}. {det['class_name']}: {det['confidence']:.1%} confidence")
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    annotated, mask, outlined, overlay, segmented, white_bg = create_visualizations(
        image, detections)
    
    # Save outputs
    print("\n5. Saving outputs...")
    
    cv2.imwrite("1_original.png", image)
    print("   ✓ 1_original.png")
    
    cv2.imwrite("2_detected.png", annotated)
    print("   ✓ 2_detected.png (bounding boxes)")
    
    cv2.imwrite("3_mask.png", mask)
    print("   ✓ 3_mask.png (segmentation mask)")
    
    cv2.imwrite("4_outlined.png", outlined)
    print("   ✓ 4_outlined.png (green outline)")
    
    cv2.imwrite("5_overlay.png", overlay)
    print("   ✓ 5_overlay.png (semi-transparent)")
    
    cv2.imwrite("6_segmented.png", segmented)
    print("   ✓ 6_segmented.png (black background)")
    
    cv2.imwrite("7_white_bg.png", white_bg)
    print("   ✓ 7_white_bg.png (white background)")
    
    # Create comprehensive visualization
    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("1. Original Image", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("2. AI Detection", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("3. Segmentation Mask", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(cv2.cvtColor(outlined, cv2.COLOR_BGR2RGB))
    plt.title("4. Outlined", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("5. Overlay", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.title("6. Segmented (Black)", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow(cv2.cvtColor(white_bg, cv2.COLOR_BGR2RGB))
    plt.title("7. Segmented (White)", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Statistics
    plt.subplot(2, 4, 8)
    plt.axis('off')
    
    stats_text = "Detection Statistics\n"
    stats_text += "=" * 35 + "\n\n"
    stats_text += f"Model: YOLOv8-{MODEL_SIZE[-1].upper()}\n"
    stats_text += f"Device: {device.upper()}\n\n"
    stats_text += f"Image: {image.shape[1]}x{image.shape[0]}\n\n"
    stats_text += f"Detections: {len(detections)}\n\n"
    
    for i, det in enumerate(detections[:5], 1):  # Show top 5
        stats_text += f"{i}. {det['class_name']}\n"
        stats_text += f"   {det['confidence']:.1%}\n"
    
    if len(detections) > 5:
        stats_text += f"\n... and {len(detections)-5} more\n"
    
    stats_text += "\n" + "=" * 35
    
    plt.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("8_results_grid.png", dpi=150, bbox_inches='tight')
    print("   ✓ 8_results_grid.png (complete visualization)")
    
    print("\n" + "="*70)
    print("✅ Detection Complete!")
    print("="*70)
    
    # Print tips
    print("\n💡 Tips:")
    print("   • Lower CONFIDENCE_THRESHOLD (e.g., 0.15) to detect more objects")
    print("   • Use MODEL_SIZE='yolov8x' for best accuracy (slower)")
    print("   • Use MODEL_SIZE='yolov8n' for fastest speed (less accurate)")
    print("   • YOLOv8 is trained on COCO dataset (80 classes)")
    print("   • For marine animals, results may vary as COCO has limited classes")
    
    print("\n📦 Model sizes:")
    print("   yolov8n: ~6MB   (fastest, least accurate)")
    print("   yolov8s: ~22MB")
    print("   yolov8m: ~52MB")
    print("   yolov8l: ~87MB")
    print("   yolov8x: ~138MB (slowest, most accurate)")
    
    if has_gpu:
        print(f"\n🚀 GPU acceleration: ENABLED")
    else:
        print(f"\n🐌 GPU acceleration: DISABLED (using CPU)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()