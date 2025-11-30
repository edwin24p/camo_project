import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path

# <-- Choose your input method -->
image_url = "https://www.osc.org/wp-content/uploads/2019/06/Add-a-heading-55.png"

# Option 2: Use a local file path (uncomment to use)
# image_path = "/home/maudie/eecs442_final/imgs/giraffe-circled.jpg"

# Configuration
USE_URL = True  # Set to False to use local file path
USE_GPU = True  # Set to False to use CPU instead

# Detection settings
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to show detection (0.0-1.0)
IOU_THRESHOLD = 0.45  # Overlap threshold for removing duplicate detections

# IMPORTANT: Use segmentation model with '-seg' suffix!
MODEL_SIZE = 'yolov8x-seg'  # Options: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg

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

def load_yolo_model(model_size='yolov8n-seg', use_gpu=True):
    """Load YOLOv8 segmentation model"""
    try:
        from ultralytics import YOLO
        
        print(f"\n📦 Loading YOLOv8-Segmentation model ({model_size})...")
        print("   (First run will download the model - ~7MB to 140MB depending on size)")
        
        # Load model
        model = YOLO(f'{model_size}.pt')
        
        # Set device
        if use_gpu:
            device = 'cuda:0' if check_gpu_availability() else 'cpu'
        else:
            device = 'cpu'
            print("   Using CPU as requested")
        
        model.to(device)
        print(f"✓ Segmentation model loaded on {device.upper()}")
        
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
    """Detect animals in image using YOLO with segmentation masks"""
    
    # Run inference with segmentation
    print("\n🔍 Running detection with segmentation...")
    results = model(image, conf=confidence_threshold, iou=iou_threshold, verbose=False)
    
    # Get the first result (single image)
    result = results[0]
    
    # Extract detections
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
    
    # Get segmentation masks
    masks = None
    if hasattr(result, 'masks') and result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        print(f"   ✓ Segmentation masks detected for {len(masks)} objects")
    else:
        print("   ⚠ No segmentation masks available - model may not be a segmentation model")
    
    # Get class names
    class_names = result.names
    
    detections = []
    for i in range(len(boxes)):
        class_id = class_ids[i]
        class_name = class_names[class_id]
        confidence = confidences[i]
        box = boxes[i]
        
        # Get mask if available
        mask = masks[i] if masks is not None else None
        
        detections.append({
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'box': box,
            'mask': mask
        })
    
    return detections, result

def create_segmentation_mask(image, detections):
    """Create a segmentation mask from detections using actual mask contours"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for det in detections:
        if det['mask'] is not None:
            # Resize mask to image dimensions
            det_mask = cv2.resize(det['mask'], (image.shape[1], image.shape[0]))
            # Convert to binary and add to overall mask
            binary_mask = (det_mask > 0.5).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, binary_mask)
        else:
            # Fallback to bounding box if no mask available
            x1, y1, x2, y2 = det['box'].astype(int)
            mask[y1:y2, x1:x2] = 255
    
    # Clean up mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return mask

def draw_detections(image, detections):
    """Draw segmentation contours and labels on image - NO BOXES!"""
    annotated = image.copy()
    
    # Generate colors for different classes
    np.random.seed(42)
    colors = {}
    
    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']
        
        # Get or create color for this class
        if class_name not in colors:
            colors[class_name] = tuple(np.random.randint(50, 255, 3).tolist())
        color = colors[class_name]
        
        # Draw segmentation contour if available
        if det['mask'] is not None:
            # Resize mask to image dimensions
            det_mask = cv2.resize(det['mask'], (image.shape[1], image.shape[0]))
            binary_mask = (det_mask > 0.5).astype(np.uint8)
            
            # Find contours of the mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw thick contour following the animal's edges
            cv2.drawContours(annotated, contours, -1, color, 4)
            
            # Optional: fill with semi-transparent color
            overlay = annotated.copy()
            cv2.drawContours(overlay, contours, -1, color, -1)
            annotated = cv2.addWeighted(annotated, 0.8, overlay, 0.2, 0)
            
        else:
            # Fallback to bounding box if no mask (shouldn't happen with -seg model)
            print(f"   ⚠ Warning: No mask for {class_name}, using box")
            x1, y1, x2, y2 = det['box'].astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Create label
        label = f"{class_name}: {confidence:.2f}"
        x1, y1 = det['box'][:2].astype(int)
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(annotated, (x1, y1 - text_height - 15), 
                     (x1 + text_width + 10, y1), color, -1)
        
        # Draw text
        cv2.putText(annotated, label, (x1 + 5, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return annotated

def create_outlined_version(image, mask):
    """Create version with green outline following edges"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = image.copy()
    cv2.drawContours(outlined, contours, -1, (0, 255, 0), 4)
    return outlined

def create_visualizations(image, detections):
    """Create all visualization outputs"""
    
    # Annotated image with segmentation contours
    annotated = draw_detections(image, detections)
    
    # Segmentation mask
    mask = create_segmentation_mask(image, detections)
    
    # Outlined version (green contours only)
    outlined = create_outlined_version(image, mask)
    
    # Overlay
    overlay = image.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 200, 0]) * 0.5
    
    # Segmented (black background)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    return annotated, outlined, overlay, segmented

def main():
    print("="*70)
    print("🤖 GPU-Accelerated Animal Detection with Edge Segmentation")
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
        mask_status = "with segmentation" if det['mask'] is not None else "box only"
        print(f"   {i}. {det['class_name']}: {det['confidence']:.1%} confidence ({mask_status})")
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    annotated, outlined, overlay, segmented = create_visualizations(image, detections)
    
    # Save outputs
    print("\n5. Saving outputs...")
    
    cv2.imwrite("1_original.png", image)
    print("   ✓ 1_original.png")
    
    cv2.imwrite("2_detected.png", annotated)
    print("   ✓ 2_detected.png (segmentation contours following edges)")
    
    cv2.imwrite("3_outlined.png", outlined)
    print("   ✓ 3_outlined.png (green outline)")
    
    cv2.imwrite("4_overlay.png", overlay)
    print("   ✓ 4_overlay.png (semi-transparent)")
    
    cv2.imwrite("5_segmented.png", segmented)
    print("   ✓ 5_segmented.png (black background)")
    
    # Create comprehensive visualization
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("1. Original Image", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("2. AI Segmentation (Edge-Following)", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(outlined, cv2.COLOR_BGR2RGB))
    plt.title("3. Outlined", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("4. Overlay", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.title("5. Segmented", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Statistics - LARGER TEXT
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    model_name = MODEL_SIZE.replace('-seg', '').upper()
    
    stats_text = "DETECTION STATISTICS\n"
    stats_text += "=" * 40 + "\n\n"
    stats_text += f"Model: YOLOv8-{model_name[-1]}-SEG\n"
    stats_text += f"Device: {device.upper()}\n\n"
    stats_text += f"Image Size:\n"
    stats_text += f"{image.shape[1]} x {image.shape[0]} pixels\n\n"
    stats_text += f"Total Detections: {len(detections)}\n\n"
    
    stats_text += "DETECTED OBJECTS:\n"
    stats_text += "-" * 40 + "\n"
    
    for i, det in enumerate(detections[:8], 1):  # Show up to 8
        stats_text += f"{i}. {det['class_name'].upper()}\n"
        stats_text += f"   Confidence: {det['confidence']:.1%}\n"
        if det['mask'] is not None:
            stats_text += f"   Status: Edge-Following ✓\n\n"
        else:
            stats_text += f"   Status: Box Only\n\n"
    
    if len(detections) > 8:
        stats_text += f"... and {len(detections)-8} more\n"
    
    stats_text += "=" * 40
    
    plt.text(0.05, 0.5, stats_text, fontsize=13, family='monospace',
             verticalalignment='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=1))
    
    plt.tight_layout()
    plt.savefig("6_results_grid.png", dpi=150, bbox_inches='tight')
    print("   ✓ 6_results_grid.png (complete visualization)")
    
    print("\n" + "="*70)
    print("✅ Detection Complete!")
    print("="*70)
    
    # Print tips
    print("\n💡 Tips:")
    print("   • Lower CONFIDENCE_THRESHOLD (e.g., 0.15) to detect more objects")
    print("   • Segmentation models (-seg) follow exact edges, not boxes!")
    print("   • Use MODEL_SIZE='yolov8x-seg' for best accuracy (slower)")
    print("   • Use MODEL_SIZE='yolov8n-seg' for fastest speed (less accurate)")
        
    if has_gpu:
        print(f"\n🚀 GPU acceleration: ENABLED")
    else:
        print(f"\n🐌 GPU acceleration: DISABLED (using CPU)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
