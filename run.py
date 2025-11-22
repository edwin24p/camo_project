import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os

# <-- Choose your input method -->
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/Sepia_com%C3%BAn_%28Sepia_officinalis%29%2C_Parque_natural_de_la_Arr%C3%A1bida%2C_Portugal%2C_2020-07-21%2C_DD_62.jpg/1200px-Sepia_com%C3%BAn_%28Sepia_officinalis%29%2C_Parque_natural_de_la_Arr%C3%A1bida%2C_Portugal%2C_2020-07-21%2C_DD_62.jpg"

# Option 2: Use a local file path (uncomment to use)
# image_path = "path/to/your/image.jpg"

# Configuration
USE_URL = True  # Set to False to use local file path

# Choose detection method:
# 1 = GrabCut (best for clear foreground/background separation)
# 2 = Color-based segmentation (good for distinct colored animals)
# 3 = Advanced edge detection (best for textured animals)
METHOD = 1

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

def method1_grabcut(image):
    """
    Method 1: GrabCut Algorithm
    Best for images with clear foreground subject
    """
    print("Using GrabCut algorithm...")
    
    # Create a mask
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define rectangle around the probable foreground
    # (adjust these values based on where your animal is)
    h, w = image.shape[:2]
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    
    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify mask: 0 and 2 are background, 1 and 3 are foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask2 * 255

def method2_color_segmentation(image):
    """
    Method 2: Color-based segmentation
    Works well when animal has distinct color from background
    """
    print("Using color-based segmentation...")
    
    # Convert to HSV and LAB color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Calculate color variance to find regions of interest
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 10)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fill holes
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(morph)
    
    if contours:
        # Find largest contour (likely the animal)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    return mask

def method3_advanced_edges(image):
    """
    Method 3: Advanced edge detection with texture analysis
    Good for animals with texture details
    """
    print("Using advanced edge detection...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Multiple edge detection methods
    # Canny edges
    edges1 = cv2.Canny(enhanced, 30, 100)
    
    # Sobel edges
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    edges2 = np.uint8(np.sqrt(sobelx**2 + sobely**2))
    _, edges2 = cv2.threshold(edges2, 50, 255, cv2.THRESH_BINARY)
    
    # Combine edges
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Fill closed contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    
    # Draw filled contours above certain size
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > image.shape[0] * image.shape[1] * 0.01:  # At least 1% of image
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    # Clean up with morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return mask

def create_visualizations(image, mask):
    """Create various visualizations of the detection"""
    
    # Find contours for outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create outlined version
    outlined = image.copy()
    cv2.drawContours(outlined, contours, -1, (0, 255, 0), 3)
    
    # Create mask overlay
    overlay = image.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 200, 0]) * 0.5
    
    # Create segmented version (animal only)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    # Create white background version
    white_bg = np.ones_like(image) * 255
    white_bg[mask > 0] = image[mask > 0]
    
    return outlined, overlay, segmented, white_bg, contours

def main():
    print("="*60)
    print("Animal Detection and Outline Extraction")
    print("="*60)
    
    # Load image
    print("\n1. Loading image...")
    if USE_URL:
        print(f"   Source: URL")
        image = url_to_image(image_url)
    else:
        print(f"   Source: Local file")
        image = load_image_from_path(image_path)
    
    if image is None:
        print("Failed to load image")
        return
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Detect animal based on chosen method
    print(f"\n2. Detecting animal using Method {METHOD}...")
    
    if METHOD == 1:
        mask = method1_grabcut(image)
    elif METHOD == 2:
        mask = method2_color_segmentation(image)
    else:
        mask = method3_advanced_edges(image)
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    outlined, overlay, segmented, white_bg, contours = create_visualizations(image, mask)
    
    # Save outputs
    print("\n4. Saving outputs...")
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("outlined.png", outlined)
    cv2.imwrite("overlay.png", overlay)
    cv2.imwrite("segmented.png", segmented)
    cv2.imwrite("white_background.png", white_bg)
    print("   Saved: mask.png, outlined.png, overlay.png, segmented.png, white_background.png")
    
    # Create comprehensive visualization
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Detection Mask", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(cv2.cvtColor(outlined, cv2.COLOR_BGR2RGB))
    plt.title("Outlined Animal", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Mask Overlay", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Animal", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(cv2.cvtColor(white_bg, cv2.COLOR_BGR2RGB))
    plt.title("White Background", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Side by side comparison
    plt.subplot(2, 4, 7)
    comparison = np.hstack([
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(outlined, cv2.COLOR_BGR2RGB)
    ])
    plt.imshow(comparison)
    plt.title("Before & After", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Statistics
    plt.subplot(2, 4, 8)
    plt.axis('off')
    stats_text = f"Detection Statistics\n\n"
    stats_text += f"Method: {METHOD}\n"
    stats_text += f"Image size: {image.shape[1]}x{image.shape[0]}\n"
    stats_text += f"Detected pixels: {np.sum(mask > 0):,}\n"
    stats_text += f"Coverage: {np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100:.1f}%\n"
    stats_text += f"Contours found: {len(contours)}"
    plt.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig("detection_results.png", dpi=150, bbox_inches='tight')
    print("   Saved: detection_results.png")
    
    print("\n" + "="*60)
    print("Detection complete!")
    print("="*60)
    print("\nTips:")
    print("- Try different methods (METHOD = 1, 2, or 3) for best results")
    print("- Method 1 (GrabCut) works best for clear subjects")
    print("- Method 2 works well for color-distinct animals")
    print("- Method 3 is good for textured/detailed animals")

if __name__ == "__main__":
    main()