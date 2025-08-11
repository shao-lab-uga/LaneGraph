import cv2
import numpy as np
import os

# ---------- Configuration ----------
image_folder = "raw_data/UGA_intersections"  # <- set your folder path
output_mask_folder = "raw_data/UGA_intersections_masks"  # <- output folder for masks
mask_color = 255  # white (binary mask)
brush_size = 20
# -----------------------------------

os.makedirs(output_mask_folder, exist_ok=True)

drawing = False
ix, iy = -1, -1
current_mask = None
current_image = None

def draw_mask(event, x, y, flags, param):
    global drawing, ix, iy, current_mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(current_mask, (x, y), brush_size, mask_color, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(current_mask, (x, y), brush_size, mask_color, -1)

def process_images(image_folder):
    global current_mask, current_image

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        current_image = cv2.imread(image_path)
        if current_image is None:
            print(f"Failed to load {image_path}")
            continue

        h, w = current_image.shape[:2]
        current_mask = np.zeros((h, w), dtype=np.uint8)

        cv2.namedWindow("Draw Mask")
        cv2.setMouseCallback("Draw Mask", draw_mask)

        while True:
            display = current_image.copy()
            mask_overlay = cv2.merge([current_mask]*3)
            display = cv2.addWeighted(display, 0.7, mask_overlay, 0.3, 0)
            cv2.imshow("Draw Mask", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Save mask
                mask_path = os.path.join(output_mask_folder, f"{os.path.splitext(image_name)[0]}_mask.png")
                cv2.imwrite(mask_path, current_mask)
                print(f"Saved mask to {mask_path}")
                break
            elif key == ord('q'):
                print(f"Skipped {image_name}")
                break
            elif key == ord('c'):
                current_mask[:] = 0  # Clear mask

    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_images(image_folder)
