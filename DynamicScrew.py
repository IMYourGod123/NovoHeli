import cv2
import numpy as np
import random
from tkinter import Tk, filedialog

# Upload image with a file picker
def upload_image():
    root = Tk()
    root.withdraw()  # Hide the main Tk window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if not file_path:
        print("❌ No file selected.")
        return None
    image = cv2.imread(file_path)
    if image is None:
        print("❌ Failed to load image.")
    return image

# Dynamic image processing
def process_dynamicImage(image):
    if image is None:
        return

    max_width = 800
    height, width = image.shape[:2]
    if width > max_width:
        scaling_factor = max_width / width
        image = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 200)
    thresh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    size_groups = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        for idx, (group_area, count) in enumerate(size_groups):
            if abs(group_area / count - area) <= 500:
                size_groups[idx] = (group_area + area, count + 1)
                break
        else:
            size_groups.append((area, 1))

    colors = [tuple(random.sample(range(256), 3)) for _ in size_groups]
    output = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        for idx, (group_area, count) in enumerate(size_groups):
            if abs(group_area / count - area) <= 800:
                cv2.drawContours(output, [contour], -1, colors[idx], 2)
                M = cv2.moments(contour)
                if M["m00"]:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.putText(output, chr(65 + idx), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
                break

    for i, (area, count) in enumerate(size_groups):
        mean_area = area / count
        cv2.putText(output, f"Size {chr(65+i)}: {count} (Mean Area: {mean_area:.1f})",
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

    cv2.imshow("Dynamic Object Classification", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image = upload_image()
    process_dynamicImage(image)

if __name__ == "__main__":
    main()
