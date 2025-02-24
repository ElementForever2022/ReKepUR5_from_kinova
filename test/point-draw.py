import cv2
import os
import sys
import numpy as np

points = []
counter = 0
drawing = True
scale_factor = 1.0

def resize_image(image, max_width=1280, max_height=720):
    global scale_factor
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        scale_factor = scale
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def mouse_callback(event, x, y, flags, param):
    global counter, drawing, scale_factor
    if event == cv2.EVENT_LBUTTONDOWN and drawing:

        original_x = int(x / scale_factor)
        original_y = int(y / scale_factor)
        points.append((original_x, original_y))
        
        text = str(counter)
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        text_x = x - text_size[0] // 2
        text_y = y - text_size[1] // 2

        cv2.rectangle(param, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), cv2.FILLED)
        cv2.putText(param, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)

        counter += 1
        cv2.imshow("Image", param)

if __name__ == "__main__":
    if len(sys.argv) < 2:   
        print("Usage: python point-draw.py <image_name>")
        sys.exit(1)

    img_name = sys.argv[1]
    base_path = r'D:\ROBO\ReKep'
    image_path = os.path.join(base_path, f"{img_name}")

    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)

    original_image = cv2.imread(image_path)
    display_image = resize_image(original_image.copy())

    cv2.imshow("Image", display_image)
    cv2.setMouseCallback("Image", mouse_callback, display_image)

    print("Press 'Enter' to finish drawing and save the image.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            drawing = False
            break

    cv2.destroyAllWindows()

    print("Selected Points:", points)
    # Get the file extension from the original image
    _, file_extension = os.path.splitext(img_name)
    output_path = os.path.join(base_path, f"point-draw\{img_name[7:-4]}_marked{file_extension}")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    cv2.imwrite(output_path, display_image)
    print(f"Marked image saved as {output_path}")
