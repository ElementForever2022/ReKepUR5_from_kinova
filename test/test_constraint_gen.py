import cv2
import numpy as np
from constraint_generation import ConstraintGenerator
import os

def generate_random_points(num_points=5):
    img = np.zeros((512, 512, 3), np.uint8)
    points = []
    for _ in range(num_points):
        x = np.random.randint(0, 512)
        y = np.random.randint(0, 512)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        points.append((x, y))
    return img, points

def test_constraint_generator():
    # Initialize ConstraintGenerator
    config = {
        'model': 'chatgpt-4o-latest',
        'temperature': 0.0,
        'max_tokens': 2048
    }
    constraint_generator = ConstraintGenerator(config)

    # Generate random points
    img, points = generate_random_points(num_points=5)

    # Save the image
    test_image_path = 'test_image.png'
    cv2.imwrite(test_image_path, img)

    # Prepare metadata
    metadata = {
        'init_keypoint_positions': np.array(points),
        'num_keypoints': len(points)
    }

    # Test the generate method
    instruction = "Move the green dots to form a triangle"
    result_dir = constraint_generator.generate(img, instruction, metadata)

    print(f"Result saved in: {result_dir}")

    # Clean up
    os.remove(test_image_path)

    # Display the image with points (optional)
    # Save the output image to the data directory
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'test_output.png')
    cv2.imwrite(output_path, img)
    print(f"Output image saved to: {output_path}")

if __name__ == "__main__":
    test_constraint_generator()
