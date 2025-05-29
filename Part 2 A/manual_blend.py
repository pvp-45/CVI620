# manual_blend.py

import cv2
import numpy as np
import argparse

def manual_blend(img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend two images using the formula:
        blended = (1 - alpha)*img1 + alpha*img2
    without using cv2.addWeighted.
    """
    # 1) Ensure same dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 2) Convert to float32 to avoid overflow/rounding errors
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)

    # 3) Compute weighted sum
    blended_f = (1.0 - alpha) * f1 + alpha * f2

    # 4) Clip to [0,255] and convert back to uint8
    blended = np.clip(blended_f, 0, 255).astype(np.uint8)
    return blended

def main():
    parser = argparse.ArgumentParser(
        description="Manually blend two images with NumPy (no cv2.addWeighted)."
    )
    parser.add_argument("img1", help="Path to first image")
    parser.add_argument("img2", help="Path to second image")
    parser.add_argument(
        "-a", "--alpha", type=float, default=0.5,
        help="Blending factor α (0.0 – 1.0)."
    )
    args = parser.parse_args()

    # Validate alpha
    if not (0.0 <= args.alpha <= 1.0):
        parser.error("Alpha must be between 0.0 and 1.0")

    # Load images
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    if img1 is None or img2 is None:
        parser.error("Could not load one of the images.")

    # Blend
    blended = manual_blend(img1, img2, args.alpha)

    # Display and save
    cv2.imshow("Manual Blend", blended)
    cv2.imwrite("manual_blend.jpg", blended)
    print(f"Saved blended image as 'manual_blend.jpg' with α={args.alpha}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


