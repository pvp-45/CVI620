import cv2
import numpy as np
from collections import deque

# ——— Utility Functions ——————————————————————————————————————————————————

def show_side_by_side(orig, edited, title="Preview"):
    # ensure same shape
    if orig.shape != edited.shape:
        edited = cv2.resize(edited, (orig.shape[1], orig.shape[0]))
    combo = np.hstack((orig, edited))

    
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow(title, combo.shape[1], combo.shape[0])
    cv2.imshow(title, combo)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load '{path}'")
    return img


def adjust_brightness(img, beta):
    return np.clip(img.astype(np.int16) + beta, 0, 255).astype(np.uint8)

def adjust_contrast(img, alpha):
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

def to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def add_padding(img, top, bottom, left, right, border_type, value=(0,0,0)):
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              borderType=border_type, value=value)

def apply_threshold(img, thresh, inv=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ttype = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, thresh, 255, ttype)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

def manual_blend(img1, img2, alpha):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    out = ((1-alpha)*f1 + alpha*f2).clip(0,255).astype(np.uint8)
    return out


def main():
    history = []               
    stack   = deque(maxlen=20) 

    path = input("Enter path of image to load: ").strip()
    img  = load_image(path)
    stack.append(img.copy())

    while True:
        print("""
==== Mini Photo Editor ====
1. Adjust Brightness
2. Adjust Contrast
3. Convert to Grayscale
4. Add Padding
5. Apply Thresholding
6. Blend with Another Image
7. Undo Last Operation
8. View History
9. Save and Exit
""")
        choice = input("Select an option [1–9]: ").strip()

        try:
            if choice == '1':
                b = int(input("  Brightness delta (−100…100): "))
                new = adjust_brightness(img, b)
                history.append(f"brightness {b:+d}")

            elif choice == '2':
                a = float(input("  Contrast factor (e.g. 1.2): "))
                new = adjust_contrast(img, a)
                history.append(f"contrast ×{a:.2f}")

            elif choice == '3':
                new = to_grayscale(img)
                history.append("grayscale")

            elif choice == '4':
                t = int(input("  Top padding px: "))
                btm = int(input("  Bottom padding px: "))
                l = int(input("  Left padding px: "))
                r = int(input("  Right padding px: "))
                print("  Border types: 0=CONST 1=REFLECT 2=REFLECT101 3=REPLICATE 4=WRAP")
                bt = int(input("  Choose border type: "))
                val = None
                if bt == 0:
                    color = input("  Constant color (B,G,R)? ")
                    val = tuple(map(int, color.split(',')))
                new = add_padding(img, t, btm, l, r, bt, val)
                history.append(f"padded {t},{btm},{l},{r} type={bt}")

            elif choice == '5':
                thr = int(input("  Threshold (0–255): "))
                inv = input("  Inverse? (y/N): ").lower().startswith('y')
                new = apply_threshold(img, thr, inv)
                history.append(f"threshold {thr} {'INV' if inv else ''}")

            elif choice == '6':
                p2 = input("  Second image path: ").strip()
                a2 = float(input("  Alpha (0.0–1.0): "))
                sec = load_image(p2)
                new = manual_blend(img, sec, a2)
                history.append(f"blend '{p2}' α={a2:.2f}")

            elif choice == '7':
                if len(stack) > 1:
                    stack.pop()
                    img = stack[-1].copy()
                    undone = history.pop()
                    print(f"Undid: {undone}")
                else:
                    print("Nothing to undo.")
                continue

            elif choice == '8':
                print("History:")
                for i, act in enumerate(history, 1):
                    print(f" {i}. {act}")
                continue

            elif choice == '9':
                fn = input(" Save as (filename): ").strip()
                cv2.imwrite(fn, img)
                print(f"Saved '{fn}'. Bye!")
                break

            else:
                print("Invalid choice.")
                continue

           
            show_side_by_side(stack[-1], new, title=history[-1])
            stack.append(new.copy())
            img = new

        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
