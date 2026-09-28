import numpy as np
import matplotlib.pyplot as plt
import random


def make_test_image(size: int = 40) -> np.ndarray:
  
    random.seed(0)
    img = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            base = 220.0 if (i // 5 + j // 5) % 2 == 0 else 40.0
            noise = random.uniform(-15, 15)
            img[i, j] = max(0.0, min(255.0, base + noise))

    return img


def pad_image(image: np.ndarray, pad: int, mode: str) -> np.ndarray:
    
    if mode == "zero":
        return np.pad(image, pad_width=pad, mode="constant", constant_values=0.0)

    if mode == "edge":
        return np.pad(image, pad_width=pad, mode="edge")

    raise ValueError("mode должен быть 'edge' или 'zero'")


def box_blur(image: np.ndarray, kernel_size: int = 3, mode: str = "edge") -> np.ndarray:
  
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size должен быть нечётным положительным числом")

    pad = kernel_size // 2
    padded = pad_image(image, pad, mode)

    height, width = image.shape
    result = np.zeros_like(image, dtype=float)

    for i in range(height):
        for j in range(width):
            window = padded[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = window.mean()

    return result


def to_black_and_white(image: np.ndarray, threshold: float = 128.0) -> np.ndarray:
    """Бинаризация: всё >= threshold становится белым, остальное — чёрным."""
    return np.where(image >= threshold, 255.0, 0.0)


def main():
    image = make_test_image(size=40)

   
    blurred_edge = box_blur(image, kernel_size=3, mode="edge")
    blurred_zero = box_blur(image, kernel_size=9, mode="zero")

  
    binary_image = to_black_and_white(image, threshold=128.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = [
        "Исходная матрица",
        "Blur k=3, edge",
        "Blur k=9, zero",
        "Бинарное ч/б",
    ]
    data = [image, blurred_edge, blurred_zero, binary_image]

    for ax, title, d in zip(axes, titles, data):
        ax.imshow(d, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()