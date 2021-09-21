import argparse
import numpy as np
import cv2


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def get_gaussian_sequence(image, levels):
    gaussian_sequence = [image.copy()]
    for _ in range(levels):
        image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
        gaussian_sequence.append(image)
    return gaussian_sequence


def get_laplacian_sequences(image, levels):
    gaussian_sequence = get_gaussian_sequence(image, levels)
    laplacian_sequence = []

    for i in range(levels):
        image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
        laplacian_sequence.append(gaussian_sequence[i] - image)

    laplacian_sequence.append(image)
    return laplacian_sequence


def blend(A, B, M, levels, output_path=None):
    A = normalize(cv2.imread(A))
    B = normalize(cv2.imread(B))
    M = normalize(cv2.imread(M, cv2.IMREAD_GRAYSCALE))

    L_A = get_laplacian_sequences(A, levels)
    L_B = get_laplacian_sequences(B, levels)
    G_M = get_gaussian_sequence(M, levels)

    L_out = [G_M[i][:, :, None] * L_A[i] + (1 - G_M[i])[:, :, None] * L_B[i] for i in range(levels+1)]

    blended = np.sum(np.asarray(L_out), axis=0)
    cv2.imwrite(output_path, blended * 255)


def main():
    parser = argparse.ArgumentParser(description="Blend images using Laplacian Pyramids or Laplacian Sequences.")

    parser.add_argument(
        "-a",
        help="Path to the image to blend with 'B'.",
        required=True,
        type=str)

    parser.add_argument(
        "-b",
        help="Path to the image to blend with 'A'.",
        required=True,
        type=str)

    parser.add_argument(
        "-m",
        help="Blend mask.",
        required=True,
        type=str)

    parser.add_argument(
        "-l",
        help="Number of levels to use.",
        required=True,
        type=int)

    parser.add_argument(
        "-o",
        help="Path where to save the blended image.",
        default="blended.png",
        type=str)

    args = parser.parse_args()
    blend(
        A=args.a,
        B=args.b,
        M=args.m,
        levels=args.l,
        output_path=args.o
    )


if __name__ == "__main__":
    main()
