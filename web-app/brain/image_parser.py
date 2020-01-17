# This part was strongly inspired by Nesh Patel (https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2)

from typing import List
import operator
import numpy as np
import cv2
import keras


def extract_sudoku(digits: List[np.array], model: keras.models.Sequential) -> np.array:
    """Apply digit recognition model to all images in `digits`.

    :param digits: List of images with digits to be recognized.
    :param model: Trained digit recognition model.
    :return: Array with recognized digits filled in.
    """

    pred_sudoku = np.zeros((81, 1))

    for i in range(len(digits)):

        # If the image is all white, then return zero (the solver interprets
        # zeros as empty squares in the sudoku).
        if (digits[i] / 255).mean() == 0.0:
            pred_sudoku[i] = 0
        # If the image is not all white, then apply the digit recognition model.
        else:
            img = scale_and_reshape(digits[i])
            pred = model.predict(img)
            pred_sudoku[i] = pred.argmax()

    return pred_sudoku.reshape(9, 9)


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def parse_grid_from_img(img: np.array) -> List[np.array]:
    """Detect sudoku grid location in image, deskew, and split into 81 squares.

    :param img: Image to analyze.
    :return: List of 81 arrays containing the squares making up the sudoku.
    """

    cropped_img = crop_to_grid(img)  # Find grid and deskew.
    enhanced_digits = enhance_digits(
        cropped_img
    )  # Enhance image for better digit recognition.
    grid_coords = infer_grid(enhanced_digits)  # Find coordinates of a 9x9 grid.
    patches = extract_patches(enhanced_digits, grid_coords)  # Cut image into 9x9 grid.
    digits = [
        extract_digit(patch) for patch in patches
    ]  # Extract digit patches more precisely.

    return digits


def crop_to_grid(img):
    """Detect sudoku grid in an image and deskew to a square."""

    enhanced_grid = enhance_grid_lines(img)
    corners = find_corners_of_largest_polygon(enhanced_grid)
    cropped = crop_and_warp_to_square(img, corners)

    # Apply twice for good measure.
    enhanced_grid = enhance_grid_lines(cropped)
    corners = find_corners_of_largest_polygon(enhanced_grid)
    cropped = crop_and_warp_to_square(cropped, corners)
    return cropped


def enhance_grid_lines(img):
    """Blur, threshold and dilate an image to enhance the grid lines."""
    blur_fraction = 50.0
    neighbourhood_factor = 45.0

    # Blur.
    blur_size = make_odd_and_greater_than_1(max(img.shape) / blur_fraction)
    enhanced_img = cv2.GaussianBlur(img.copy(), (blur_size, blur_size), 0)

    # Adaptive threshold.
    neighbourhood_size = make_odd_and_greater_than_1(
        max(img.shape) / neighbourhood_factor
    )
    enhanced_img = cv2.adaptiveThreshold(
        src=enhanced_img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=neighbourhood_size,
        C=2,
    )

    # Invert.
    enhanced_img = cv2.bitwise_not(enhanced_img, enhanced_img)

    # Dilate.
    enhanced_img = cv2.dilate(
        enhanced_img,
        np.array([[0.0, 0.5, 0.0], [0.5, 0.5, 0.5], [0.0, 0.5, 0.0]], dtype=np.uint8),
    )

    return enhanced_img


def enhance_digits(img):
    """Blur and threshold an image to enhance the digits."""
    blur_fraction = 50.0
    neighbourhood_factor = 45.0

    # Blur.
    blur_size = make_odd_and_greater_than_1(max(img.shape) / blur_fraction)
    enhanced_img = cv2.GaussianBlur(img.copy(), (blur_size, blur_size), 0)

    # Adaptive threshold.
    neighbourhood_size = make_odd_and_greater_than_1(
        max(img.shape) / neighbourhood_factor
    )

    enhanced_img = cv2.adaptiveThreshold(
        src=enhanced_img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=neighbourhood_size,
        C=5,
    )

    # Invert.
    enhanced_img = cv2.bitwise_not(enhanced_img, enhanced_img)

    return enhanced_img


def make_odd_and_greater_than_1(i):
    i = int(i)
    if i % 2 == 0:
        return i + 1
    elif i == 1:
        return 3
    return i


def find_corners_of_largest_polygon(img):
    """Find the four extreme corners of the largest contour in the image."""

    # Find contours.
    contours, _ = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get the largest contour by area.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_polygon = contours[0]

    # Use of `operator.itemgetter` with `max` and `min` allows us to get the index
    # of the point.
    # Each point is an array of 1 coordinate, hence the [0] getter,
    # then [0] or [1] used to get x and y respectively.

    # Bottom-right point has the largest (x + y) value.
    # Top-left has point smallest (x + y) value.
    # Bottom-left point has smallest (x - y) value.
    # Top-right point has largest (x - y) value.
    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in largest_polygon]),
        key=operator.itemgetter(1),
    )
    top_left, _ = min(
        enumerate([pt[0][0] + pt[0][1] for pt in largest_polygon]),
        key=operator.itemgetter(1),
    )
    bottom_left, _ = min(
        enumerate([pt[0][0] - pt[0][1] for pt in largest_polygon]),
        key=operator.itemgetter(1),
    )
    top_right, _ = max(
        enumerate([pt[0][0] - pt[0][1] for pt in largest_polygon]),
        key=operator.itemgetter(1),
    )

    # Return list of all 4 points using the indices.
    # Each point is in its own array of one coordinate.
    return [
        largest_polygon[top_left][0],
        largest_polygon[top_right][0],
        largest_polygon[bottom_right][0],
        largest_polygon[bottom_left][0],
    ]


def distance_between(p1, p2):
    """Return the scalar distance between two points."""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp_to_square(img, crop_rect):
    """Crop and warp a four-sided rectangular part of an image into a square."""

    top_left, top_right, bottom_right, bottom_left = (
        crop_rect[0],
        crop_rect[1],
        crop_rect[2],
        crop_rect[3],
    )

    # Set the data type to float32 or `getPerspectiveTransform` will throw an error.
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    # Get the longest side in the rectangle.
    longest_side = max(
        [
            distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),
            distance_between(top_left, top_right),
        ]
    )

    # Describe a square with `longest_side` of the calculated length,
    # this is the new perspective we want to warp to.
    dst = np.array(
        [
            [0, 0],
            [longest_side - 1, 0],
            [longest_side - 1, longest_side - 1],
            [0, longest_side - 1],
        ],
        dtype="float32",
    )

    # Get the transformation matrix for skewing the image to fit a square
    # by comparing the four before and after points.
    m = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img.copy(), m, (int(longest_side), int(longest_side)))


def infer_grid(img):
    """Infer the coordinates of a 9x9 grid from a square image."""
    coordinates = []
    side_length_whole = img.shape[:1]
    side_length_part = side_length_whole[0] / 9

    # Note that we swap j and i here so that the rectangles are stored
    # in the list reading left-right instead of top-down.
    for j in range(9):
        for i in range(9):
            p1 = (
                i * side_length_part,
                j * side_length_part,
            )  # Top left corner of bounding box.
            p2 = (
                (i + 1) * side_length_part,
                (j + 1) * side_length_part,
            )  # Bottom right corner of bounding box.
            coordinates.append((p1, p2))

    return coordinates


def cut_rectangle_from_img(img, rect):
    """Cut rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]) : int(rect[1][1]), int(rect[0][0]) : int(rect[1][0])]


def extract_patches(img, squares):
    """Return list of the image cut up in squares."""
    return [cut_rectangle_from_img(img, square) for square in squares]


def extract_digit(patch):
    """Approximately extract the digit patch from an image, resizing and centering it."""
    patch = cv2.copyMakeBorder(patch, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
    contours, _ = cv2.findContours(patch, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    width = patch.shape[1]
    height = patch.shape[0]

    contours = filter_contour_length(contours)
    contours = filter_center_of_mass(width, height, contours)
    contours = filter_contours_near_corner(width, height, contours)

    # If no suitable contours are found, return just a white image.
    if len(contours) == 0:
        return np.zeros((28, 28))

    digit = crop_from_contours(patch, contours)
    digit = pad_digit(digit)
    digit = resize_digit(digit)
    return digit


def filter_contour_length(contours):
    return [c for c in contours if len(c) > 4]


def filter_center_of_mass(width, height, contours):
    approved_contours = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if not in_center(width, height, cX, cY):
            continue

        approved_contours.append(c)
    return approved_contours


def in_center(width, height, c_x, c_y, fraction=0.4):
    margin_fraction = (1.0 - fraction) / 2

    width_margin = width * margin_fraction
    in_center_x = width_margin <= c_x <= (width - width_margin)

    height_margin = height * margin_fraction
    in_center_y = height_margin <= c_y <= (height - height_margin)

    return in_center_x and in_center_y


def filter_contours_near_corner(width, height, contours, fraction=1 / 8):
    accepted_contours = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        corners = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))

        near_corner = False
        for point in corners:
            if in_corner(width, height, point[0], point[1], fraction):
                near_corner = True
                break

        if not near_corner:
            accepted_contours.append(c)

    return accepted_contours


def in_corner(width, height, x, y, fraction=1 / 8):
    width_margin = int(width * fraction)
    height_margin = int(height * fraction)

    in_top_left = x <= width_margin and y <= height_margin
    in_top_right = x >= width - width_margin and y <= height_margin
    in_bot_left = x <= width_margin and y >= height - height_margin
    in_bot_right = x >= width - width_margin and y >= height - height_margin

    return in_top_left or in_top_right or in_bot_left or in_bot_right


def crop_from_contours(patch, contours):
    x, y, w, h = cv2.boundingRect(np.concatenate(contours, axis=0))
    rect = ((x, y), (x + w, y + h))
    return cut_rectangle_from_img(patch.copy(), rect)


def pad_digit(digit, v_margin=0.1):
    height = digit.shape[0]
    width = digit.shape[1]

    target_height = max(height, width)

    vertical_margin = int(target_height * v_margin)
    padded_height = target_height + 2 * vertical_margin

    horizontal_margin = int((padded_height - width) / 2)

    padded_digit = cv2.copyMakeBorder(
        digit,
        vertical_margin,
        vertical_margin,
        horizontal_margin,
        horizontal_margin,
        cv2.BORDER_CONSTANT,
    )

    return padded_digit


def resize_digit(digit, size=28):
    return cv2.resize(digit, (size, size))


def scale_and_reshape(img):
    return ((img) / 255).reshape(1, 28, 28, 1)
