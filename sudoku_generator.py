import numpy as np
import itertools
import copy
from PIL import Image, ImageDraw
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage


class SudokuGenerator:
    def __init__(self, dim, border=0.05, batch_size=32):
        self.dim = dim
        self.border = border
        self.batch_size = batch_size

        self.font = self._get_font()
        self.outline, self.outer_corners = self._get_outline()
        self.squares = self._get_square_coordinates(
            self.outer_corners["t"],
            self.outer_corners["l"],
            self.outer_corners["r"],
            self.outer_corners["b"],
        )
        self.corner_points = self._get_unique_corner_points()
        self.base_img = self._draw_raster()

    def _get_font(self):
        from matplotlib.font_manager import FontManager
        from PIL import ImageFont

        font_finder = FontManager()
        font_path = font_finder.findfont("arial")
        return ImageFont.truetype(font_path, 35)

    def generate_batch(self, augment=True):
        while True:
            sudoku_list = []
            corners_list = []

            for i in range(self.batch_size):
                sudoku, corners = self._generate_one(
                    number_density=np.random.uniform(0.05, 0.20)
                )
                sudoku_list.append(sudoku)
                corners_list.append(corners)

            sudoku_array = np.array(sudoku_list)
            corners_array = np.array(corners_list)

            if augment:
                sudoku_array, corners_array = self._augment_batch(
                    sudoku_array, corners_array
                )

            yield (sudoku_array, corners_array.reshape(32, 8))

    def _augment_batch(self, images, corners):
        aug = iaa.SomeOf(
            (0, None),
            [
                iaa.KeepSizeByResize(
                    iaa.Affine(
                        translate_px={"x": (10, 30)},
                        rotate=(-5, 5),
                        mode="edge",
                        fit_output=True,
                    )
                ),
                iaa.KeepSizeByResize(
                    iaa.Affine(shear=(-10, 10), mode="edge", fit_output=True)
                ),
                iaa.AddToHueAndSaturation((-50, 50)),
                iaa.AverageBlur(k=(2, 5)),
            ],
            random_order=True,
        )

        # Convert array of corners to list of KeypointsOnImage instances for use with the augmenter.
        keypoints_from_corners = [
            KeypointsOnImage(
                [Keypoint(x=point[0], y=point[1]) for point in img_corners],
                shape=self.dim,
            )
            for img_corners in corners
        ]

        images_augm, keypoints_augm = aug.augment(
            images=images, keypoints=keypoints_from_corners
        )

        # Convert augmented keypoints back to array of size (batch_size, 4, 2).
        corners_augm = np.array(
            [keypoints.to_xy_array() for keypoints in keypoints_augm]
        )

        return images_augm, corners_augm

    def _generate_one(self, number_density):
        sudoku = copy.copy(self.base_img)
        for sq in self.squares:
            if np.random.uniform() < number_density:
                self._add_number_to_square(sudoku, sq)

        sudoku = np.array(sudoku)
        corners = np.array(self.outline)
        return sudoku, corners

    def _get_outline(self):
        top = int(self.border * self.dim[1])
        left = int(self.border * self.dim[0])
        right = self.dim[0] - left
        bottom = self.dim[1] - top

        outer_corners = {"t": top, "l": left, "r": right, "b": bottom}
        outline = ((left, top), (left, bottom), (right, bottom), (right, top))

        return outline, outer_corners

    def _get_square_coordinates(self, t, l, r, b):
        self.total_width = r - l
        self.total_height = b - t

        self.sq_width = self.total_width / 9
        self.sq_height = self.total_height / 9

        squares = []

        for i, j in itertools.product(range(9), range(9)):
            sq_left = int(l + i * self.sq_width)
            sq_top = int(t + j * self.sq_height)
            sq_right = int(l + (i + 1) * self.sq_width)
            sq_bottom = int(t + (j + 1) * self.sq_height)

            sq = (
                (sq_left, sq_top),
                (sq_left, sq_bottom),
                (sq_right, sq_bottom),
                (sq_right, sq_top),
            )
            squares.append(sq)

        return squares

    def _draw_raster(self):
        img = np.array(np.ones((self.dim[0], self.dim[1], 3)) * 255).astype(int)
        img = Image.fromarray(np.uint8(img))
        img = self._draw_outline(img)
        img = self._draw_squares(img)
        return img

    def _draw_outline(self, img):
        draw = ImageDraw.Draw(img)
        draw.polygon(self.outline, outline="black")
        return img

    def _draw_squares(self, img):
        draw = ImageDraw.Draw(img)
        for sq in self.squares:
            draw.polygon(sq, outline="black")
        return img

    def _add_number_to_square(self, img, sq_coords):
        number = np.random.randint(1, 10)
        sq_left = sq_coords[0][0]
        sq_top = sq_coords[0][1]

        draw = ImageDraw.Draw(img)
        draw.text((sq_left + 15, sq_top + 5), str(number), fill="black", font=self.font)
        return img

    def _get_unique_corner_points(self):
        # Flatten list of square coordinates, then find all unique corners.
        return list(set([coord for sq in self.squares for coord in sq]))
