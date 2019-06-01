import numpy as np
import itertools
import copy
from PIL import Image, ImageDraw


class SudokuGenerator:
    def __init__(self, dim, font, border=0.05):
        self.dim = dim
        self.font = font
        self.border = border
        self.outline, self.outer_corners = self._get_outline()
        self.squares = self._get_square_coordinates(
            self.outer_corners["t"],
            self.outer_corners["l"],
            self.outer_corners["r"],
            self.outer_corners["b"],
        )
        self.corner_points = self._get_unique_corner_points()
        self.base_img = self._draw_raster()

    def generate(self, number_density):
        img = copy.copy(self.base_img)
        for sq in self.squares:
            if np.random.uniform() < number_density:
                self._add_number_to_square(img, sq)

        return np.asarray(img), self.corner_points

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
