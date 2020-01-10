import flask
import os
import logging
import numpy as np
from typing import List
import cv2
from werkzeug.utils import secure_filename
from brain.solver import solve_sudoku
from brain.image_parser import load_image, parse_grid_from_img, extract_sudoku
from brain.plotting import image_from_digits
from brain.model.model import load_trained_model


logging.basicConfig(level=logging.DEBUG)


def retrieve_uploaded_files() -> List[str]:
    return [
        file
        for file in os.listdir(os.path.join("static", "uploaded_images", "raw"))
        if not file.startswith(".")
    ]


def save_preprocessed_image(
    digit_patches: List[np.array], file_name: str, save_folder: str
):
    """Put the preprocessed digit patches back together and save the preprocessed
    image in its entirety.

    :param digit_patches: List of image patches cut from the sudoku.
    :param file_name: Name of the file matching the patches.
    :param save_folder: Folder to save preprocessed image to.
    :return: Path the image was saved to.
    """

    digit_img = image_from_digits(digit_patches)
    save_path = os.path.join(save_folder, file_name)
    cv2.imwrite(save_path, digit_img)
    return save_path


def solve_sudoku_from_img(raw_img_path: str) -> dict:
    """Parse image of sudoku and solve it.

    :param raw_img_path: Image of sudoku to be solved.
    :return: Dict of data needed to render html template.
    """

    img = load_image(raw_img_path)
    digit_patches = parse_grid_from_img(img)
    preprocessed_img_path = save_preprocessed_image(
        digit_patches,
        file_name=os.path.basename(raw_img_path),
        save_folder=os.path.join("static", "uploaded_images", "preprocessed"),
    )
    # Load the trained digit recognition model.
    model = load_trained_model(
        os.path.join("brain", "model", "model_weights_custom.hdf5")
    )
    recognized_digits = extract_sudoku(digit_patches, model)
    solved_sudoku = solve_sudoku(recognized_digits)
    print(raw_img_path)
    print(preprocessed_img_path)
    return {
        "image_path": raw_img_path,
        "preprocessed_img_path": preprocessed_img_path,
        "recognized_digits": recognized_digits.flatten().reshape(-1).astype(int),
        "solved_sudoku": solved_sudoku.flatten().reshape(-1),
    }  # `recognized_digits` and `solved_sudoku` are flattened, because the HTML
    # sudoku solution template expects a vector input.


def process_corrected_digits(request: flask.request):
    """Rerun the solver on sudoku matrix as corrected by the user.

    :param request: Flask request containing the submitted form with corrections.
    :return: Dict of data needed to render html template.
    """

    recognized_digits = np.array(
        [
            float(request.form[str(i)]) if (request.form[str(i)] != "") else 0.0
            for i in range(81)
        ]
    ).reshape(9, 9)
    solved_sudoku = solve_sudoku(recognized_digits)

    return {
        "image_path": request.form["image_path"],
        "preprocessed_img_path": request.form["preprocessed_img_path"],
        "recognized_digits": recognized_digits.flatten().reshape(-1).astype(int),
        "solved_sudoku": solved_sudoku.flatten().reshape(-1),
    }


def process_file_upload_request(request: flask.request):
    f = request.files["file"]
    image_path = os.path.join(
        "static", "uploaded_images", "raw", secure_filename(f.filename)
    )
    f.save(image_path)
    data = solve_sudoku_from_img(image_path)
    return data


def process_old_file_resubmission_request(request: flask.request):
    filename = request.form["filename"]
    image_path = os.path.join("static", "uploaded_images", "raw", filename)
    data = solve_sudoku_from_img(image_path)
    return data


def process_delete_file_history_request():
    logging.info("Deleting all previously uploaded files.")
    files = retrieve_uploaded_files()
    for file_name in files:
        # Delete both raw and preprocessed image.
        for folder in ["raw", "preprocessed"]:
            file_path = os.path.join("static", "uploaded_images", folder, file_name)
            os.remove(file_path)
            logging.info(f"\tDeleted {file_path}")
