from flask import Flask, request, render_template
import logging
import request_processor

import pyutilib.subprocess.GlobalData

# To be able to use Pyomo + solver on a server.
# See https://github.com/PyUtilib/pyutilib/issues/31.
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, static_url_path="/static")

# Set maximum image upload size to 8MB.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


@app.route("/digit_correction", methods=["POST"])
def digit_correction_by_user():
    data = request_processor.process_corrected_digits(request)
    return render_template(
        "index.html",
        available_files=request_processor.retrieve_uploaded_files(),
        data=data,
        error=None,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    data = {}
    error = None

    if request.method == "POST":
        if "file" in request.files:
            data = request_processor.process_file_upload_request(request)
        elif "filename" in request.form:
            data = request_processor.process_old_file_resubmission_request(request)
        else:
            error = "Could not understand request"

    return render_template(
        "index.html",
        available_files=request_processor.retrieve_uploaded_files(),
        data=data,
        error=error,
    )


@app.route("/flush", methods=["GET"])
def delete_files():
    request_processor.process_delete_file_history_request()
    return render_template(
        "index.html",
        available_files=request_processor.retrieve_uploaded_files(),
        data={},
        error=None,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
