from os import makedirs, path
from threading import Timer
from time import sleep

import config_utils
import IPCUtils as ipc_utils
import prediction_utils

from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import time


# initialize a flask object
app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def start_flask():
    # start the flask app
	app.run(host="0.0.0.0", port=8080, debug=False,
		threaded=True, use_reloader=False)

if __name__ == '__main__':
    thread = threading.Thread(target=start_flask)
    thread.start()
    
def set_configuration(config):
    r"""
    Sets a new config object with the combination of updated and default configuration as applicable.
    Calls inference code with the new config and indicates that the configuration changed.
    """
    new_config = {}

    if "ImageName" in config:
        new_config["image_name"] = config["ImageName"]
    else:
        new_config["image_name"] = config_utils.DEFAULT_IMAGE_NAME
        config_utils.logger.warning(
            "Using default image name: {}".format(config_utils.DEFAULT_IMAGE_NAME)
        )

    if "ImageDirectory" in config:
        new_config["image_dir"] = config["ImageDirectory"]
    else:
        new_config["image_dir"] = config_utils.IMAGE_DIR
        config_utils.logger.warning(
            "Using default image directory: {}".format(config_utils.IMAGE_DIR)
        )

    if "InferenceInterval" in config:
        new_config["prediction_interval_secs"] = config["InferenceInterval"]
    else:
        new_config["prediction_interval_secs"] = config_utils.DEFAULT_PREDICTION_INTERVAL_SECS
        config_utils.logger.warning(
            "Using default inference interval: {}".format(
                config_utils.DEFAULT_PREDICTION_INTERVAL_SECS
            )
        )

    if "UseCamera" in config:
        new_config["use_camera"] = config["UseCamera"]
    else:
        new_config["use_camera"] = config_utils.DEFAULT_USE_CAMERA
        config_utils.logger.warning(
            "Using default camera: {}".format(config_utils.DEFAULT_USE_CAMERA)
        )
    
    if "UseRTSP" in config:
        new_config["rtsp_camera"] = config["UseRTSP"]
        config_utils.logger.info("Using RTSP")
    else:
        new_config["rtsp_camera"] = False

    if "RTSPURL" in config:
        new_config["rtsp_url"] = config["RTSPURL"]
        config_utils.logger.info("RTSP URL: {}".format(config["RTSPURL"]))
    else:
        new_config["rtsp_url"] = ""

    if "PublishResultsOnTopic" in config:
        config_utils.TOPIC = config["PublishResultsOnTopic"]
    else:
        config_utils.TOPIC = ""
        config_utils.logger.warning("Topic to publish inference results is empty.")

    if new_config["use_camera"]:
        prediction_utils.enable_camera(new_config["rtsp_camera"],new_config["rtsp_url"])

    new_config["image"] = prediction_utils.load_image(
        path.join(new_config["image_dir"], new_config["image_name"])
    )

    # Create the directory for output images with overlaid bounding boxes, if it does not exist already
    makedirs(config_utils.BOUNDED_OUTPUT_DIR, exist_ok=True)

    # Run inference with the updated config indicating the config change.
    run_inference(new_config, True)


def run_inference(new_config, config_changed):
    r"""
    Uses the new config to run inference.

    :param new_config: Updated config if the config changed. Else, the last updated config.
    :param config_changed: Is True when run_inference is called after setting the newly updated config.
    Is False if run_inference is called using scheduled thread as the config hasn't changed.
    """
    #removed scheduler and running as an infinite loop
    prediction_utils.predict_from_cam(new_config["rtsp_camera"], new_config["rtsp_url"])
    

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(prediction_utils.generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# Get intial configuration from the recipe and run inference for the first time.
set_configuration(ipc_utils.IPCUtils().get_configuration())

# Subscribe to the subsequent configuration changes
ipc_utils.IPCUtils().get_config_updates()

while True:
    if config_utils.UPDATED_CONFIG:
        set_configuration(ipc_utils.IPCUtils().get_configuration())
        config_utils.UPDATED_CONFIG = False
    sleep(1)