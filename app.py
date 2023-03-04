import streamlit as st
import cv2
import numpy as np
from roboflow import Roboflow
import time
import bcrypt
import tensorflow as tf
import tensorflow_models as tfm
from official.vision.ops.preprocess_ops import resize_and_crop_image
from google.oauth2 import service_account
from google.cloud import storage
import os
import zipfile


@st.cache_resource(show_spinner='Preparing Model...', max_entries=2)
def get_local_model(name):
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = storage.Client(credentials=credentials)

    bucket_name = "burger-detect-models"

    if name == 'RetinaNet + ResNet + FPN (0.6 AP)':
        file_path = "retinanet_resnetfpn_coco.zip"
    if name == 'RetinaNet + SpineNet (0.55 AP)':
        file_path = "retinanet_spinenet_coco.zip"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    file_path = os.path.basename(file_path)
    blob.download_to_filename(file_path)

    with zipfile.ZipFile(file_path, 'r') as zf:
        for entry in zf.infolist():
            try:
                zf.extract(entry, './')
            except zipfile.error as e:
                pass

    imported = tf.saved_model.load(f'./{file_path[:-4]}')
    model = imported.signatures['serving_default']
    return model


def build_inputs_for_object_detection(image, input_image_size):
    image, _ = resize_and_crop_image(
        image,
        input_image_size,
        padded_size=input_image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    return image


# @st.cache_resource(show_spinner='Preparing Model...', max_entries=2)
# def get_local_model(name):
#     if name == 'RetinaNet + ResNet + FPN (0.6 AP)':
#         imported = tf.saved_model.load('./retinanet_resnetfpn_coco')
#     if name == 'RetinaNet + SpineNet (0.55 AP)':
#         imported = tf.saved_model.load('./retinanet_spinenet_coco')
#     model = imported.signatures['serving_default']
#     return model


@st.cache_resource(show_spinner='Preparing Model...', ttl=300.0, max_entries=1)
def get_remote_model():
    rf = Roboflow(st.secrets["PRIVATE_API"])
    project = rf.workspace().project("mobile-web-pages")
    model = project.version(9).model
    return model


def verify_remote_password(password):
    with open('remote_password', 'rb') as f:
        hashed_password = f.read()
        permission = bcrypt.checkpw(password.encode(), hashed_password)
    return permission


FLAG = True

st.title(":violet[David's predictions] :crystal_ball:")
st.subheader('Burger-menu detection')
uploaded_file = st.file_uploader("Choose an image", ["jpg", "jpeg", "png"])
opencv_image = None

col1, col2, col3 = st.columns([2, 1, 1])

show_box_type = col1.radio("Display Bounding Boxes As:",
                           options=['regular', 'fill'])

model_type = col2.radio("Model Type:",
                        options=['Local', 'Remote'])

if model_type == 'Remote':
    if 'permission' not in st.session_state:
        st.session_state['permission'] = False
    col3.caption(
        'Remote model is much more accurate (97.7% mAP), but requires choosing the right level of confidence')
    if not st.session_state['permission']:
        st.session_state['password'] = st.text_input(
            "Please enter a password to get access to remote model",
            placeholder="Password",
            type='password'
        )
        if st.session_state['password']:
            st.session_state['permission'] = verify_remote_password(st.session_state['password'])
            if st.session_state['permission']:
                st.success('Password correct!')
            else:
                st.error('Wrong password, please try again')

    if st.session_state['permission'] and uploaded_file:
        confidence = st.slider("Confidence", value=40,
                               min_value=0, max_value=100)
        model = get_remote_model()
        st.success('Model is ready!')

if model_type == 'Local':
    local_name = col3.radio("Local model specification:",
                            options=['RetinaNet + ResNet + FPN (0.6 AP)',
                                     'RetinaNet + SpineNet (0.55 AP)'])
    if uploaded_file:
        model_fn = get_local_model(local_name)
        st.success('Model is ready!')

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

_, col_center, _ = st.columns(3)

if col_center.button('Detect :hamburger:', use_container_width=True):
    if opencv_image is None:
        st.text('Please, load an image')
    else:
        start_total_time = time.time()

        if model_type == 'Remote':
            start_time = time.time()
            predictions = model.predict(
                opencv_image, confidence=confidence, overlap=30)
            time_run = time.time() - start_time
            time_run = f'Inference time: {time_run:.1f}s'
            try:
                bounding_box = predictions[0]
                x0 = bounding_box['x'] - bounding_box['width'] / 2
                x1 = bounding_box['x'] + bounding_box['width'] / 2
                y0 = bounding_box['y'] - bounding_box['height'] / 2
                y1 = bounding_box['y'] + bounding_box['height'] / 2
            except Exception:
                FLAG = False

            thickness = 10
            img = opencv_image

        if model_type == 'Local':
            HEIGHT, WIDTH = 896, 512
            input_image_size = (HEIGHT, WIDTH)
            min_score_thresh = 0.01

            image = opencv_image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = build_inputs_for_object_detection(image, input_image_size)
            image = tf.expand_dims(image, axis=0)
            image = tf.cast(image, dtype=tf.uint8)
            # img = image[0].numpy()
            start_time = time.time()
            result = model_fn(image)
            time_run = time.time() - start_time
            time_run = f'Inference time: {time_run:.1f}s'
            y0, x0, y1, x1 = result['detection_boxes'][0][0].numpy()
            limit = int(
                opencv_image.shape[1] / (opencv_image.shape[0] / input_image_size[0]))
            
            thickness = 4
            img = cv2.resize(opencv_image, (limit, 896), interpolation=cv2.INTER_AREA)

        if FLAG:
            start_point = (int(x0), int(y0))
            end_point = (int(x1), int(y1))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                if show_box_type == 'regular':
                    img = cv2.rectangle(img, start_point, end_point, color=(
                        255, 0, 0), thickness=thickness)
                elif show_box_type == 'fill':
                    sub_img = img[int(y0):int(y1), int(x0):int(x1)]
                    red_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                    red_rect[:, :, 0::3] = 255
                    res = cv2.addWeighted(sub_img, 0.5, red_rect, 0.5, 1.0)
                    img[int(y0):int(y1), int(x0):int(x1)] = res
            except Exception:
                FLAG = False

        total_time_run = time.time() - start_total_time
        total_time_run = f'Total run time: {total_time_run:.1f}s'

        _, col_center, _ = st.columns([1, 2, 1])
        col_center.text(total_time_run)
        col_center.text(time_run)

        if FLAG:
            col_center.image(img, channels="RGB", width=400,
                             use_column_width='auto')
        else:
            if model_type == 'Remote':
                col_center.error(
                    'No hamburger menu found, please try to set a lower confidence and we will definitely catch it!')
            if model_type == 'Local':
                col_center.error(
                    'No hamburger menu found, please try another model and we will definitely catch it!')
