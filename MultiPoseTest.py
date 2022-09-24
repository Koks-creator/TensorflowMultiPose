from time import time
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def loop_through_people(frame: np.array, keypoints_with_scores: np.array, bbox_with_scores: np.array,
                        edges: dict, confidence_threshold: float) -> None:
    for person, bbox in zip(keypoints_with_scores, bbox_with_scores):
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)
        draw_bboxes(frame, bbox, confidence_threshold)


def draw_keypoints(frame: np.array, keypoints: np.array, confidence_threshold: float) -> None:
    h, w, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [h, w, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame: np.array, keypoints: np.array, edges: dict, confidence_threshold: float) -> None:
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def draw_bboxes(frame: np.array, bbox_keypoints: np.array, confidence_threshold: float) -> None:
    h, w, c = frame.shape

    shaped = np.squeeze(bbox_keypoints)
    y_min, x_min, y_max, x_max, kp_conf = shaped[0] * h, shaped[1] * w, shaped[2] * h, shaped[3] * w, shaped[4]

    if kp_conf > confidence_threshold:
        line_w = int(x_max - x_min) // 80
        line_w = 5 if line_w > 5 else line_w
        line_w = 2 if line_w <= 1 else line_w

        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), line_w)
        cv2.putText(frame, f"{int((kp_conf * 100))}%", (int(x_min), int(y_min - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 50, 200), 2)


model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']

cap = cv2.VideoCapture('tenis.mp4')
p_time = 0
while cap.isOpened():
    success, frame = cap.read()
    if success is False:
        break

    frame = cv2.flip(frame, 1)
    img = frame.copy()

    # for camera
    # img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 256)  # 480/640 = .75, 256*.75 = 192, 192 / 8 == 0

    # for hd
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 320)
    # for full hd videos
    # img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 352, 640)
    input_img = tf.cast(img, dtype=tf.int32)

    # Detection
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))  # 6ppl, 17 keypoints, 3 vals per key
    bbox_with_scores = results['output_0'].numpy()[:, :, 51:56].reshape((6, 5, 1))

    loop_through_people(frame, keypoints_with_scores, bbox_with_scores, EDGES, .1)

    c_time = time()
    fps = int(1 / (c_time - p_time))
    p_time = c_time

    cv2.putText(frame, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 50, 200), 2)
    cv2.imshow("Res", frame)
    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
