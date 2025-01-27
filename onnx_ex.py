import onnxruntime as ort
import numpy as np
import cv2
import postprocess_dmonitoring as pst

def BGR2YUV420(img_bgr):
    # convert bgr picture to YUV420 
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    y_channel = img_yuv[:, :, 0]
    y_channel_normalized = y_channel / 255.0
    output = y_channel_normalized.reshape((1,1382400)).astype(np.float32)
    return output

if __name__ == "__main__":
    # ONNX Model Path
    model_path = "dmonitoring_model.onnx"

    # ONNX Model Load
    session = ort.InferenceSession(model_path)

    # ONNX Model input info
    inputs = session.get_inputs()

    # Calibration Info: zero
    calib_shape = inputs[1].shape
    calib_name = inputs[1].name
    calib_data = np.zeros([dim if dim else 1 for dim in calib_shape], dtype=np.float32)

    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = cv2.resize(frame, (1440,960))
        input_data = BGR2YUV420(frame)

        # ONNX Model Run
        outputs = session.run(None, {inputs[0].name: input_data, inputs[1].name: calib_data})

        # Get Output
        pst_output = pst.get_result_onnx(outputs[0][0])

        y = 0
        # for key, value in pst_output['driver_state_lhd'].items():
        #     cv2.putText(frame, '{0}: {1}'.format(key, value), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        #     y += 20
        #     print(key, ":", value)

        cv2.putText(frame, '{0}: pitch:{1:.2f} yaw:{2:.2f} roll:{3:.2f}'.format('face_orientation', *pst_output['driver_state_lhd']['face_orientation']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f} {2:.2f} {3:.2f}'.format('face_orientation_std', *pst_output['driver_state_lhd']['face_orientation_std']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f} {2:.2f}'.format('face_position', *pst_output['driver_state_lhd']['face_position']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f} {2:.2f}'.format('face_position_std', *pst_output['driver_state_lhd']['face_position_std']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f}'.format('face_prob', pst_output['driver_state_lhd']['face_prob']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f}'.format('left_eye_prob', pst_output['driver_state_lhd']['left_eye_prob']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f}'.format('right_eye_prob', pst_output['driver_state_lhd']['right_eye_prob']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f}'.format('left_blink_prob', pst_output['driver_state_lhd']['left_blink_prob']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f}'.format('right_blink_prob', pst_output['driver_state_lhd']['right_blink_prob']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f}'.format('sunglasses_prob', pst_output['driver_state_lhd']['sunglasses_prob']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f}'.format('occluded_prob', pst_output['driver_state_lhd']['occluded_prob']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f} {2:.2f} {3:.2f} {4:.2f}'.format('ready_prob', *pst_output['driver_state_lhd']['ready_prob']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.putText(frame, '{0}: {1:.2f} {2:.2f}'.format('not_ready_prob', *pst_output['driver_state_lhd']['not_ready_prob']), (50, 50+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        y += 20
        cv2.imshow("Webcam", frame)
