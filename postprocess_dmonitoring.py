import numpy as np

def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def parse_driver_data(output, offset=0):
    REG_SCALE = 0.25
    ds_res = {}

    # Face Orientation
    ds_res['face_orientation'] = [
        output[offset + i] * REG_SCALE for i in range(3)]
    ds_res['face_orientation_std'] = [
        np.exp(output[offset + 6 + i]) for i in range(3)]

    # Face Position
    ds_res['face_position'] = [
        output[offset + 3 + i] * REG_SCALE for i in range(2)]
    ds_res['face_position_std'] = [
        np.exp(output[offset + 9 + i]) for i in range(2)]

    # Face and Eye probability
    ds_res['face_prob'] = sigmoid(output[offset + 12])
    ds_res['left_eye_prob'] = sigmoid(output[offset + 21])
    ds_res['right_eye_prob'] = sigmoid(output[offset + 30])
    ds_res['left_blink_prob'] = sigmoid(output[offset + 31])
    ds_res['right_blink_prob'] = sigmoid(output[offset + 32])
    ds_res['sunglasses_prob'] = sigmoid(output[offset + 33])
    ds_res['occluded_prob'] = sigmoid(output[offset + 34])

    # Ready: touching wheel probability, paying attention probability, (deprecated) distracted probabilities
    ds_res['ready_prob'] = [sigmoid(output[offset + 35 + i]) for i in range(4)]

    # Not Ready: using phone probability, distracted probability
    ds_res['not_ready_prob'] = [
        sigmoid(output[offset + 39 + i]) for i in range(2)]

    return ds_res

# 실행 함수
def get_result_onnx(outputs):
    model_res = {}

    # Left Hand Driver(LHD)
    model_res['driver_state_lhd'] = parse_driver_data(outputs, offset=0)

    # Right Hand Driver(RHD)
    model_res['driver_state_rhd'] = parse_driver_data(outputs, offset=41)

    # Common Probability
    model_res['poor_vision_prob'] = sigmoid(outputs[82])
    model_res['wheel_on_right_prob'] = sigmoid(outputs[83])

    return model_res
