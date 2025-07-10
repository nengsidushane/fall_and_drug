from rknn.api import RKNN


def create_rknn_session(
    model_path: str
) -> RKNN:
    rknn = RKNN()

    ret = rknn.load_rknn(model_path)
    if ret:
        raise OSError(f"{model_path}: Export rknn model failed!")

    ret = rknn.init_runtime(async_mode=True,target='rk3588',core_mask=RKNN.NPU_CORE_ALL)
    if ret:
        raise OSError(f"{model_path}: Init runtime enviroment failed!")

    return rknn
