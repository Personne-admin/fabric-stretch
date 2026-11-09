from __future__ import annotations

import logging
import threading

import numpy as np
from PIL import Image

try:
    from pyueye import ueye

    UEYE_AVAILABLE = True
except ImportError:  # vendor SDK / wheel not installed
    ueye = None
    UEYE_AVAILABLE = False

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


def frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))


class UEyeCamera:

    simulated = False

    def __init__(self):
        self._hcam = None
        self._mem_ptr = None
        self._mem_id = None
        self._width = 0
        self._height = 0
        self._bpp = 24
        self._pitch = 0
        self._open = False
        self._sensor_name = ""
        self._exposure_ms = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def size(self) -> tuple[int, int]:
        return self._width, self._height

    @staticmethod
    def _check(ret: int, what: str) -> None:
        if ret != ueye.IS_SUCCESS:
            raise CameraError(f"{what} failed (uEye error {ret})")

    def open(self, device_id: int = 0, width: int = 0, height: int = 0) -> None:
        if not UEYE_AVAILABLE:
            raise CameraError("pyueye is not installed")
        with self._lock:
            self._hcam = ueye.HIDS(device_id)
            self._check(ueye.is_InitCamera(self._hcam, None), "is_InitCamera")
            try:
                self._configure(width, height)
            except Exception:
                ueye.is_ExitCamera(self._hcam)
                raise
            self._open = True
        log.info("Camera open %dx%d (%s)", self._width, self._height, self._sensor_name)

    def _configure(self, width: int, height: int) -> None:
        self._check(
            ueye.is_SetColorMode(self._hcam, ueye.IS_CM_BGR8_PACKED), "is_SetColorMode"
        )
        self._bpp = 24
        sensor_info = ueye.SENSORINFO()
        self._check(ueye.is_GetSensorInfo(self._hcam, sensor_info), "is_GetSensorInfo")
        try:
            self._sensor_name = (
                bytes(sensor_info.strSensorName)
                .split(b"\0")[0]
                .decode(errors="replace")
            )
        except (TypeError, ValueError):
            self._sensor_name = ""
        self._width = width or sensor_info.nMaxWidth.value
        self._height = height or sensor_info.nMaxHeight.value

        rect = ueye.IS_RECT()
        rect.s32X = ueye.int(0)
        rect.s32Y = ueye.int(0)
        rect.s32Width = ueye.int(self._width)
        rect.s32Height = ueye.int(self._height)
        self._check(
            ueye.is_AOI(self._hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect, ueye.sizeof(rect)),
            "is_AOI",
        )
        self._mem_ptr = ueye.c_mem_p()
        self._mem_id = ueye.int()
        self._check(
            ueye.is_AllocImageMem(
                self._hcam,
                self._width,
                self._height,
                self._bpp,
                self._mem_ptr,
                self._mem_id,
            ),
            "is_AllocImageMem",
        )
        self._check(
            ueye.is_SetImageMem(self._hcam, self._mem_ptr, self._mem_id),
            "is_SetImageMem",
        )
        pitch = ueye.int()
        if ueye.is_GetImageMemPitch(self._hcam, pitch) == ueye.IS_SUCCESS:
            self._pitch = pitch.value
        else:
            self._pitch = self._width * (self._bpp // 8)

    def set_exposure(self, ms: float) -> float:
        if not self._open:
            raise CameraError("camera is not open")
        with self._lock:
            exposure = ueye.double(ms)
            self._check(
                ueye.is_Exposure(
                    self._hcam,
                    ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
                    exposure,
                    ueye.sizeof(exposure),
                ),
                "set exposure",
            )
            actual = ueye.double()
            self._check(
                ueye.is_Exposure(
                    self._hcam,
                    ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
                    actual,
                    ueye.sizeof(actual),
                ),
                "get exposure",
            )
            self._exposure_ms = float(actual.value)
        if abs(self._exposure_ms - ms) > 0.05 * ms:
            log.warning(
                "Requested exposure %.3f ms, camera uses %.3f ms", ms, self._exposure_ms
            )

        return self._exposure_ms

    def capture_frame(self) -> np.ndarray | None:
        with self._lock:
            if not self._open:
                return None
            ret = ueye.is_FreezeVideo(self._hcam, ueye.IS_WAIT)
            if ret != ueye.IS_SUCCESS:
                log.warning("FreezeVideo returned %d", ret)

                return None
            data = ueye.get_data(
                self._mem_ptr,
                self._width,
                self._height,
                self._bpp,
                self._pitch,
                copy=True,
            )

            return np.reshape(data, (self._height, self._width, 3))

    def info(self) -> dict:
        return {
            "model": self._sensor_name,
            "driver": "uEye",
            "width": self._width,
            "height": self._height,
            "exposure_ms_actual": self._exposure_ms or None,
            "simulated": False,
        }

    def close(self) -> None:
        with self._lock:
            if not self._open:
                return
            self._open = False
            ueye.is_StopLiveVideo(self._hcam, ueye.IS_FORCE_VIDEO_STOP)
            ueye.is_FreeImageMem(self._hcam, self._mem_ptr, self._mem_id)
            ueye.is_ExitCamera(self._hcam)
        log.info("Camera closed")


class SimCamera:

    simulated = True

    def __init__(self, width: int = 320, height: int = 240):
        self._width = width
        self._height = height
        self._open = False
        self._exposure_ms = 0.0
        self._shots = 0

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def size(self) -> tuple[int, int]:
        return self._width, self._height

    def open(self, *_args, **_kwargs) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def set_exposure(self, ms: float) -> float:
        self._exposure_ms = float(ms)

        return self._exposure_ms

    def capture_frame(self) -> np.ndarray | None:
        if not self._open:
            return None
        self._shots += 1
        ys, xs = np.mgrid[0 : self._height, 0 : self._width]
        value = 40 + 120 * (xs + ys + 7 * self._shots) / (
            self._width + self._height + 7 * self._shots
        )
        frame = np.stack([value, value * 0.8, value * 0.6], axis=-1)

        return np.clip(frame, 0, 255).astype(np.uint8)

    def info(self) -> dict:
        return {
            "model": "SimCamera",
            "driver": "simulation",
            "width": self._width,
            "height": self._height,
            "exposure_ms_actual": self._exposure_ms or None,
            "simulated": True,
        }
