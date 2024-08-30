from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mujoco_utils import mjcf_utils
from typing_extensions import Self

from mojo.elements import Body
from mojo.elements.element import TransformElement

if TYPE_CHECKING:
    from mojo import Mojo


class Camera(TransformElement):
    @staticmethod
    def get(
        mojo: Mojo,
        name: str,
        parent: TransformElement = None,
    ) -> Self:
        root_mjcf = mojo.root_element.mjcf if parent is None else parent.mjcf
        mjcf = mjcf_utils.safe_find(root_mjcf, "camera", name)
        return Camera(mojo, mjcf)

    @staticmethod
    def get_all(
        mojo: Mojo,
    ) -> list[Self]:
        try:
            mjcfs = mjcf_utils.safe_find_all(mojo.root_element.mjcf, "camera")
        except ValueError:
            mjcfs = []
        return [Camera(mojo, mjcf) for mjcf in mjcfs]

    @staticmethod
    def create(
        mojo: Mojo,
        parent: TransformElement = None,
        position: np.ndarray = None,
        quaternion: np.ndarray = None,
        fovy: float = None,
        focal: np.ndarray = None,
        sensor_size: np.ndarray = None,
    ) -> Self:
        position = np.array([0, 0, 0]) if position is None else position
        quaternion = np.array([1, 0, 0, 0]) if quaternion is None else quaternion
        if parent is not None and not isinstance(parent, Body):
            msg = "Parent must be of type body for camera."
            raise ValueError(msg)
        parent_mjcf = (
            mojo.root_element.mjcf.worldbody if parent is None else parent.mjcf
        )
        camera_params = {}
        if fovy:
            camera_params["fovy"] = fovy
        if focal:
            camera_params["focal"] = focal
        if sensor_size:
            camera_params["sensor_size"] = sensor_size
        new_camera = parent_mjcf.add(
            "camera",
            pos=position,
            quat=quaternion,
            **camera_params,
        )
        mojo.mark_dirty()
        return Camera(mojo, new_camera)

    def set_focal(self, focal: np.ndarray):
        if self.mjcf.sensorsize is None:
            self.mjcf.sensorsize = np.array([0, 0])
        if self.mjcf.resolution is None:
            self.mjcf.resolution = np.array([1, 1])
        self.mjcf.focal = focal
        self._mojo.mark_dirty()

    def get_focal(self) -> np.ndarray:
        return self.mjcf.focal

    def set_sensor_size(self, sensor_size: np.ndarray):
        # Either focal or focalpixel must be set
        if self.mjcf.focal is None or self.mjcf.focalpixel is None:
            self.mjcf.focal = np.array([0, 0])
        # Resolution must be set
        if self.mjcf.resolution is None:
            self.mjcf.resolution = np.array([1, 1])
        self.mjcf.sensorsize = np.array(sensor_size)
        self._mojo.mark_dirty()

    def get_sensor_size(self) -> np.ndarray:
        return self.mjcf.sensorsize

    def set_focal_pixel(self, focal_pixel: np.ndarray):
        if self.mjcf.sensorsize is None:
            self.mjcf.sensorsize = np.array([0, 0])
        if self.mjcf.resolution is None:
            self.mjcf.resolution = np.array([1, 1])
        self.mjcf.focalpixel = focal_pixel
        self._mojo.mark_dirty()

    def get_focal_pixel(self) -> np.ndarray:
        return self.mjcf.focalpixel

    def set_fovy(self, fovy: float):
        self.mjcf.fovy = fovy
        self._mojo.mark_dirty()

    def get_fovy(self) -> np.ndarray:
        return self.mjcf.fovy

    def render(
        self,
        resolution: tuple[int, int] = None,
        rgb: bool = True,
        depth: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        resolution = resolution or (128, 128)
        renderer = self._mojo.get_renderer(resolution)
        renderer.update_scene(self._mojo.data, self.name)
        result_to_return = None
        if rgb:
            result_to_return = rgb_render = renderer.render()
        if depth:
            renderer.enable_depth_rendering()
            depth_render = renderer.render()
            renderer.disable_depth_rendering()
            result_to_return = (rgb_render, depth_render) if rgb else depth_render
        return result_to_return
