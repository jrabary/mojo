import re
from collections.abc import Callable

import mujoco.viewer
import numpy as np
from dm_control import mjcf

from mojo.elements.actuators import (
    Actuator,
    GeneralActuator,
    MotorActuator,
    PositionActuator,
    VelocityActuator,
)
from mojo.elements.body import Body
from mojo.elements.element import TransformElement
from mojo.elements.model import MujocoModel
from mojo.elements.utils import AssetStore, resolve_freejoints


class Mojo:
    def __init__(
        self,
        base_model_path: str,
        timestep: float = 0.01,
        texture_store_capacity: int = AssetStore.DEFAULT_CAPACITY,
        mesh_store_capacity: int = AssetStore.DEFAULT_CAPACITY,
    ):
        model_mjcf = mjcf.from_path(base_model_path)
        self.root_element = MujocoModel(self, model_mjcf)
        self._texture_store: AssetStore = AssetStore(texture_store_capacity)
        self._mesh_store: AssetStore = AssetStore(mesh_store_capacity)
        self._camera_renderers: dict[tuple[int, int], mujoco.Renderer] | None = {}
        self._dirty = True
        self._passive_dirty = False
        self._passive_viewer_handle = None
        self.set_timestep(timestep)

    def _create_physics_from_model(self):
        self._physics = mjcf.Physics.from_mjcf_model(self.root_element.mjcf)
        self._physics.legacy_step = False
        self._dirty = False
        self.clear_renderers()

    @property
    def physics(self):
        if self._dirty:
            self._create_physics_from_model()
        return self._physics

    @property
    def model(self):
        if self._dirty:
            self._create_physics_from_model()
        return self._physics.model.ptr

    @property
    def data(self):
        if self._dirty:
            self._create_physics_from_model()
        return self._physics.data.ptr

    @property
    def actuators(self) -> list[Actuator]:
        act_root = self.root_element.mjcf.actuator
        actuators_to_return = []
        for act_class, attrib_name in [
            (GeneralActuator, "general"),
            (MotorActuator, "motor"),
            (PositionActuator, "position"),
            (VelocityActuator, "velocity"),
        ]:
            for act in getattr(act_root, attrib_name):
                mojo_act = act_class(self, act)
                actuators_to_return.append(mojo_act)
        return actuators_to_return

    def set_timestep(self, timestep: float):
        self.root_element.mjcf.option.timestep = timestep

    def launch_viewer(self, passive: bool = False, name: str = None) -> None:
        if name:
            self.root_element.mjcf.model = name
        # passive viewer does not step.
        if self._dirty:
            self._create_physics_from_model()
        if passive:
            self._passive_dirty = False
            self._passive_viewer_handle = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        else:
            mujoco.viewer.launch(self._physics.model.ptr, self._physics.data.ptr)

    def sync_passive_viewer(self):
        if self._passive_viewer_handle is None:
            msg = "You do not have a passive viewer running."
            raise RuntimeError(msg)
        if self._passive_dirty:
            self._passive_dirty = False
            self._create_physics_from_model()
            self._passive_viewer_handle._sim().load(
                self._physics.model.ptr,
                self._physics.data.ptr,
                "",
            )
        self._passive_viewer_handle.sync()

    def close_passive_viewer(self):
        if self._passive_viewer_handle is None:
            msg = "You do not have a passive viewer running."
            raise RuntimeError(msg)
        self._passive_viewer_handle.close()

    def mark_dirty(self):
        self._passive_dirty = True
        self._dirty = True

    def step(self):
        """Advances the physics state by 1 step."""
        if self._dirty:
            self._create_physics_from_model()
        self.physics.step()

    def get_material(self, path: str) -> mjcf.Element | None:
        return self._texture_store.get(path)

    def store_material(self, path: str, material_mjcf: mjcf.Element) -> None:
        self._texture_store.add(path, material_mjcf)

    def get_mesh(self, path: str) -> mjcf.Element | None:
        return self._mesh_store.get(path)

    def store_mesh(self, path: str, mesh_mjcf: mjcf.Element) -> None:
        self._mesh_store.add(path, mesh_mjcf)

    def load_model(
        self,
        path: str,
        parent: TransformElement = None,
        on_loaded: Callable[[mjcf.RootElement], None] | None = None,
        handle_freejoints: bool = False,
    ):
        """Load a Mujoco model from xml file and attach to specified parent element.

        :param path: The file path to the Mujoco model XML file.
        :param parent: Parent MujocoElement to which the loaded model will be attached.
        If None, it attaches to the root element.
        :param on_loaded: Optional callback to be executed after model is loaded.
        Use it to customize the Mujoco model before attaching it to the parent.
        :param handle_freejoints: If true handles <freejoint/> elements.
        Freejoint bodies will be re-parented to the worldbody.
        :return: A Body element representing the attached model.
        """
        model_mjcf = mjcf.from_path(path)
        if on_loaded is not None:
            on_loaded(model_mjcf)
        attach_site = self.root_element.mjcf if parent is None else parent.mjcf
        attached_model_mjcf = attach_site.attach(model_mjcf)
        if handle_freejoints:
            root_model_mjcf = resolve_freejoints(
                self.root_element.mjcf,
                attached_model_mjcf,
            )
            self.root_element = TransformElement(self, root_model_mjcf)
        self.mark_dirty()
        return Body(self, attached_model_mjcf)

    def set_headlight(
        self,
        active: bool = True,
        ambient: np.ndarray = None,
        diffuse: np.ndarray = None,
        specular: np.ndarray = None,
    ):
        ambient = np.array([0.1, 0.1, 0.1]) if ambient is None else ambient
        diffuse = np.array([0.4, 0.4, 0.4]) if diffuse is None else diffuse
        specular = np.array([0.5, 0.5, 0.5]) if specular is None else specular
        self.root_element.mjcf.visual.headlight.ambient = ambient
        self.root_element.mjcf.visual.headlight.diffuse = diffuse
        self.root_element.mjcf.visual.headlight.specular = specular
        self.root_element.mjcf.visual.headlight.active = active
        self.mark_dirty()

    def get_renderer(self, resolution: tuple[int, int]):
        if resolution not in self._camera_renderers:
            self._camera_renderers[resolution] = mujoco.Renderer(
                self.model,
                resolution[0],
                resolution[1],
            )
        return self._camera_renderers[resolution]

    def clear_renderers(self):
        for renderer in self._camera_renderers.values():
            renderer.close()
        self._camera_renderers.clear()

    def __str__(self):
        xml = self.root_element.mjcf.to_xml_string()

        # Match the <mesh> tags and remove the hash from the file parameter
        pattern = r'(<mesh[^>]*file=")([^"]+)-[a-fA-F0-9]{40}(\.STL")'

        # Replace the matched pattern with the modified file parameter
        modified_xml = re.sub(pattern, r"\1\2\3", xml)

        return modified_xml
