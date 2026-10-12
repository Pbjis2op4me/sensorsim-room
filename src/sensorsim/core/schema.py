"""Obstacle-box schema (pydantic v2) used by the room renderer.

Conventions:
- World frame: local ENU in metres, z = up, z = 0 at local ground.
- Azimuth: compass degrees, clockwise from north (0 = north, 90 = east).
- Time: audio-unit sample timing is carried by SampleTimeMap; no recording-clock simulator is part of the training path.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

Vec3 = tuple[float, float, float]


def _entity_id(value: str) -> str:
    if not value.strip() or "\0" in value:
        raise ValueError("entity ID must be nonempty and contain no NUL")
    return value


EntityId = Annotated[str, AfterValidator(_entity_id)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


SceneFace = Literal["roof", "floor", "front", "back", "left", "right"]


class SceneOpening(_Model):
    """An opening in a box face (open window, door gap, vent) or a different-material panel (e.g. glass).
    (u, v): face-local coordinates [m] of the opening centre relative to the face centre; u is horizontal
    along the face, v is up (for roof/floor: u along the box length, v along its width)."""
    face: SceneFace
    center_uv_m: tuple[float, float] = (0.0, 0.0)
    size_uv_m: tuple[float, float]
    kind: Literal["open", "panel"] = "open"
    material: str | None = Field(None, description="for kind='panel' (e.g. 'glass'); a free-form material label")


class SceneBox(_Model):
    """Oriented box obstacle used by the room renderer's acoustic propagation model.
    `base_center_enu_m` is the centre of the bottom face; `size_m` = (length, width, height), length along
    `heading_deg` (compass). Faces: front (+length), back, left, right, roof, floor."""
    id: EntityId
    base_center_enu_m: Vec3
    size_m: Vec3
    heading_deg: float = 0.0
    material: str = Field("concrete", description="default material label for all faces")
    face_materials: dict[str, str] = Field(default_factory=dict, description="per-face overrides {face: material}")
    openings: list[SceneOpening] = []

    @model_validator(mode="after")
    def _positive_size(self):
        if any(x <= 0 for x in self.size_m):
            raise ValueError("box size_m must have positive extents")
        return self
