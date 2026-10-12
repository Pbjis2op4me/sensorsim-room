"""Strict, engine-neutral scenario schema. Coordinates are metres, z is up."""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from sensorsim.acoustics.room.model import Bands, BoxObstacle, RoomModel, SURFACES

XYZ = tuple[float, float, float]
SOUNDS = ("transient", "speech_like", "tonal_machine", "broadband")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


@lru_cache(maxsize=32)
def material(name: str):
    # Reuse the public inversion catalogue adapter; no duplicated coefficients.
    from sensorsim.acoustics.room.fit import material_prior
    coefficients, evidence = material_prior(name)
    if evidence.get("profile") is None:
        raise ValueError(f"material has no cited absorption profile: {name}")
    return coefficients, evidence


class Surface(Strict):
    material: str | None = None
    absorption: Bands | None = None
    scattering: Bands = (0.,) * 7

    @model_validator(mode="after")
    def coefficients(self):
        if (self.material is None) == (self.absorption is None):
            raise ValueError("specify material OR absorption")
        if self.material is not None:
            material(self.material)
        if any(not 0 <= a <= 1 for a in (*self.scattering, *(self.absorption or ()))):
            raise ValueError("coefficients outside [0,1]")
        return self

    def alpha(self):
        return self.absorption if self.absorption is not None else material(self.material)[0]


class Room(Strict):
    dims_m: XYZ
    surfaces: dict[str, Surface]
    temperature_c: float = Field(default=20., ge=-20, le=50)
    humidity_pct: float = Field(default=50., ge=10, le=100)


class Obstacle(Strict):
    name: str
    min_xyz: XYZ
    size_xyz: XYZ
    material: str
    scattering: Bands = (.1,) * 7

    def model(self):
        return BoxObstacle(**self.model_dump(), absorption=material(self.material)[0])


class NovelGrid(Strict):
    source: XYZ
    x_m: tuple[float, ...]
    y_m: tuple[float, ...]
    z_m: tuple[float, ...]

    def positions(self):
        return {f"N{i+1}": xyz for i, xyz in enumerate(product(self.x_m, self.y_m, self.z_m))}


class Evaluation(Strict):
    heldout_mic: str
    heldout_speaker: str | None = None
    novel_grid: NovelGrid


class Acquisition(Strict):
    sequential: bool = True
    reference: bool = True
    lead_s: tuple[float, float] = (.1, 60.)
    trail_s: tuple[float, float] = (.1, 3.)
    clock_ppm: tuple[float, float] = (-300., 300.)
    jitter_s: tuple[float, float] = (-.02, .02)
    extraneous: tuple[Literal["speech_like", "handling", "door", "footsteps"], ...] = (
        "speech_like", "handling", "door", "footsteps")
    event_level_dbfs: float = Field(default=-32., le=-6.)

    @model_validator(mode="after")
    def bounds(self):
        for key in ("lead_s", "trail_s", "clock_ppm", "jitter_s"):
            a, b = getattr(self, key)
            if a > b:
                raise ValueError(f"unordered {key}")
        if min(self.lead_s + self.trail_s) < 0 or max(self.lead_s+self.trail_s) > 120:
            raise ValueError("lead/trail outside bounded [0,120] seconds")
        if max(abs(x) for x in self.clock_ppm) > 300:
            raise ValueError("clock outside +/-300 ppm")
        if max(abs(x) for x in self.jitter_s) > .1:
            raise ValueError("jitter outside +/-0.1 seconds")
        return self


class Render(Strict):
    profile: Literal["full", "guard", "smoke"] = "full"
    rir_duration_s: float = Field(default=1.5, gt=.1, le=4.)


class Scenario(Strict):
    version: Literal["room-closedloop/2"] = "room-closedloop/2"
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    variant: Literal["sequential", "idealised"] = "sequential"
    room: Room
    obstacles: tuple[Obstacle, ...] = ()
    positions: dict[str, XYZ]
    eval_positions: Evaluation
    real_sounds: tuple[Literal["transient", "speech_like", "tonal_machine", "broadband"], ...] = SOUNDS
    impairments: Literal["clean", "typical", "poor"] = "clean"
    acquisition: Acquisition = Field(default_factory=Acquisition)
    seed: int = Field(default=20260930, ge=0)
    render: Render = Field(default_factory=Render)

    @property
    def speakers(self):
        return {k: v for k, v in self.positions.items() if k.startswith("S")}

    @property
    def mics(self):
        return {k: v for k, v in self.positions.items() if k.startswith("M")}

    @property
    def all_positions(self):
        grid = self.eval_positions.novel_grid
        return {**self.positions, "NS1": grid.source, **grid.positions()}

    @property
    def train_pairs(self):
        e = self.eval_positions
        return [(s, m) for s in self.speakers if s != e.heldout_speaker
                for m in self.mics if m != e.heldout_mic]

    @property
    def heldout_pairs(self):
        train = set(self.train_pairs)
        return [(s, m) for s in self.speakers for m in self.mics if (s, m) not in train]

    @property
    def novel_pairs(self):
        return [("NS1", m) for m in self.eval_positions.novel_grid.positions()]

    def room_model(self):
        return RoomModel(room_id=self.id, shoebox_dims_m=self.room.dims_m,
                         absorption={s: v.alpha() for s, v in self.room.surfaces.items()},
                         scattering={s: v.scattering for s, v in self.room.surfaces.items()},
                         obstacles=tuple(o.model() for o in self.obstacles),
                         temperature_c=self.room.temperature_c, humidity_pct=self.room.humidity_pct,
                         provenance={"material_evidence": self.material_evidence(),
                                     "scope": "scenario assumptions; not measured room properties"})

    def material_evidence(self):
        names = {s.material for s in self.room.surfaces.values() if s.material}
        names.update(o.material for o in self.obstacles)
        return {n: material(n)[1] for n in sorted(names)}

    @model_validator(mode="after")
    def physical_and_split(self):
        if set(self.room.surfaces) != set(SURFACES):
            raise ValueError("exactly six room surfaces required")
        model = self.room_model()
        if "S1" not in self.speakers or "M1" not in self.mics:
            raise ValueError("S1 and M1 required")
        if self.variant == "sequential" and len(self.mics) != 3:
            raise ValueError("sequential protocol uses exactly three mic positions")
        if not 1 <= len(self.speakers) <= 4 or len(self.mics) < 3:
            raise ValueError("need 1..4 speaker placements and at least three mic positions")
        allowed = {*self.speakers, *self.mics, *(f"REF_{s}" for s in self.speakers)}
        if set(self.positions) != allowed:
            raise ValueError("positions must contain speakers, mics and their reference coordinates")
        if not self.real_sounds or len(set(self.real_sounds)) != len(self.real_sounds):
            raise ValueError("need distinct real sound types")
        for s, xyz in self.speakers.items():
            if not np.allclose(self.positions[f"REF_{s}"], np.asarray(xyz)+[.3, 0, 0], atol=1e-10):
                raise ValueError("REF must be 0.3 m on +x")
        e = self.eval_positions
        if e.heldout_mic not in self.mics or e.heldout_mic == "M1":
            raise ValueError("heldout mic must exist and preserve M1")
        if e.heldout_speaker is not None and (e.heldout_speaker not in self.speakers or e.heldout_speaker == "S1"):
            raise ValueError("heldout speaker must exist and preserve S1")
        if not self.novel_pairs or len(self.novel_pairs) > 32:
            raise ValueError("novel grid needs 1..32 pairs")
        positions = self.all_positions
        for name, xyz in positions.items():
            p = np.array(xyz)
            if np.any(p <= 0) or np.any(p >= model.shoebox_dims_m) or any(b.contains(p) for b in model.obstacles):
                raise ValueError(f"{name} lies outside free room interior")
        measured_xyz = set(self.positions.values())
        if any(xyz in measured_xyz for name, xyz in positions.items() if name.startswith("N")):
            raise ValueError("novel coordinate overlaps a measurement")
        for s, m in [*self.train_pairs, *self.heldout_pairs, *self.novel_pairs]:
            if np.linalg.norm(np.array(positions[s])-positions[m]) < .05:
                raise ValueError("source/receiver separation below 5 cm")
        return self


def load_scenario(path: str | Path) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(Path(path).read_text()))
