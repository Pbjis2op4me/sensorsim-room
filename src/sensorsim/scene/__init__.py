"""Scene geometry for the acoustics: a flat ground plane plus oriented boxes (furniture, walls, rooms).

Physics (transmission loss, reflection, diffraction) lives in `sensorsim.acoustics`; this package only
answers geometric questions, vectorised over many segments.
"""
from .geometry import FACES, Box, Face, Scene
from .raycore import RayCore, RayQueries, TriangleMesh, ClosestHit, Crossings

__all__ = ["Box", "Face", "FACES", "Scene",
           "RayCore", "RayQueries", "TriangleMesh", "ClosestHit", "Crossings"]
