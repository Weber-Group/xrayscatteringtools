"""Tests for xrayscatteringtools.theory.geometries."""

import numpy as np
import pytest
import h5py
from types import SimpleNamespace

from xrayscatteringtools.theory import geometries
from xrayscatteringtools.theory.geometries import (
    __all__ as GEOMETRIES_ALL,
    _make_default_obj,
    _make_default_docstring,
)


# ── Module-level lazy loading ──────────────────────────────────────────


class TestGeometryLazyLoading:
    """Each name in __all__ should load a valid geometry object."""

    @pytest.mark.parametrize("name", GEOMETRIES_ALL)
    def test_loads_without_error(self, name):
        obj = getattr(geometries, name)
        assert obj is not None

    @pytest.mark.parametrize("name", GEOMETRIES_ALL)
    def test_has_required_attributes(self, name):
        obj = getattr(geometries, name)
        for attr in ("geometry", "molecule", "method", "basis_set",
                      "n_electrons", "charge", "energy", "atoms", "notes"):
            assert hasattr(obj, attr), f"Missing attribute '{attr}' on {name}"

    @pytest.mark.parametrize("name", GEOMETRIES_ALL)
    def test_geometry_shape(self, name):
        obj = getattr(geometries, name)
        assert obj.geometry.ndim == 2
        assert obj.geometry.shape[1] == 3

    @pytest.mark.parametrize("name", GEOMETRIES_ALL)
    def test_atoms_length_matches_geometry(self, name):
        obj = getattr(geometries, name)
        assert len(obj.atoms) == obj.geometry.shape[0]

    @pytest.mark.parametrize("name", GEOMETRIES_ALL)
    def test_n_electrons_positive(self, name):
        obj = getattr(geometries, name)
        assert obj.n_electrons > 0

    @pytest.mark.parametrize("name", GEOMETRIES_ALL)
    def test_molecule_non_empty(self, name):
        obj = getattr(geometries, name)
        assert isinstance(obj.molecule, str)
        assert len(obj.molecule) > 0

    @pytest.mark.parametrize("name", GEOMETRIES_ALL)
    def test_has_docstring(self, name):
        obj = getattr(geometries, name)
        assert obj.__doc__ is not None
        assert len(obj.__doc__) > 50

    def test_invalid_attribute_raises(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(geometries, "NONEXISTENT_GEOMETRY_XYZ")


# ── __dir__ ────────────────────────────────────────────────────────────


class TestDir:
    def test_dir_returns_all_sorted(self):
        result = geometries.__dir__()
        assert result == sorted(GEOMETRIES_ALL)

    def test_dir_contains_all_entries(self):
        result = geometries.__dir__()
        for name in GEOMETRIES_ALL:
            assert name in result


# ── _make_default_obj ──────────────────────────────────────────────────


class TestMakeDefaultObj:
    @pytest.fixture
    def h5_geometry(self, tmp_path):
        """Create a temporary HDF5 geometry file."""
        path = str(tmp_path / "test_geom.h5")
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        with h5py.File(path, "w") as f:
            f.create_dataset("geometry", data=coords)
            f.attrs["molecule"] = "H2"
            f.attrs["method"] = "CCSD(T)"
            f.attrs["basis_set"] = "cc-pVTZ"
            f.attrs["n_electrons"] = 2
            f.attrs["charge"] = 0
            f.attrs["energy"] = -1.172
            f.attrs["atoms"] = ["H", "H"]
            f.attrs["notes"] = "test geometry"
        return path

    def test_loads_all_fields(self, h5_geometry):
        with h5py.File(h5_geometry, "r") as f:
            obj = _make_default_obj(f)
        np.testing.assert_array_equal(
            obj.geometry, [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
        )
        assert obj.molecule == "H2"
        assert obj.method == "CCSD(T)"
        assert obj.basis_set == "cc-pVTZ"
        assert obj.n_electrons == 2
        assert obj.charge == 0
        assert obj.energy == pytest.approx(-1.172)
        assert list(obj.atoms) == ["H", "H"]
        assert obj.notes == "test geometry"

    def test_returns_simplenamespace(self, h5_geometry):
        with h5py.File(h5_geometry, "r") as f:
            obj = _make_default_obj(f)
        assert isinstance(obj, SimpleNamespace)


# ── _make_default_docstring ────────────────────────────────────────────


class TestMakeDefaultDocstring:
    def test_contains_molecule_and_method(self):
        obj = SimpleNamespace(
            geometry=np.zeros((3, 3)),
            molecule="H2O",
            method="MP2",
            basis_set="aug-cc-pVDZ",
            n_electrons=10,
            charge=0,
            energy=-76.0,
            atoms=["O", "H", "H"],
            notes="optimized geometry",
        )
        doc = _make_default_docstring(obj)
        assert "H2O" in doc
        assert "MP2" in doc
        assert "aug-cc-pVDZ" in doc

    def test_contains_standard_sections(self):
        obj = SimpleNamespace(
            geometry=np.zeros((2, 3)),
            molecule="N2",
            method="HF",
            basis_set="STO-3G",
            n_electrons=14,
            charge=0,
            energy=-108.0,
            atoms=["N", "N"],
            notes="minimal basis",
        )
        doc = _make_default_docstring(obj)
        assert "Attributes" in doc
        assert "geometry" in doc
        assert "atoms" in doc
        assert "n_electrons" in doc
        assert "charge" in doc
        assert "energy" in doc
        assert "Notes" in doc

    def test_notes_appear_in_docstring(self):
        note_text = "Special relativistic corrections applied"
        obj = SimpleNamespace(
            geometry=np.zeros((1, 3)),
            molecule="Ar",
            method="CCSD",
            basis_set="cc-pVQZ",
            n_electrons=18,
            charge=0,
            energy=-527.0,
            atoms=["Ar"],
            notes=note_text,
        )
        doc = _make_default_docstring(obj)
        assert note_text in doc
