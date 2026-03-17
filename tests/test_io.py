"""Tests for xrayscatteringtools.io."""

import numpy as np
import pytest
import h5py
import yaml
import os
import tempfile

from xrayscatteringtools.io import (
    runNumToString,
    read_xyz,
    write_xyz,
    read_mol,
    is_leaf,
    get_leaves,
    get_tree,
    get_config,
    get_config_for_runs,
    get_data_paths,
)


# ── runNumToString ─────────────────────────────────────────────────────


class TestRunNumToString:
    def test_single_digit(self):
        assert runNumToString(1) == "0001"

    def test_two_digits(self):
        assert runNumToString(42) == "0042"

    def test_three_digits(self):
        assert runNumToString(123) == "0123"

    def test_four_digits(self):
        assert runNumToString(9999) == "9999"

    def test_five_digits(self):
        # zfill(4) still works, just returns the full number
        assert runNumToString(12345) == "12345"

    def test_zero(self):
        assert runNumToString(0) == "0000"


# ── read_xyz / write_xyz round-trip ────────────────────────────────────


class TestReadWriteXYZ:
    """Tests for XYZ file I/O."""

    @pytest.fixture
    def sample_xyz(self, tmp_path):
        """Write a minimal XYZ file and return its path."""
        content = (
            "3\n"
            "water molecule\n"
            "O  0.000000  0.000000  0.117300\n"
            "H  0.000000  0.757200 -0.469200\n"
            "H  0.000000 -0.757200 -0.469200\n"
        )
        path = tmp_path / "water.xyz"
        path.write_text(content)
        return str(path)

    def test_read_xyz_num_atoms(self, sample_xyz):
        n, _, _, _ = read_xyz(sample_xyz)
        assert n == 3

    def test_read_xyz_comment(self, sample_xyz):
        _, comment, _, _ = read_xyz(sample_xyz)
        assert comment == "water molecule"

    def test_read_xyz_atoms(self, sample_xyz):
        _, _, atoms, _ = read_xyz(sample_xyz)
        assert atoms == ["O", "H", "H"]

    def test_read_xyz_coords_shape(self, sample_xyz):
        _, _, _, coords = read_xyz(sample_xyz)
        assert len(coords) == 3
        assert all(len(c) == 3 for c in coords)

    def test_read_xyz_coords_values(self, sample_xyz):
        _, _, _, coords = read_xyz(sample_xyz)
        assert coords[0] == pytest.approx((0.0, 0.0, 0.117300))
        assert coords[1][1] == pytest.approx(0.757200)

    def test_write_then_read_roundtrip(self, tmp_path):
        """write_xyz -> read_xyz should return the same data."""
        path = str(tmp_path / "test.xyz")
        atoms = ["C", "O"]
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        comment = "test molecule"
        write_xyz(path, comment, atoms, coords)

        n, c, a, co = read_xyz(path)
        assert n == 2
        assert c == comment
        assert a == atoms
        np.testing.assert_allclose(co, coords, atol=1e-6)

    def test_write_xyz_atomic_numbers(self, tmp_path):
        """write_xyz should accept integer atomic numbers and convert them."""
        path = str(tmp_path / "nums.xyz")
        atoms = [6, 8]  # C, O
        coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
        write_xyz(path, "numbers", atoms, coords)

        n, _, a, _ = read_xyz(path)
        assert n == 2
        assert a == ["C", "O"]

    def test_read_existing_sf6(self):
        """Read the SF6 test file shipped with the repo."""
        sf6_path = os.path.join(
            os.path.dirname(__file__), "sf6_test.xyz"
        )
        if not os.path.exists(sf6_path):
            pytest.skip("sf6_test.xyz not found")
        n, comment, atoms, coords = read_xyz(sf6_path)
        assert n == 7
        assert "S" in atoms
        assert atoms.count("F") == 6


# ── HDF5 helpers ───────────────────────────────────────────────────────


class TestHDF5Helpers:
    """Tests for is_leaf, get_leaves, and get_tree."""

    @pytest.fixture
    def h5_file(self, tmp_path):
        """Create a temporary HDF5 file with groups and datasets."""
        path = str(tmp_path / "test.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("scalar", data=42)
            f.create_dataset("array", data=np.arange(10))
            g = f.create_group("group")
            g.create_dataset("nested", data=np.ones((3, 3)))
        return path

    def test_is_leaf_dataset(self, h5_file):
        with h5py.File(h5_file, "r") as f:
            assert is_leaf(f["scalar"]) is True
            assert is_leaf(f["array"]) is True
            assert is_leaf(f["group/nested"]) is True

    def test_is_leaf_group(self, h5_file):
        with h5py.File(h5_file, "r") as f:
            assert is_leaf(f["group"]) is False

    def test_get_leaves(self, h5_file):
        result = {}
        with h5py.File(h5_file, "r") as f:
            get_leaves(f, saveto=result)
        assert "scalar" in result
        assert "array" in result
        assert "group/nested" in result
        assert result["scalar"] == 42
        np.testing.assert_array_equal(result["array"], np.arange(10))
        np.testing.assert_array_equal(result["group/nested"], np.ones((3, 3)))

    def test_get_leaves_verbose(self, h5_file, capsys):
        result = {}
        with h5py.File(h5_file, "r") as f:
            get_leaves(f, saveto=result, verbose=True)
        captured = capsys.readouterr()
        assert "scalar" in captured.out
        assert "array" in captured.out

    def test_get_leaves_no_saveto(self, h5_file):
        """Should not raise even when saveto is None."""
        with h5py.File(h5_file, "r") as f:
            get_leaves(f, saveto=None)

    def test_get_tree(self, h5_file, capsys):
        with h5py.File(h5_file, "r") as f:
            get_tree(f)
        captured = capsys.readouterr()
        assert "scalar" in captured.out
        assert "array" in captured.out
        assert "group" in captured.out
        assert "group/nested" in captured.out


# ── YAML config helpers ────────────────────────────────────────────────


class TestConfigHelpers:
    """Tests for get_config, get_config_for_runs, and get_data_paths."""

    @pytest.fixture
    def config_file(self, tmp_path):
        """Write a minimal YAML config and return its path."""
        config = {
            "data_paths": [
                {"runs": [1, 50], "path": "/data/exp1/"},
                {"runs": [51, 100], "path": "/data/exp2/"},
            ],
            "calibration": [
                {"runs": [1, 50], "file": "calib_v1.h5"},
                {"runs": [51, 100], "file": "calib_v2.h5"},
            ],
            "beamline": "CXI",
        }
        path = str(tmp_path / "config.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f)
        return path

    def test_get_config(self, config_file):
        assert get_config("beamline", config_path=config_file) == "CXI"

    def test_get_config_list(self, config_file):
        result = get_config("data_paths", config_path=config_file)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_config_for_runs_single(self, config_file):
        result = get_config_for_runs(5, "data_paths", "path", config_path=config_file)
        assert result == "/data/exp1/"

    def test_get_config_for_runs_multiple(self, config_file):
        result = get_config_for_runs([5, 75], "data_paths", "path", config_path=config_file)
        assert result == ["/data/exp1/", "/data/exp2/"]

    def test_get_config_for_runs_boundary(self, config_file):
        # Boundaries should be inclusive
        assert get_config_for_runs(1, "data_paths", "path", config_path=config_file) == "/data/exp1/"
        assert get_config_for_runs(50, "data_paths", "path", config_path=config_file) == "/data/exp1/"
        assert get_config_for_runs(51, "data_paths", "path", config_path=config_file) == "/data/exp2/"
        assert get_config_for_runs(100, "data_paths", "path", config_path=config_file) == "/data/exp2/"

    def test_get_config_for_runs_missing_raises(self, config_file):
        with pytest.raises(ValueError, match="No data_paths/path value found"):
            get_config_for_runs(999, "data_paths", "path", config_path=config_file)

    def test_get_config_for_runs_different_key(self, config_file):
        result = get_config_for_runs(25, "calibration", "file", config_path=config_file)
        assert result == "calib_v1.h5"

    def test_get_data_paths_single(self, config_file):
        result = get_data_paths(10, config_path=config_file)
        assert result == "/data/exp1/"

    def test_get_data_paths_multiple(self, config_file):
        result = get_data_paths([10, 60], config_path=config_file)
        assert result == ["/data/exp1/", "/data/exp2/"]

    def test_get_data_paths_missing_raises(self, config_file):
        with pytest.raises(ValueError):
            get_data_paths(200, config_path=config_file)

    def test_config_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            get_config("anything", config_path="nonexistent_config.yaml")


# ── read_mol ───────────────────────────────────────────────────────────


class TestReadMol:
    """Tests for MOL file reading."""

    @pytest.fixture
    def sample_mol(self, tmp_path):
        """Write a minimal V2000 MOL file and return its path."""
        # Minimal methane-like structure: C with 4 H
        content = (
            "methane\n"
            " prog_info_line\n"
            " comment line\n"
            "  5  4  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0\n"
            "    0.6300    0.6300    0.6300 H   0  0\n"
            "   -0.6300   -0.6300    0.6300 H   0  0\n"
            "   -0.6300    0.6300   -0.6300 H   0  0\n"
            "    0.6300   -0.6300   -0.6300 H   0  0\n"
            "  1  2  1  0\n"
            "  1  3  1  0\n"
            "  1  4  1  0\n"
            "  1  5  1  0\n"
            "M  END\n"
        )
        path = tmp_path / "methane.mol"
        path.write_text(content)
        return str(path)

    def test_mol_metadata(self, sample_mol):
        name, prog, comment, *_ = read_mol(sample_mol)
        assert name == "methane"
        assert "prog_info_line" in prog
        assert "comment" in comment

    def test_mol_atom_count(self, sample_mol):
        _, _, _, n_atoms, n_bonds, _, _, _, _ = read_mol(sample_mol)
        assert n_atoms == 5
        assert n_bonds == 4

    def test_mol_atoms(self, sample_mol):
        _, _, _, _, _, atoms, _, _, _ = read_mol(sample_mol)
        assert atoms[0] == "C"
        assert atoms.count("H") == 4

    def test_mol_coords(self, sample_mol):
        _, _, _, _, _, _, coords, _, _ = read_mol(sample_mol)
        assert len(coords) == 5
        assert coords[0] == pytest.approx((0.0, 0.0, 0.0))
        assert coords[1] == pytest.approx((0.63, 0.63, 0.63))

    def test_mol_bonds(self, sample_mol):
        _, _, _, _, _, _, _, bonds, _ = read_mol(sample_mol)
        assert len(bonds) == 4
        # Each bond should connect atom 1 (C) to atoms 2-5 (H), single bond
        for b in bonds:
            assert b[0] == 1  # first atom is always C (index 1)
            assert b[2] == 1  # single bond

    def test_mol_properties(self, sample_mol):
        _, _, _, _, _, _, _, _, props = read_mol(sample_mol)
        assert "atom_properties" in props
        assert "bond_properties" in props
        assert len(props["atom_properties"]) == 5
        assert len(props["bond_properties"]) == 4
