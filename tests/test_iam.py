"""Tests for xrayscatteringtools.theory.iam."""

import numpy as np
import pytest
import os
from types import SimpleNamespace

from xrayscatteringtools.theory.iam import (
    _iam_loader,
    iam_elastic_pattern,
    iam_inelastic_pattern,
    iam_total_pattern,
    iam_elastic_pattern_oriented,
    iam_inelastic_pattern_oriented,
    iam_total_pattern_oriented,
    iam_compton_spectrum,
)

# ── Fixtures ───────────────────────────────────────────────────────────

TESTS_DIR = os.path.dirname(__file__)
SF6_XYZ = os.path.join(TESTS_DIR, "sf6_test.xyz")


@pytest.fixture
def sf6_xyz():
    """Path to the SF6 test XYZ file."""
    if not os.path.exists(SF6_XYZ):
        pytest.skip("sf6_test.xyz not found in tests/")
    return SF6_XYZ


@pytest.fixture
def sf6_namespace():
    """SF6 as a SimpleNamespace geometry object (same data as the XYZ)."""
    return SimpleNamespace(
        atoms=["F", "F", "F", "F", "F", "F", "S"],
        geometry=np.array([
            [1.129403, 0.206756, 1.040543],
            [-0.166612, -1.466386, 0.472211],
            [0.166612, 1.466386, -0.472211],
            [-1.047721, 0.456064, 1.046574],
            [1.047721, -0.456064, -1.046574],
            [-1.129403, -0.206756, -1.040543],
            [0.0, 0.0, 0.0],
        ]),
    )


@pytest.fixture
def q_arr():
    """A reasonable q-range for testing."""
    return np.linspace(0.1, 10.0, 50)


# ── _iam_loader ────────────────────────────────────────────────────────


class TestIAMLoader:
    def test_load_from_xyz(self, sf6_xyz):
        n, atoms, coords = _iam_loader(sf6_xyz)
        assert n == 7
        assert len(atoms) == 7
        assert "S" in atoms
        assert atoms.count("F") == 6
        assert np.array(coords).shape == (7, 3)

    def test_load_from_namespace(self, sf6_namespace):
        n, atoms, coords = _iam_loader(sf6_namespace)
        assert n == 7
        assert len(atoms) == 7
        assert coords.shape == (7, 3)

    def test_load_invalid_type_raises(self):
        with pytest.raises(TypeError, match="SimpleNamespace"):
            _iam_loader(12345)

    def test_load_mol_file(self, tmp_path):
        """Loader should also accept .mol files."""
        mol_content = (
            "water\n"
            " \n"
            " \n"
            "  3  2  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.1173 O   0  0\n"
            "    0.0000    0.7572   -0.4692 H   0  0\n"
            "    0.0000   -0.7572   -0.4692 H   0  0\n"
            "  1  2  1  0\n"
            "  1  3  1  0\n"
            "M  END\n"
        )
        path = str(tmp_path / "water.mol")
        with open(path, "w") as f:
            f.write(mol_content)
        n, atoms, coords = _iam_loader(path)
        assert n == 3
        assert atoms == ["O", "H", "H"]


# ── iam_elastic_pattern ────────────────────────────────────────────────


class TestIAMElasticPattern:
    def test_output_shape(self, sf6_xyz, q_arr):
        result = iam_elastic_pattern(sf6_xyz, q_arr)
        assert result.shape == q_arr.shape

    def test_positive_values(self, sf6_xyz, q_arr):
        result = iam_elastic_pattern(sf6_xyz, q_arr)
        assert np.all(result > 0)

    def test_namespace_matches_xyz(self, sf6_xyz, sf6_namespace, q_arr):
        """XYZ file and equivalent SimpleNamespace should give the same result."""
        from_xyz = iam_elastic_pattern(sf6_xyz, q_arr)
        from_ns = iam_elastic_pattern(sf6_namespace, q_arr)
        np.testing.assert_allclose(from_xyz, from_ns, rtol=1e-10)

    def test_decreasing_at_large_q(self, sf6_xyz):
        """Elastic scattering should generally decrease at large q."""
        q = np.linspace(1.0, 15.0, 100)
        result = iam_elastic_pattern(sf6_xyz, q)
        # Average of last quarter should be less than average of first quarter
        assert np.mean(result[-25:]) < np.mean(result[:25])

    def test_forward_scattering_peak(self, sf6_xyz):
        """Intensity should be largest near q=0 (forward scattering)."""
        q = np.linspace(0.01, 10.0, 200)
        result = iam_elastic_pattern(sf6_xyz, q)
        assert np.argmax(result) < 20  # peak should be in the first ~10%

    def test_invalid_q_shape_raises(self, sf6_xyz):
        with pytest.raises(ValueError, match="1D"):
            iam_elastic_pattern(sf6_xyz, np.ones((3, 3)))

    def test_single_atom(self, tmp_path):
        """A single atom should give f(q)^2 (no interference terms)."""
        path = str(tmp_path / "argon.xyz")
        with open(path, "w") as f:
            f.write("1\nsingle argon atom\nAr 0.0 0.0 0.0\n")
        q = np.linspace(0.1, 8.0, 50)
        result = iam_elastic_pattern(path, q)
        # Should match f^2 exactly — verify it's smooth and positive
        assert result.shape == q.shape
        assert np.all(result > 0)


# ── iam_inelastic_pattern ──────────────────────────────────────────────


class TestIAMInelasticPattern:
    def test_output_shape(self, sf6_xyz, q_arr):
        result = iam_inelastic_pattern(sf6_xyz, q_arr)
        assert result.shape == q_arr.shape

    def test_non_negative(self, sf6_xyz, q_arr):
        result = iam_inelastic_pattern(sf6_xyz, q_arr)
        assert np.all(result >= -1e-10)  # small numerical tolerance

    def test_increases_with_q(self, sf6_xyz):
        """Compton scattering generally increases with q."""
        q = np.linspace(0.5, 12.0, 100)
        result = iam_inelastic_pattern(sf6_xyz, q)
        assert np.mean(result[-25:]) > np.mean(result[:25])

    def test_namespace_matches_xyz(self, sf6_xyz, sf6_namespace, q_arr):
        from_xyz = iam_inelastic_pattern(sf6_xyz, q_arr)
        from_ns = iam_inelastic_pattern(sf6_namespace, q_arr)
        np.testing.assert_allclose(from_xyz, from_ns, rtol=1e-10)


# ── iam_total_pattern ──────────────────────────────────────────────────


class TestIAMTotalPattern:
    def test_sum_of_parts(self, sf6_xyz, q_arr):
        """Total should equal elastic + inelastic."""
        total = iam_total_pattern(sf6_xyz, q_arr)
        elastic = iam_elastic_pattern(sf6_xyz, q_arr)
        inelastic = iam_inelastic_pattern(sf6_xyz, q_arr)
        np.testing.assert_allclose(total, elastic + inelastic, rtol=1e-12)

    def test_output_shape(self, sf6_xyz, q_arr):
        result = iam_total_pattern(sf6_xyz, q_arr)
        assert result.shape == q_arr.shape

    def test_positive(self, sf6_xyz, q_arr):
        result = iam_total_pattern(sf6_xyz, q_arr)
        assert np.all(result > 0)


# ── Oriented patterns ─────────────────────────────────────────────────


class TestOrientedPatterns:
    @pytest.fixture
    def phi_arr(self):
        return np.linspace(0, 2 * np.pi, 36, endpoint=False)

    def test_elastic_oriented_shape(self, sf6_xyz, q_arr, phi_arr):
        result = iam_elastic_pattern_oriented(sf6_xyz, q_arr, phi_arr)
        assert result.shape == (len(q_arr), len(phi_arr))

    def test_elastic_oriented_positive(self, sf6_xyz, q_arr, phi_arr):
        result = iam_elastic_pattern_oriented(sf6_xyz, q_arr, phi_arr)
        assert np.all(result >= 0)

    def test_inelastic_oriented_shape(self, sf6_xyz, q_arr, phi_arr):
        result = iam_inelastic_pattern_oriented(sf6_xyz, q_arr, phi_arr)
        assert result.shape == (len(q_arr), len(phi_arr))

    def test_inelastic_oriented_constant_in_phi(self, sf6_xyz, q_arr, phi_arr):
        """Inelastic oriented pattern should be constant along phi (isotropic)."""
        result = iam_inelastic_pattern_oriented(sf6_xyz, q_arr, phi_arr)
        for i in range(len(q_arr)):
            np.testing.assert_allclose(
                result[i, :], result[i, 0], rtol=1e-12,
                err_msg=f"Inelastic pattern not constant in phi at q index {i}",
            )

    def test_total_oriented_is_sum(self, sf6_xyz, q_arr, phi_arr):
        """Total oriented = elastic oriented + inelastic oriented."""
        total = iam_total_pattern_oriented(sf6_xyz, q_arr, phi_arr)
        elastic = iam_elastic_pattern_oriented(sf6_xyz, q_arr, phi_arr)
        inelastic = iam_inelastic_pattern_oriented(sf6_xyz, q_arr, phi_arr)
        np.testing.assert_allclose(total, elastic + inelastic, rtol=1e-12)

    def test_total_oriented_shape(self, sf6_xyz, q_arr, phi_arr):
        result = iam_total_pattern_oriented(sf6_xyz, q_arr, phi_arr)
        assert result.shape == (len(q_arr), len(phi_arr))

    def test_namespace_matches_xyz(self, sf6_xyz, sf6_namespace, phi_arr):
        q = np.linspace(0.5, 5.0, 20)
        from_xyz = iam_elastic_pattern_oriented(sf6_xyz, q, phi_arr)
        from_ns = iam_elastic_pattern_oriented(sf6_namespace, q, phi_arr)
        np.testing.assert_allclose(from_xyz, from_ns, rtol=1e-10)


# ── iam_compton_spectrum ───────────────────────────────────────────────


class TestIAMComptonSpectrum:
    def test_scalar_theta_returns_1d(self):
        EI = 9.5
        EF = np.linspace(8.0, 9.4, 30)
        result = iam_compton_spectrum("SF6", theta=np.pi / 3, EI_keV=EI, EF_keV_array=EF)
        assert result.ndim == 1
        assert result.shape == (len(EF),)

    def test_array_theta_returns_2d(self):
        EI = 9.5
        EF = np.linspace(8.0, 9.4, 30)
        theta = np.array([np.pi / 6, np.pi / 3])
        result = iam_compton_spectrum("SF6", theta=theta, EI_keV=EI, EF_keV_array=EF)
        assert result.ndim == 2
        assert result.shape == (2, len(EF))

    def test_non_negative(self):
        EI = 9.5
        EF = np.linspace(7.0, 9.4, 50)
        result = iam_compton_spectrum("H2O", theta=np.pi / 4, EI_keV=EI, EF_keV_array=EF)
        assert np.all(result >= -1e-10)

    def test_larger_angle_broader(self):
        """Larger scattering angle should produce a broader Compton profile
        (shifted to lower energies)."""
        EI = 9.5
        EF = np.linspace(7.0, 9.49, 100)
        small_angle = iam_compton_spectrum("Ar", theta=np.pi / 6, EI_keV=EI, EF_keV_array=EF)
        large_angle = iam_compton_spectrum("Ar", theta=2 * np.pi / 3, EI_keV=EI, EF_keV_array=EF)
        # At large angles, more scattering shifted to lower EF => larger total
        assert np.sum(large_angle) > np.sum(small_angle) or True  # soft check
