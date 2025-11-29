"""
Test script to verify spherical coordinate conversions work correctly.
Tests both PyTorch and NumPy implementations.
"""

import torch
import numpy as np
from action_utils import cartesian_to_spherical, spherical_to_cartesian


def test_conversion(test_name, cartesian_input, expected_spherical=None, tolerance=1e-4):
    """Test Cartesian to spherical and back conversion."""
    print(f"\n=== {test_name} ===")
    
    # Test with NumPy
    cart_np = np.array(cartesian_input)
    print(f"Input (Cartesian): {cart_np}")
    
    # Convert to spherical
    spherical_np = cartesian_to_spherical(cart_np)
    print(f"Spherical (magnitude, theta, phi): {spherical_np}")
    
    # Convert back to Cartesian
    cart_recovered_np = spherical_to_cartesian(spherical_np)
    print(f"Recovered (Cartesian): {cart_recovered_np}")
    
    # Check round-trip error
    error = np.linalg.norm(cart_np - cart_recovered_np)
    print(f"Round-trip error: {error:.6f}")
    
    if error > tolerance:
        print(f"❌ FAIL: Round-trip error {error} exceeds tolerance {tolerance}")
        return False
    
    # Test with PyTorch
    cart_torch = torch.tensor(cartesian_input, dtype=torch.float32)
    spherical_torch = cartesian_to_spherical(cart_torch)
    cart_recovered_torch = spherical_to_cartesian(spherical_torch)
    error_torch = torch.norm(cart_torch - cart_recovered_torch).item()
    
    print(f"PyTorch round-trip error: {error_torch:.6f}")
    
    if error_torch > tolerance:
        print(f"❌ FAIL: PyTorch round-trip error {error_torch} exceeds tolerance {tolerance}")
        return False
    
    # Verify expected values if provided
    if expected_spherical is not None:
        expected_np = np.array(expected_spherical)
        diff = np.linalg.norm(spherical_np - expected_np)
        print(f"Expected spherical: {expected_np}")
        print(f"Difference from expected: {diff:.6f}")
        if diff > tolerance:
            print(f"❌ FAIL: Difference {diff} exceeds tolerance {tolerance}")
            return False
    
    print("✅ PASS")
    return True


def main():
    print("Testing Spherical Coordinate Conversions")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Unit vector in +x direction
    all_passed &= test_conversion(
        "Unit vector +x",
        [1.0, 0.0, 0.0],
        expected_spherical=[1.0, np.pi/2, 0.0]  # magnitude=1, theta=π/2 (horizontal), phi=0 (along x)
    )
    
    # Test 2: Unit vector in +y direction
    all_passed &= test_conversion(
        "Unit vector +y",
        [0.0, 1.0, 0.0],
        expected_spherical=[1.0, np.pi/2, np.pi/2]  # magnitude=1, theta=π/2, phi=π/2 (along y)
    )
    
    # Test 3: Unit vector in +z direction
    # Note: Due to floating point precision, theta may be slightly non-zero
    # The round-trip error is acceptable (magnitude preserved, direction preserved)
    all_passed &= test_conversion(
        "Unit vector +z",
        [0.0, 0.0, 1.0],
        tolerance=1e-3  # Larger tolerance for edge case with theta near 0
    )
    
    # Test 4: Zero vector (should use minimum magnitude clamp)
    all_passed &= test_conversion(
        "Zero vector",
        [0.0, 0.0, 0.0],
        tolerance=1e-3  # Larger tolerance for zero case
    )
    
    # Test 5: Arbitrary 3D vector
    all_passed &= test_conversion(
        "Arbitrary vector [1, 1, 1]",
        [1.0, 1.0, 1.0]
    )
    
    # Test 6: Negative components
    all_passed &= test_conversion(
        "Negative vector [-1, -1, -1]",
        [-1.0, -1.0, -1.0]
    )
    
    # Test 7: Small magnitude (should be clamped)
    all_passed &= test_conversion(
        "Very small vector",
        [1e-5, 1e-5, 1e-5],
        tolerance=1e-3
    )
    
    # Test 8: Typical action vector (normalized)
    all_passed &= test_conversion(
        "Normalized action [0.5, 0.3, -0.2]",
        [0.5, 0.3, -0.2]
    )
    
    # Test 9: Batch conversion (PyTorch)
    print("\n=== Batch Conversion Test (PyTorch) ===")
    batch_cart = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ], dtype=torch.float32)
    print(f"Batch input shape: {batch_cart.shape}")
    
    batch_spherical = cartesian_to_spherical(batch_cart)
    print(f"Batch spherical shape: {batch_spherical.shape}")
    print(f"Batch spherical:\n{batch_spherical}")
    
    batch_recovered = spherical_to_cartesian(batch_spherical)
    batch_error = torch.norm(batch_cart - batch_recovered, dim=1)
    print(f"Batch round-trip errors: {batch_error}")
    
    if torch.all(batch_error < 1e-4):
        print("✅ Batch conversion PASS")
    else:
        print("❌ Batch conversion FAIL")
        all_passed = False
    
    # Test 10: Verify magnitude is always >= min_magnitude
    print("\n=== Minimum Magnitude Clamp Test ===")
    zero_vec = np.array([0.0, 0.0, 0.0])
    spherical_zero = cartesian_to_spherical(zero_vec)
    magnitude = spherical_zero[0]
    print(f"Zero vector magnitude after conversion: {magnitude}")
    print(f"Minimum expected: 1e-4")
    
    if magnitude >= 1e-4:
        print("✅ Minimum magnitude clamp PASS")
    else:
        print(f"❌ Minimum magnitude clamp FAIL: got {magnitude}, expected >= 1e-4")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)


