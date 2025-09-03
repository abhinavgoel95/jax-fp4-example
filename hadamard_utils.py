"""
Hadamard transform utilities for JAX.
Includes sign vector loading, random Hadamard matrix generation, and transform operations.
"""

import jax
import jax.numpy as jnp
import scipy
import json
from pathlib import Path
from typing import Optional


def load_random_sign_vec_from_json(
    json_file: str,
    gemm_type: str,
    hadamard_dim: int,
) -> jnp.ndarray:
    """
    JAX equivalent of the PyTorch load_random_sign_vec_from_json function.
    
    Load random sign vector for Hadamard transform from JSON file.
    
    Args:
        json_file: Path to JSON file (relative to current directory)
        gemm_type: Type of GEMM operation ("fprop", "dgrad", "wgrad")
        hadamard_dim: Dimension of Hadamard matrix
        
    Returns:
        JAX array containing the sign vector
        
    Raises:
        RuntimeError: If loading fails or required keys are not found
    """
    json_path = Path(json_file)
    
    try:
        with open(json_path, "r") as f:
            stored_hadamard_dims = json.load(f)
        
        if gemm_type not in stored_hadamard_dims:
            raise RuntimeError(
                f"Error loading random sign vector for Hadamard transform: "
                f"gemm_type {gemm_type} not found in {json_path}"
            )
        
        if str(hadamard_dim) not in stored_hadamard_dims[gemm_type]:
            raise RuntimeError(
                f"Error loading random sign vector for Hadamard transform: "
                f"hadamard_dim {hadamard_dim} for gemm_type {gemm_type} not found in {json_path}. "
                f"Available dimensions: {list(stored_hadamard_dims[gemm_type].keys())}"
            )
        
        hadamard_sign_vec = stored_hadamard_dims[gemm_type][str(hadamard_dim)]
        
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading sign vector: {str(e)}") from e
    
    return jnp.array(hadamard_sign_vec, dtype=jnp.bfloat16)


def random_hadamard(key, n, gemm_type="fprop", json_file="hadamard_random_sign_vec.json"):
    """
    Generate a random Hadamard matrix of size n x n.
    n must be a power of 2.
    
    Args:
        key: JAX random key (unused if loading from JSON)
        n: Size of Hadamard matrix
        gemm_type: Type of GEMM operation ("fprop", "dgrad", "wgrad")
        json_file: Path to JSON file containing sign vectors
    
    Returns:
        Random Hadamard matrix of size n x n
    """
    try:
        # Try to load from JSON file first
        s = load_random_sign_vec_from_json(json_file, gemm_type, n)
    except (FileNotFoundError, RuntimeError, KeyError):
        # Fall back to random generation if JSON loading fails
        print(f"Warning: Could not load sign vector from JSON for {gemm_type}, dim {n}. Using random generation.")
        s = jax.random.randint(key, (n,), 0, 2) * 2 - 1
        s = s.astype(jnp.bfloat16)
    
    # Generate the Hadamard matrix
    out = jnp.diag(s) @ jnp.array(scipy.linalg.hadamard(n)) / jnp.sqrt(n)
    return out.astype(jnp.bfloat16)


class HadamardTransform:
    """
    A class to encapsulate Hadamard transform operations.
    Allows for multiple Hadamard matrices with different configurations.
    """
    
    def __init__(self, 
                 hadamard_dim: int = 16, 
                 gemm_type: str = "fprop",
                 json_file: str = "hadamard_random_sign_vec.json",
                 key: Optional[jax.Array] = None):
        """
        Initialize Hadamard transform.
        
        Args:
            hadamard_dim: Dimension of Hadamard matrix (must be power of 2)
            gemm_type: Type of GEMM operation ("fprop", "dgrad", "wgrad")
            json_file: Path to JSON file containing sign vectors
            key: JAX random key (used if JSON loading fails)
        """
        self.hadamard_dim = hadamard_dim
        self.json_file = json_file
        self.key = key
        
    def transform(self, A: jnp.ndarray, rht_for_transpose: bool = False, gemm_types: list[str] = ["fprop", "wgrad"]) -> jnp.ndarray:
        """
        Apply Hadamard transform to input array.
        
        Args:
            A: Input array, last dimension must be compatible with hadamard_dim
            rht_for_transpose: If True, applies transpose transform (A.T @ H).T equivalent to H.T @ A
            
        Returns:
            Transformed array with same shape as input
        """
        original_shape = A.shape

        if self.key is None:
            self.key = jax.random.PRNGKey(42)
        if rht_for_transpose:
            # Returns (A.T @ H).T, equivalent to H.T @ A
            # preserves tiling dimension
            hadamard_matrix = random_hadamard(self.key, self.hadamard_dim, gemm_types[1], self.json_file)
            A_transformed = (A.T.reshape(-1, self.hadamard_dim) @ hadamard_matrix)
            rht_t =   A_transformed.reshape(original_shape[::-1]).T
        else:
            rht_t = A
        hadamard_matrix = random_hadamard(self.key, self.hadamard_dim, gemm_types[0], self.json_file)
        A_transformed = (A.reshape(-1, self.hadamard_dim) @ hadamard_matrix)
        rht =  A_transformed.reshape(original_shape)
        return rht, rht_t

    def __call__(self, A: jnp.ndarray, gemm_types: list[str] = ["fprop", "wgrad"], rht_for_transpose: bool = False) -> jnp.ndarray:
        return self.transform(A, rht_for_transpose=rht_for_transpose, gemm_types=gemm_types)

    def get_rht(self) -> jnp.ndarray:
        """Get the regular Hadamard transform matrix (rht)."""
        return self.rht

    def get_rht_t(self) -> jnp.ndarray:
        """Get the transpose Hadamard transform matrix (rht_t)."""
        return self.rht_t


def create_hadamard_transforms(hadamard_dim: int = 16, 
                              key: Optional[jax.Array] = None,
                              json_file: str = "hadamard_random_sign_vec.json"):
    """
    Create Hadamard transforms for different GEMM types.
    
    Args:
        hadamard_dim: Dimension of Hadamard matrix
        key: JAX random key
        json_file: Path to JSON file containing sign vectors
        
    Returns:
        Dictionary with transforms for different GEMM types
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    transforms = {}
    for gemm_type in ["fprop", "dgrad", "wgrad"]:
        key, subkey = jax.random.split(key)
        transforms[gemm_type] = HadamardTransform(
            hadamard_dim=hadamard_dim,
            gemm_type=gemm_type,
            json_file=json_file,
            key=subkey
        )
    
    return transforms


def simple_hadamard_transform(A: jnp.ndarray, 
                             hadamard_matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Simple function to apply a pre-computed Hadamard transform.
    
    Args:
        A: Input array
        hadamard_matrix: Pre-computed Hadamard matrix
        
    Returns:
        Transformed array
    """
    hadamard_dim = hadamard_matrix.shape[0]
    return (A.reshape(-1, hadamard_dim) @ hadamard_matrix).reshape(A.shape)


def validate_hadamard_dimension(n: int) -> bool:
    """
    Validate that n is a valid Hadamard dimension (power of 2).
    
    Args:
        n: Dimension to validate
        
    Returns:
        True if valid, False otherwise
    """
    return n > 0 and (n & (n - 1)) == 0


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Hadamard utilities...")
    
    # Test dimension validation
    print(f"Is 16 a valid Hadamard dimension? {validate_hadamard_dimension(16)}")
    print(f"Is 15 a valid Hadamard dimension? {validate_hadamard_dimension(15)}")
    
    # Create a simple Hadamard transform
    key = jax.random.PRNGKey(42)
    hadamard_dim = 16
    
    print(f"\nCreating Hadamard transform for dimension {hadamard_dim}...")
    transform = HadamardTransform(hadamard_dim=hadamard_dim, key=key)
    
    # Test the transform
    test_input = jax.random.normal(key, (8, 16)).astype(jnp.bfloat16)
    transformed = transform(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Transformed shape: {transformed.shape}")
    print(f"Input mean: {jnp.mean(test_input):.4f}")
    print(f"Transformed mean: {jnp.mean(transformed):.4f}")
    
    # Create transforms for all GEMM types
    print(f"\nCreating transforms for all GEMM types...")
    all_transforms = create_hadamard_transforms(hadamard_dim=hadamard_dim, key=key)
    
    for gemm_type, transform in all_transforms.items():
        result = transform(test_input)
        print(f"{gemm_type}: mean = {jnp.mean(result):.4f}, std = {jnp.std(result):.4f}")
    
    print("Hadamard utilities test completed!")
