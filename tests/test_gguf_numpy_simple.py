#!/usr/bin/env python3
"""
Simple demonstration of saving numpy arrays to GGUF and reading them back.
Shows the data before saving and after reading to verify correctness.
"""

import os
import tempfile
import numpy as np
import gguf


def save_and_read_numpy_arrays():
    """Save numpy arrays to GGUF and read them back, printing before/after."""
    
    # Create test arrays
    print("=== Creating test numpy arrays ===")
    
    # 1D array
    arr_1d = np.array([1.0, 2.5, 3.7, 4.2, 5.9], dtype=np.float32)
    print(f"\n1D array (shape {arr_1d.shape}):")
    print(arr_1d)
    
    # 2D array
    arr_2d = np.array([[1, 2, 3], 
                       [4, 5, 6], 
                       [7, 8, 9]], dtype=np.int32)
    print(f"\n2D array (shape {arr_2d.shape}):")
    print(arr_2d)
    
    # 3D array with random data
    arr_3d = np.random.randn(2, 3, 4).astype(np.float16)
    print(f"\n3D array (shape {arr_3d.shape}):")
    print(arr_3d)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as tmp:
        temp_file = tmp.name
    
    try:
        # Write arrays to GGUF
        print("\n=== Writing arrays to GGUF file ===")
        writer = gguf.GGUFWriter(temp_file, "numpy_test")
        
        # Add arrays as tensors
        arrays = {
            "array_1d": arr_1d,
            "array_2d": arr_2d,
            "array_3d": arr_3d
        }
        
        # Add tensor info
        for name, array in arrays.items():
            writer.add_tensor_info(name, array.shape, array.dtype, array.nbytes)
        
        # Write file structure
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        
        # Write tensor data
        for name, array in arrays.items():
            writer.write_tensor_data(array)
        
        writer.close()
        print(f"Written to: {temp_file}")
        
        # Read arrays back from GGUF
        print("\n=== Reading arrays from GGUF file ===")
        reader = gguf.GGUFReader(temp_file)
        
        # Read each tensor
        for tensor in reader.tensors:
            print(f"\nReading tensor '{tensor.name}':")
            
            # Map GGML types to numpy dtypes
            type_map = {
                gguf.GGMLQuantizationType.F32: np.float32,
                gguf.GGMLQuantizationType.F16: np.float16,
                gguf.GGMLQuantizationType.I32: np.int32,
            }
            
            dtype = type_map.get(tensor.tensor_type)
            
            # GGUF stores shapes in reversed order
            shape = list(reversed(tensor.shape))
            
            # Read and reshape data
            array = np.frombuffer(tensor.data, dtype=dtype).reshape(shape)
            
            print(f"Shape: {array.shape}, dtype: {array.dtype}")
            print(array)
            
            # Verify against original
            original = arrays[tensor.name]
            if np.array_equal(original, array):
                print("✓ Data matches original!")
            else:
                print("✗ Data does NOT match original!")
        
        # Close reader
        if hasattr(reader, 'file') and reader.file:
            reader.file.close()
            
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"\nCleaned up temporary file: {temp_file}")


if __name__ == "__main__":
    save_and_read_numpy_arrays()