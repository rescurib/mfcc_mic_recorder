#!/usr/bin/env python3
import argparse
import numpy as np

def compute_ideal_delta(mfcc_float_row, num_hops, num_coeffs, win_length=2):
    """
    Computes ideal delta MFCCs using pure floating point precision.
    This serves as a reference to measure the error introduced by the MCU's Q15 fixed-point arithmetic.
    mfcc_float_row is a 1D array of length (num_hops * num_coeffs)
    """
    # Reshape for easy indexing (num_hops, num_coeffs)
    mfcc_mat = mfcc_float_row.reshape((num_hops, num_coeffs))
    delta_mat = np.zeros_like(mfcc_mat, dtype=np.float32)
    
    den = float(sum([n * n for n in range(1, win_length + 1)]) * 2)
    
    for t in range(num_hops):
        for c in range(num_coeffs):
            num = 0.0
            for n in range(1, win_length + 1):
                t_plus = min(t + n, num_hops - 1)
                t_minus = max(t - n, 0)
                
                val_plus = mfcc_mat[t_plus, c]
                val_minus = mfcc_mat[t_minus, c]
                
                num += n * (val_plus - val_minus)
                
            delta_mat[t, c] = num / den
            
    return delta_mat.flatten()

def main():
    parser = argparse.ArgumentParser(description="Extract MFCC only and measure MCU Q15 delta precision error")
    parser.add_argument('--input', type=str, default='features.npy', help='Input feature file containing interleaved float32 MFCC+Delta')
    parser.add_argument('--output', type=str, default='mfcc_only.npy', help='Output file for MFCC only matrix')
    parser.add_argument('--num-coeffs', type=int, default=13, help='Number of MFCC coefficients')
    parser.add_argument('--win-length', type=int, default=2, help='Delta computation window length (N)')
    args = parser.parse_args()

    # Load interleaved data
    try:
        data = np.load(args.input)
    except FileNotFoundError:
        print(f"Error: Could not find input file '{args.input}'.")
        return

    num_samples = data.shape[0]
    total_features = data.shape[1]
    
    # Check total features count matches expected dimensions
    if total_features % (2 * args.num_coeffs) != 0:
        print(f"Error: Total features ({total_features}) is not divisible by 2 * num_coeffs ({2 * args.num_coeffs}).")
        return
        
    num_hops = total_features // (2 * args.num_coeffs)
    print(f"Loaded {num_samples} samples. Detected {num_hops} hops per sample with {args.num_coeffs} MFCC coeffs.")
    
    # Reshape format, inside MCU was:
    # interleaved as [MFCC_Hop0...][Delta_Hop0...][MFCC_Hop1...][Delta_Hop1...]
    data_reshaped = data.reshape((num_samples, num_hops, 2, args.num_coeffs))
    
    # Extract MFCC and MCU Deltas
    # Index 0 is MFCC, Index 1 is Delta
    mfcc_float = data_reshaped[:, :, 0, :].reshape((num_samples, num_hops * args.num_coeffs))
    mcu_delta_float = data_reshaped[:, :, 1, :].reshape((num_samples, num_hops * args.num_coeffs))

    # Save MFCC only matrix
    np.save(args.output, mfcc_float)
    print(f"Saved extracted MFCC data (shape: {mfcc_float.shape}) to '{args.output}'")

    # Verify deltas: Compare MCU Q15 against ideal float calculations
    print("\nComparing MCU computed deltas against ideal pure float32 calculations...")
    
    max_error = 0.0
    total_mae = 0.0
    total_elements = num_samples * num_hops * args.num_coeffs
    
    for i in range(num_samples):
        # Compute Py Delta using pure floating point arithmetic
        ideal_delta_float = compute_ideal_delta(mfcc_float[i], num_hops, args.num_coeffs, args.win_length)
        
        # Check against MCU floats
        abs_errors = np.abs(ideal_delta_float - mcu_delta_float[i])
        
        total_mae += np.sum(abs_errors)
        
        error = np.max(abs_errors)
        if error > max_error:
            max_error = error

    mean_absolute_error = total_mae / total_elements
    print(f"Maximum absolute error per delta coefficient: {max_error:.6f}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error:.8f}")
    
    print("\nNote: Small errors are expected due to the q15 integer truncation on the MCU.")

if __name__ == '__main__':
    main()
