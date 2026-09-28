#include "mfcc_features.h"
#include "mfcc_config.h"

void mfcc_features_init_f32(arm_mfcc_instance_f32 *mfcc_ctx) {
  arm_mfcc_init_f32(
      mfcc_ctx, FFT_SIZE, NB_MFCC_NB_FILTER_CONFIG_8K_F32, MFCC_COEFFS_NUM,
      mfcc_dct_coefs_config1_f32, mfcc_filter_pos_config_8k_f32,
      mfcc_filter_len_config_8k_f32, mfcc_filter_coefs_config_8k_f32,
      mfcc_window_coefs_config3_f32);
}

void mfcc_features_init_q15(arm_mfcc_instance_q15 *mfcc_ctx) {
  arm_mfcc_init_q15(
      mfcc_ctx, FFT_SIZE, NB_MFCC_NB_FILTER_CONFIG_8K_Q15, MFCC_COEFFS_NUM,
      mfcc_dct_coefs_config1_q15, mfcc_filter_pos_config_8k_q15,
      mfcc_filter_len_config_8k_q15, mfcc_filter_coefs_config_8k_q15,
      mfcc_window_coefs_config3_q15);
}

void mean_mfcc_q15(const q15_t *mfcc_mat, uint32_t num_hops, uint32_t num_coeffs,
               q15_t *feature_vector) {
  q15_t col_buf[num_hops]; // The correct size is num_hops

  for (uint16_t i = 0; i < num_coeffs; i++) {
    for (uint16_t j = 0; j < num_hops; j++) {
      // Extract the i-th column of the MFCC matrix
      col_buf[j] = *(mfcc_mat + (j * num_coeffs) + i);
    }
    arm_mean_q15(col_buf, num_hops, &feature_vector[i]);
  }
}

void mean_mfcc_f32(const float32_t *mfcc_mat, uint32_t num_hops, uint32_t num_coeffs,
               float32_t *feature_vector) {
  float32_t col_buf[num_hops]; // The correct size is num_hops

  for (uint16_t i = 0; i < num_coeffs; i++) {
    for (uint16_t j = 0; j < num_hops; j++) {
      // Extract the i-th column of the MFCC matrix
      col_buf[j] = *(mfcc_mat + (j * num_coeffs) + i);
    }
    arm_mean_f32(col_buf, num_hops, &feature_vector[i]);
  }
}


// Compute delta MFCCs for a q15_t MFCC matrix (HOPS_PER_FRAME x MFCC_COEFFS_NUM)
// Input:  mfcc_mat [HOPS_PER_FRAME][MFCC_COEFFS_NUM] (row-major)
// Output: delta_mat [HOPS_PER_FRAME][MFCC_COEFFS_NUM] (row-major)
// win_length: N (typically 2)
// Uses edge padding (replicates border values)
// High-performance, fixed-point, and SIMD-friendly delta MFCC computation for q15_t
void compute_delta_mfcc_q15(const q15_t *mfcc_mat, q15_t *delta_mat, uint32_t num_hops, uint32_t num_coeffs, uint16_t win_length)
{
  if (win_length == 0 || num_hops == 0) {
    return;
  }

  int32_t den = 0;
  for (uint16_t n = 1; n <= win_length; n++) {
    den += n * n;
  }
  den *= 2;

  for (uint32_t t = 0; t < num_hops; t++) {
    for (uint32_t c = 0; c < num_coeffs; c++) {
      int32_t num = 0;

      for (uint16_t n = 1; n <= win_length; n++) {
        int32_t t_plus = (int32_t)t + (int32_t)n;
        if (t_plus >= (int32_t)num_hops) {
          t_plus = (int32_t)num_hops - 1;
        }

        int32_t t_minus = (int32_t)t - (int32_t)n;
        if (t_minus < 0) {
          t_minus = 0;
        }

        int32_t val_plus  = mfcc_mat[t_plus * num_coeffs + c];
        int32_t val_minus = mfcc_mat[t_minus * num_coeffs + c];

        num += (int32_t)n * (val_plus - val_minus);
      }

      delta_mat[t * num_coeffs + c] = (q15_t)(num / den);
    }
  }
}

// Interleave MFCC and Delta MFCC matrices into a single feature vector
// Structure: [MFCCHop0][DeltaHop0]...[MFCCHop(N-1)][DeltaHop(N-1)]
void interleave_mfcc_delta_q15(const q15_t *mfcc_mat, const q15_t *delta_mat, uint32_t num_hops, uint32_t num_coeffs, q15_t *feature_vector)
{
  for (uint32_t t = 0; t < num_hops; t++) {
    arm_copy_q15(&mfcc_mat[t * num_coeffs], &feature_vector[t * 2 * num_coeffs], num_coeffs);
    arm_copy_q15(&delta_mat[t * num_coeffs], &feature_vector[t * 2 * num_coeffs + num_coeffs], num_coeffs);
  }
}