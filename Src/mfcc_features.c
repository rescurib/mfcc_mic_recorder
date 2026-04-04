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