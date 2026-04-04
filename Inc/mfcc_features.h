#ifndef MFCC_FEATURES_H
#define MFCC_FEATURES_H

#include <stdint.h>
#include <arm_math.h>

// MFCC Parameters
#define MFCC_COEFFS_NUM 13U
#define FFT_SIZE 256U

/**
 * @brief Initialize the f32 MFCC feature extractor.
 * @param mfcc_ctx Pointer to the MFCC f32 instance.
 */
void mfcc_features_init_f32(arm_mfcc_instance_f32 *mfcc_ctx);

/**
 * @brief Initialize the q15 MFCC feature extractor.
 * @param mfcc_ctx Pointer to the MFCC q15 instance.
 */
void mfcc_features_init_q15(arm_mfcc_instance_q15 *mfcc_ctx);

/**
 * @brief Calculate the mean over all hops for a Q15 MFCC matrix.
 * @param mfcc_mat Pointer to the matrix of MFCC coefficients.
 * @param num_hops Number of hops (frames) in the matrix.
 * @param num_coeffs Number of MFCC coefficients per hop.
 * @param feature_vector Output pointer for the mean feature vector.
 */
void mean_mfcc_q15(const q15_t *mfcc_mat, uint32_t num_hops, uint32_t num_coeffs, q15_t *feature_vector);

/**
 * @brief Calculate the mean over all hops for an F32 MFCC matrix.
 * @param mfcc_mat Pointer to the matrix of MFCC coefficients.
 * @param num_hops Number of hops (frames) in the matrix.
 * @param num_coeffs Number of MFCC coefficients per hop.
 * @param feature_vector Output pointer for the mean feature vector.
 */
void mean_mfcc_f32(const float32_t *mfcc_mat, uint32_t num_hops, uint32_t num_coeffs, float32_t *feature_vector);

#endif /* MFCC_FEATURES_H */