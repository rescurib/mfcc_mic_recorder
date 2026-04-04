#if 0
/**
 * @file serial_mic_recorder.c
 * @brief I2S audio acquisition from an INMP441 MEMS microphone and MFCC feature extraction.
 *
 * This example shows how to use the STM32 I2S peripheral in DMA mode to acquire audio data
 * from an INMP441 digital MEMS microphone. The acquired samples are processed using CMSIS-DSP
 * to extract Mel-Frequency Cepstral Coefficients (MFCCs). When a signal above the noise floor
 * is detected, the extracted MFCC features are sent over UART for further classification.
 *
 * Hardware:
 *   - STM32 series MCU
 *   - INMP441 MEMS microphone (I2S interface)
 *   - UART for serial output
 *
 * Usage:
 *   - Speak into the microphone. Voice activity detection will trigger MFCC extraction.
 *   - MFCC features are streamed over UART in a simple binary format.
 */
#endif

#include "serial_mic_recorder.h"
#include "main.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// CMSIS-DSP includes
#include "mfcc_config.h"
#include <arm_math.h>

// User includes
#include "mfcc_features.h"

/******** Definitions ******** */
#define SAMPLES_PER_HOP 256U // 32 ms at 8 kHz
#define HOPS_PER_FRAME 8U    // 256 ms total frame length at 8 kHz
#define FULL_BUFFER_SIZE                                                       \
  (SAMPLES_PER_HOP * HOPS_PER_FRAME) // 256 ms of audio at 8 kHz

// MFCC Parameters defined in mfcc_features.h

typedef union {
  float32_t f32;
  uint8_t b[4];
} float_packet_t;

// Indicates if the microphone is currently recording
static volatile bool is_recording = false;

// Buffer for I2S stereo samples (2 words for left and right channels)
static uint8_t i2s_stereo_samples[SAMPLES_PER_HOP * 2 *
                                  4]; // Two channels, 4 bytes per sample.
static uint8_t sample_buff[SAMPLES_PER_HOP * 2 *
                           4]; // Buffer for one hop of mono samples (32 ms)
static float32_t full_buff[FULL_BUFFER_SIZE]; // 256 ms of mono samples at 8 kHz
static q15_t hop[SAMPLES_PER_HOP];

// MFCC matrix for a frame [8 hops]
q15_t mfcc_matrix[HOPS_PER_FRAME][MFCC_COEFFS_NUM];

// Average of the MFCC coefficients for the current frame.
#define FEATURE_VECTOR_SIZE                                                    \
  (MFCC_COEFFS_NUM) // Only mean for each coefficient
static q15_t feature_vector[FEATURE_VECTOR_SIZE];

// Global variables
/**< @brief Coefficient for noise floor update (exponential
 * smoothing) */
#define NOISE_ALPHA (q15_t)(0.01f * 32768)
q15_t g_noise_floor =
    (q15_t)(0.01f * 32768); // Initial background noise floor value (RMS)
volatile bool g_signal_detected =
    false; // Indicates if a signal above the noise threshold has been detected
volatile bool g_dma_data_ready = false; // Indicates if the DMA transfer has completed

// External handles for I2S and UART peripherals (defined elsewhere)
extern I2S_HandleTypeDef hi2s2;
extern UART_HandleTypeDef huart2;

// Internal function prototypes
static void mic_start(void);
static void mic_stop(void);
static inline q15_t i2s_sample_to_q15(uint8_t *sample);

/**
 * @brief  Update the status LED to indicate recording state.
 * @param  recording: true if recording, false otherwise
 */
static inline void update_status_led(bool recording) {
  if (recording) {
    // Turn ON status LED (e.g., LD2)
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
  } else {
    // Turn OFF status LED
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
  }
}

/**
 * @brief  Main loop for the serial microphone recorder.
 *         Handles initialization and waits for user input (button interrupt).
 */
void serial_recorder_loop(void) {
  // Initialize state
  is_recording = false;
  update_status_led(is_recording);
  memset(i2s_stereo_samples, 0, sizeof(i2s_stereo_samples));
  uint16_t hop_index = 0;

  // Temporary buffer for complex data
  q31_t mfcc_complex_buff[2 * FFT_SIZE];

  // Initialize MFCC context
  arm_mfcc_instance_q15 mfcc_ctx;
  mfcc_features_init_q15(&mfcc_ctx);
  while (1) {
    if (g_signal_detected) {
      if (g_dma_data_ready &&
          hop_index < 8) // Only process if DMA data is ready and we haven't
                         // filled the entire buffer
      {
        memcpy(full_buff + (hop_index * SAMPLES_PER_HOP), hop, sizeof(hop));
        g_dma_data_ready = false; // Reset flag after processing

        // Calculate MFCCs of the current hop
        // arm_mfcc_f32(&mfcc_ctx, hop, mfcc_matrix[hop_index],
        // mfcc_complex_buff);
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_SET);
        arm_mfcc_q15(&mfcc_ctx, hop, mfcc_matrix[hop_index], mfcc_complex_buff);
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_RESET);
        hop_index++;

      } else if (hop_index == 8) {
        mean_mfcc_q15(&mfcc_matrix[0][0], HOPS_PER_FRAME, MFCC_COEFFS_NUM,
                  feature_vector);

        hop_index = 0;
        g_signal_detected = false;

        // Send feature vector via UART
        HAL_UART_Transmit(&huart2, (uint8_t *)"MFCC_START\r\n", 10, 100U);
        HAL_Delay(1);

        // Send each element as 2 bytes (int16_t)
        uint8_t send_buff[3];
        for (int i = 0; i < FEATURE_VECTOR_SIZE; i++) {
          memcpy(send_buff, &feature_vector[i], 2);
          send_buff[2] = '\n';
          HAL_UART_Transmit(&huart2, send_buff, 3, 100U);
        }

        HAL_UART_Transmit(&huart2, (uint8_t *)"MFCC_END\r\n", 8, 100U);

        // Delay to avoid recording echoes.
        HAL_Delay(200);
        g_signal_detected = false;
      }
    }
  }
}

/**
 * @brief  Start microphone acquisition using I2S DMA.
 *         Updates state and notifies via UART.
 */
static void mic_start(void) {
  if (!is_recording) {
    /* Start I2S DMA reception (2 words, 24 bits each)
     I know, this size argument is confusing since uint16_t* is used,
     but the HAL API expects it this way when 24 or 32 bit data formats are
     used.
    */
    if (HAL_I2S_Receive_DMA(&hi2s2, (uint16_t *)i2s_stereo_samples,
                            2 * SAMPLES_PER_HOP) == HAL_OK) {
      is_recording = true;
      update_status_led(is_recording);

      // Notify host that acquisition has started
      const char *msg = "Mic acquisition: START\r\n";
      HAL_UART_Transmit(&huart2, (uint8_t *)msg, strlen(msg), 100U);
    } else {
      // Notify host of error
      const char *msg = "Mic acquisition: START ERROR\r\n";
      HAL_UART_Transmit(&huart2, (uint8_t *)msg, strlen(msg), 100U);
    }
  }
}

/**
 * @brief  Stop microphone acquisition and DMA.
 *         Updates state and notifies via UART.
 */
static void mic_stop(void) {
  if (is_recording) {
    // Stop I2S DMA
    HAL_I2S_DMAStop(&hi2s2);
    is_recording = false;
    update_status_led(is_recording);

    // Notify host that acquisition has stopped
    const char *msg = "Mic acquisition: STOP\r\n";
    HAL_UART_Transmit(&huart2, (uint8_t *)msg, strlen(msg), 100U);
  }
}

/**
 * @brief  EXTI line detection callback (button press).
 *         Toggles recording state on user button (B1) press.
 * @param  GPIO_Pin: Specifies the pins connected EXTI line
 */
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
  if (GPIO_Pin == B1_Pin) {
    // Toggle recording state on button press
    if (is_recording) {
      mic_stop();
    } else {
      mic_start();
    }
  }
}

/*
HAL_I2S_RxHalfCpltCallback(I2S_HandleTypeDef *hi2s)
{
    // This callback is called when half of the DMA buffer is filled.
    // You can use this to process the first half of the samples while the
second half is being filled.
}
*/

/**
 * @brief  I2S DMA receive complete callback.
 *         Called when a block of I2S data is received.
 *         Sends the received samples over UART.
 * @param  hi2s: I2S handle
 */
void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s) {
  // Only process if this is our I2S instance
  if (hi2s->Instance != hi2s2.Instance) {
    return; // Not our I2S instance
  }

  // Only process if currently recording
  if (is_recording == false) {
    return; // Not recording, ignore
  }

  memcpy(sample_buff, i2s_stereo_samples, sizeof(sample_buff));

  g_dma_data_ready =
      true; // Indicates that the DMA data is ready to be processed

  q15_t result;

  for (uint32_t i = 0; i < SAMPLES_PER_HOP; i++)
  {
    // Convert left channel sample (first 4 bytes of each stereo pair)
    uint8_t *sample_ptr = &sample_buff[i * 8]; // 8 bytes per stereo sample (4 for left, 4 for right)
    hop[i] = i2s_sample_to_q15(sample_ptr);
  }

  arm_rms_q15(hop, SAMPLES_PER_HOP, &result);

  if (result > 7.5 * g_noise_floor)
  {
    g_signal_detected = true;
  } else {
    // Update noise floor with exponential smoothing in Q15:
    // equivalent to g_noise_floor = (1.0f - NOISE_ALPHA) * g_noise_floor +
    // NOISE_ALPHA * result. Mathematically optimized to: g_noise_floor =
    // g_noise_floor + NOISE_ALPHA * (result - g_noise_floor)
    g_noise_floor = (q15_t)(g_noise_floor + ((((q31_t)result - g_noise_floor) * NOISE_ALPHA) >> 15));
  }
}

static inline float32_t i2s_sample_to_float32(uint8_t *sample) {
  int32_t reord_sample =
      (int32_t)((sample[1] << 16) | (sample[0] << 8) | (sample[3]));

  if (reord_sample & 0x00800000) // if 24-bit sign bit is set
  {
    reord_sample |= 0xFF000000; // sign-extend to 32 bits
  }

  // Convert 24-bit signed sample to float in range [-1.0, 1.0]
  return (float32_t)(reord_sample / 8388608.0f); // Divide by 2^23 to normalize
}

static inline q15_t i2s_sample_to_q15(uint8_t *sample) {
  return (q15_t)((sample[1] << 8) | sample[0]);
}

