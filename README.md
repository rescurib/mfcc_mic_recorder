# MFCC Microfone Recorder

I originally had this project as branch of a wake word project, but I decided to make it a standalone project.

This program records audio samples from a I2S mic and compute MFCC features with the CMSIS-DSP lib. I use this program to collect data to train a ML model to recognize a wake word. It can be configured for float32_t or q15_t MFCC features.

## Hardware

- STM32F303K8T6
- INMP441 MEMS microphone
- UART for serial output

## Software

- STM32CubeIDE
- CMSIS-DSP lib

## Usage

1. Compile and flash the firmware to the STM32F303K8T6.
2. Connect the I2S microphone to the STM32F303K8T6.
3. Connect the UART to the computer.
4. Run python serial_to_features_q15.py
5. Press the button to start recording.
6. Speak into the microphone and the MFCC features will be printed to the serial terminal.
7. Press the button again to stop recording and finish data collecting script.
