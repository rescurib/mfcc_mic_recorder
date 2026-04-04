#!/usr/bin/env python3
"""
UART to feature vector for ML training (Q15 version)

This script listens to a serial port for MFCC data streamed from an STM32 microcontroller 
and saves it as a numpy array. It expects data in Q15 format (2 bytes per feature).

Usage:
    python serial_to_features_q15.py [--port COM_PORT] [--output OUTPUT_FILE]

Arguments:
    --port      Serial COM port to listen to (default: COM12)
    --output    Output numpy file name (default: features_q15.npy)
    --keep-int  Keep the values as raw 16-bit integers instead of converting to float

Example:
    python serial_to_features_q15.py --port COM3 --output mfcc_features_data.npy

"""
import argparse
import serial
import struct
import numpy as np

DEFAULT_COM_PORT    = 'COM12'
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_NPY_FILE    = 'features_q15.npy'

MFCC_FEATURES_PER_FRAME = 13
training_data = []  # List to hold all feature vectors

def parse_args():
    parser = argparse.ArgumentParser(description='UART to feature vector for ML training (Q15)')
    parser.add_argument('--port', type=str, default=DEFAULT_COM_PORT, help='Serial COM port (default: COM12)')
    parser.add_argument('--output', type=str, default=DEFAULT_NPY_FILE, help='Output numpy file name (default: features_q15.npy)')
    parser.add_argument('--keep-int', action='store_true', help='Save features as integer Q15 values instead of converting to float')
    return parser.parse_args()


def main():
    args = parse_args()
    ser = serial.Serial(args.port, baudrate=460800, timeout=1)
    print('Listening on {}...'.format(args.port))
    samples = []
    streaming = False
    buffer = b''
    sample_count = 0

    while True:
        try:
            byte = ser.read(1)
            if not byte:
                continue
            buffer += byte
            # Only check for text commands if not streaming
            if not streaming and b'\n' in buffer:
                try:
                    text = buffer.decode(errors='ignore').strip()
                except Exception:
                    buffer = b''
                    continue
                if text == 'Mic acquisition: START':
                    print('Mic acquisition: START')
                    streaming = True
                    buffer = b''
                    continue
                buffer = b''
                continue

            if streaming:
                # Check for stop command
                if b"Mic acquisition: STOP" in buffer:
                    print('Mic acquisition: STOP')
                    break

                # Check for MFCC feature vector start
                if b'MFCC_START' in buffer:
                    print('MFCC_START detected')
                    buffer = b''
                    continue

                # Check for MFCC feature vector stop
                if b'MFCC_END' in buffer:
                    if samples and len(samples) % MFCC_FEATURES_PER_FRAME == 0:
                        training_data.append(samples)
                        samples = []
                        sample_count += 1
                        print("MFCC sample saved: {}".format(sample_count))
                    else:
                        print("Wrong number of features received: {}".format(len(samples)))
                        samples = []
                    buffer = b''
                    continue

                # Process 2 bytes + newline (int16 + '\n') for Q15 implementation
                if len(buffer) == 3 and buffer[2] == ord('\n'):
                    int16_bytes = buffer[:2]
                    try:
                        # Unpack little-endian int16 (q15 format)
                        q15_val = struct.unpack('<h', int16_bytes)[0]
                        if args.keep_int:
                            samples.append(q15_val)
                        else:
                            # Convert to float representation
                            f = float(q15_val) / 32768.0
                            samples.append(f)
                    except Exception as e:
                        print('Error decoding q15:', e)
                    buffer = b''
        except KeyboardInterrupt:
            break
            
    ser.close()
    
    if training_data:
        print('Saving {} features to {}'.format(len(training_data), args.output))
        # Decide data type based on the format used
        dtype = np.int16 if args.keep_int else np.float32
        arr = np.array(training_data, dtype=dtype)
        np.save(args.output, arr)
        print('Numpy file saved.')
    else:
        print('No samples received.')

if __name__ == '__main__':
    main()
