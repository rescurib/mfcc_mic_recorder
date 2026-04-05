import numpy as np
import librosa
from matplotlib import pyplot as plt

def compute_delta_mfcc(mfcc, win_length=2):
    N = win_length
    denominator = 2 * np.sum(np.arange(1, N + 1) ** 2)

    # Pad the MFCC matrix along the time axis using edge values
    padded = np.pad(mfcc, ((0, 0), (N, N)), mode='edge')
    delta_mfcc = np.zeros_like(mfcc)
    for t in range(mfcc.shape[1]):
        # For each frame, use the padded window
        c = padded[:, t : t + 2 * N + 1]
        # Numerator: sum n * (c[t+n] - c[t-n])
        numerator = 0.0
        for n in range(1, N + 1):
            numerator += n * (c[:, N + n] - c[:, N - n])
        delta = numerator / denominator
        delta_mfcc[:, t] = delta
    return delta_mfcc

wav_file = "e_32ms.wav"

# Import wav file
audio, sr = librosa.load(wav_file,
                             sr=8000,
                             dtype=np.float32,
                             mono = True)

# Compute MFCC features
mfcc = librosa.feature.mfcc(y=audio, 
                            sr=sr, 
                            n_mfcc=13,
                            hop_length = 256,
                            n_fft = 256,
                            #norm = None,
                            window = "hamming",
                            htk = True,
                            power = 1
                            )

delta_mfcc = compute_delta_mfcc(mfcc)

# Print MFCC features
#print("Number of hops: ", mfcc.shape[1])
#for i in range(mfcc.shape[0]):
#    print(f"MFCC {i+1}: {mfcc[i]}")
#    print(f"Delta MFCC {i+1}: {delta_mfcc[i]}")

#Print fist column of delta MFCCs
print("First Delta MFCCs for the first frame:")
print(delta_mfcc[0, :])


#Plot a single MFCC coefficient over time
coef_index = 0  # Index of the MFCC coefficient to plot (0-based)
y_coef = mfcc[coef_index]  # First MFCC coefficient
t = np.arange(len(y_coef)) * (256 / sr)  # Time axis in seconds
plt.figure(figsize=(10, 4))
plt.plot(t, y_coef, marker='o', linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("MFCC Coefficient")
plt.title(f"MFCC Coefficient {coef_index + 1} Over Time")
plt.grid(True)
plt.show()