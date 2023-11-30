# https://physionet.org/content/ptb-xl/1.0.3/records100/01000/#files-panel
import numpy as np
import wfdb

# Load an example ECG signal
record = wfdb.rdrecord('01000_lr')

# Extract the leads
I = record.p_signal[:, 0]  # Lead I
II = record.p_signal[:, 1]  # Lead II
V1 = record.p_signal[:, 6]  # Lead V1
V5 = record.p_signal[:, 10]  # Lead V5


# Parameters(also junk)
M = 10  # Determined after multiple iterations
delta_e = 0.05  # delta_e value
C = 1  # constant value

n = len(I)
x1 = [0]*n
x2 = [0]*n
x3 = [0]*n
d12 = [0]*n
d23 = [0]*n
e = [0]*n
W2 = [0]*n
W3 = [0]*n
K = 2  # Taken as 2 for the ease of hardware implementation

# Assuming I, II, V1, and V5 are defined and hold realistic data
III = [II[i] - I[i] for i in range(n)]
aVR = [-(I[i] + II[i]) / 2 for i in range(n)]
aVF = [(II[i] + III[i]) / 2 for i in range(n)]
aVL = [(I[i] - III[i]) / 2 for i in range(n)]

V2 = [(0.088733 * I[i]) - (0.09116 * II[i]) + (1.57862 * V1[i]) + (0.230214 * V5[i]) for i in range(n)]
V3 = [(0.245068 * I[i]) + (0.447773 * II[i]) + (1.14726 * V1[i]) + (0.609744 * V5[i]) for i in range(n)]
V4 = [(0.111111 * I[i]) + (0.064849 * II[i]) + (0.465706 * V1[i]) + (1.07423 * V5[i]) for i in range(n)]
V6 = [(0.202721 * I[i]) + (0.038811 * II[i]) - (0.176913 * V1[i]) + (0.59492 * V5[i]) for i in range(n)]

# Adaptive Linear Prediction
for i in range(3, n):
    # I-, II-, and III-order linear predictions
    x1[i] = I[i-1]
    x2[i] = 2*I[i-1] - I[i-2]
    x3[i] = 3*I[i-1] - 3*I[i-2] + I[i-3]

    # Slope parameters
    d12[i] = I[i-1] - I[i-2]
    d23[i] = I[i-2] - I[i-3]

    # Threshold value of the slope
    threshold = 10  # Chosen adaptively by considering the tradeoff between accuracy and CR

    # Final prediction
    if abs(d12[i]) < threshold and abs(d23[i]) < threshold:
        x_hat = x1[i]
    else:
        W2[i] = K * (M - ((e[i-1] + e[i-2] + e[i-3])/3))
        W3[i] = K * (M - ((e[i-1] + e[i-2] + e[i-3])/3))
        x_hat = (x2[i]*W2[i] + x3[i]*W3[i]) / (W2[i] + W3[i])
    print(f"Final prediction: i={i}: {e[i]}")
    # Final prediction error
    e[i] = I[i] - x_hat
    print(f"Final prediction error: i={i}: {e[i]}")

    # If the error exceeds the threshold
    if abs(e[i]) > threshold:
        # The weighted average of II- and III-order linear predictions is used for compression
        pred_val = (x2[i]*(2**C - e[i]) + x3[i]*(2**C - e[i])) / (2**C - e[i] + 2**C - e[i])
        if delta_e == 0:
            pred_val = (x2[i]*(1+2*0) + x3[i]*(1+2*0)) / (x2[i] + x3[i])
        elif delta_e == 2:
            pred_val = (x2[i] >> 2) + (x2[i] >> 4) + (x2[i] >> 7) + (x3[i] >> 1) + (x3[i] >> 2) + (x3[i] >> 5)
        elif delta_e == 3:
            pred_val = (x2[i] >> 4) + (x2[i] >> 5) + (x2[i] >> 6) + (x3[i] >> 1) + (x3[i] >> 2) + (x3[i] >> 3)
        elif delta_e >= 4:
            pred_val = (x2[i] >> delta_e) + x3[i]
        else: # For negative values of delta_e, shifting operations of x2[n] and x3[n] are interchanged
            pred_val = (x3[i] >> abs(delta_e)) + x2[i]
        print(f"Predicted value at i={i}: {pred_val}")