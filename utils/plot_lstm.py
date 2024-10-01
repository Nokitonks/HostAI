import numpy as np
import matplotlib.pyplot as plt
def plot_lstm(x,y):
    # Plotting the two sections (clumps) for the new data with the updated x-values
    plt.figure(figsize=(12, 6))
    x = x[:-1]
    plt.subplot(1, 3, 3)
    plt.bar(x,y, width=1.0, color='red')
    plt.title('All data')
    plt.xlabel('Action Numbers')
    plt.ylabel('Values')
    # First clump: action numbers below 60
    plt.subplot(1, 3, 1)
    plt.bar(x[(x < 60)],y[(x < 60)], width=1.0, color='green')
    plt.title('Zoom in on Action Numbers below 60 (New Data)')
    plt.xlabel('Action Numbers')
    plt.ylabel('Values')

    # Second clump: action numbers around 270
    plt.subplot(1, 3, 2)
    plt.bar(x[(x > 260) & (x < 280)],
            y[(x > 260) & (x < 280)],
            width=1.0, color='orange')
    plt.title('Zoom in on Action Numbers around 270 (New Data)')
    plt.xlabel('Action Numbers')
    plt.ylabel('Values')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    x =[261.,  43., 230., 228.,   7.,  32., 582., 347., 258., 272., 582., 251.,
          25.,  14.,  54., 226., 231.,   4., 582., 236.,  24.,  36.,  18., 582.,
         220.,  37., 269., 265.,  51.,   2., 249., 246., 582., 158.,  53., 267.,
          19.,  44., 255., 582., 232.,  21.,  46.,  20.,  28.,  40., 582., 221.,
          42.,  23., 223.,  50.,  27., 582.,  15., 259., 253., 254., 582., 582.,
         233., 225., 261.,  43.,  32., 228.,   7., 230.,  22., 582., 582.,   9.,
         258., 234., 226., 272., 231.,  54.,  25. ,50 ]
    y =[ 4.2718e+01, -8.0660e+00, -3.2497e+00,  7.7262e+00, -5.8455e+00,
         6.1110e+00,  1.2423e+01, -8.4270e+00, -5.0775e+00, -4.2082e+00,
         3.3919e+00, -4.6816e+00, -1.6325e+00, -8.8919e-01, -4.4394e-02,
         8.7761e-01, -9.0785e-01, -9.0704e-02,  6.9339e-01, -2.3154e-01,
        -2.3873e-01,  6.1992e-01, -8.6474e-03,  1.0091e+00, -1.7924e+00,
         4.8771e-01,  4.6455e-01,  1.4152e+00,  8.0279e-01,  7.0820e-01,
         9.8571e-01,  3.7201e-01,  8.2124e-01,  1.9430e-01, -8.6448e-01,
        -6.2045e-01, -6.2497e-01, -3.9639e-02,  1.8379e-01, -3.2211e-01,
         2.1271e-01,  1.2872e-01, -6.6552e-02, -6.1134e-01, -3.9186e-01,
         1.1581e-01, -1.0106e-01,  4.5982e-01,  9.3217e-02, -3.9987e-01,
         3.3789e-02, -7.7739e-02, -1.1430e-01,  3.4409e-02,  4.4014e-01,
         2.8074e-02, -9.6893e-02,  1.5338e-01,  4.1714e-01,  2.6235e-01,
        -3.4069e-01, -4.6949e-01, -2.6045e-01, -2.5009e-01,  2.6659e-01,
         2.4421e-02, -1.8279e-01, -8.9123e-02,  2.0313e-01,  5.5499e-02,
         1.8866e-01, -3.6286e-01, -4.2105e-03, -7.5571e-01, -1.1161e-01,
        -1.1802e-01, -3.4070e-01, -1.9062e-01, -1.5226e-01]
    plot_lstm(np.array(x),np.array(y))