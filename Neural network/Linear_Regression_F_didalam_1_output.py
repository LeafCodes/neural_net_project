import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

#Deklarasi Data
os.chdir('/Users/ASUS/Downloads/skripsi/data')
data = pd.read_csv('datalemari(-1,1).csv', sep=',', header=0)
data.head()
x           = data.iloc[:,:2].values
y           = data.iloc[:,2].values

# Calculate the mean of x and y
mean_x = np.mean(x, axis=0)
mean_y = np.mean(y)

# Center the data (subtract the mean from each column of x and y)
centered_x = x - mean_x
centered_y = y - mean_y

# Calculate the numerator and denominator for the slope (m) calculation
numerator = np.dot(centered_x.T, centered_y)
denominator = np.dot(centered_x.T, centered_x)

# Calculate the slope (m) and the intercept (b)
slope_m = np.linalg.solve(denominator, numerator)
intercept_b = mean_y - np.dot(mean_x, slope_m)

# Predicted y values using the calculated line equation
predicted_y = np.dot(x, slope_m) + intercept_b

# Print the slope and intercept
print("Slope (m):", slope_m)
print("Intercept (b):", intercept_b)

# Print the predicted y values
print("Predicted y:", predicted_y)

# Plot
sumbu_x = np.array([0, 2.5])
garis =  - (sumbu_x*(slope_m[0]/slope_m[1])) - (intercept_b/slope_m[1])
print(garis)
plt.plot(x[0:5, 0], x[0:5, 1], 'or', label='lemari')
plt.plot(x[5:12, 0], x[5:12, 1], 'ob', label='buffet')
plt.plot(x[12:16, 0], x[12:16, 1], 'og', label='wardrobe')
plt.plot(sumbu_x, garis, '-r', label='garis 1')
plt.xlabel('Lebar')
plt.ylabel('Tinggi')
plt.xlim(0.5, 2.5)
plt.ylim(0, 2.5)
plt.title('Linear Regression')
plt.legend()
plt.show()