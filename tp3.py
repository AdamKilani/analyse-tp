import numpy as np
import matplotlib.pyplot as plt

#question 1:

def f(x) :
  return x * np.sin(x)

a, b =0, np.pi/2
n=20
h=(b-a)/n
x_points = np.linspace (a ,b-h ,n)
I_rect =h*np.sum(f(x_points))

#question 2:

x_points = np.linspace(a, b, n+1)
I_trap = (h/2) * (f(a) + 2 * np.sum(f(x_points[1:-1])) + f(b))

#question 3-4:
n_values = [20, 50, 100, 150, 1000]
errors_rect = []
errors_trap = []

for n in n_values:
    h = (b - a)/n
    
    # Rectangle
    x_rect = np.linspace(a, b-h, n)
    I_rect = h * np.sum(f(x_rect))
    errors_rect.append(abs(I_rect - 1))
    
    # Trapèze
    x_trap = np.linspace(a, b, n+1)
    I_trap = (h/2) * (f(a) + 2 * np.sum(f(x_trap[1:-1])) + f(b))
    errors_trap.append(abs(I_trap - 1))

plt.figure()
plt.plot(1/np.array(n_values), errors_rect, 'o-', label='Rectangles')
plt.plot(1/np.array(n_values), errors_trap, 'o-', label='Trapèzes')
plt.xlabel('1/n')
plt.ylabel('Erreur absolue')
plt.legend()
plt.grid()
plt.show()

#question 6-7 : 
# Transformation logarithmique
log_n = np.log(n_values)
log_err_rect = np.log(errors_rect)
log_err_trap = np.log(errors_trap)

# Régression linéaire
coeff_rect = np.polyfit(log_n, log_err_rect, 1)
coeff_trap = np.polyfit(log_n, log_err_trap, 1)

print(f"Ordre méthode rectangles: {-coeff_rect[0]}")
print(f"Ordre méthode trapèzes: {-coeff_trap[0]}")
