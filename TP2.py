import numpy as np
import matplotlib.pyplot as plt
#question 1 :
def polynome(x):
    return x**3 + 2*x**2 + 3

x = np.arange(-10,10,0.01)
# Tracé de la courbe
plt.figure() #Crée une nouvelle figure.
plt.plot(x, polynome(x), label="f(x) = x^3 + 2x^2 + 3")

# Ajout des étiquettes et du titre
plt.xlabel('x') #Ajoute une étiquette à l'axe des xx.
plt.ylabel('f(x)') #Ajoute une étiquette à l'axe des yy.
plt.title('Courbe du polynôme x^3 + 2x^2 + 3')
plt.grid(True)  # Ajouter une grille
plt.legend()  # Afficher la légende
plt.show()


#question 2:
data = np.loadtxt("/adhome/a/ad/adam.kilani/Bureau/analyse/polynome.txt")


# Séparer les colonnes
x = data[:, 0]  # Première colonne : valeurs de x
y = data[:, 1]  # Deuxième colonne : valeurs de y

# Tracer les données

plt.scatter(x, y, color='blue', label="Données")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Données du polynôme')
plt.grid(True)
plt.legend()
plt.show()

#question 3:

# Calculer les logarithmes
log_x = np.log(x)
log_y = np.log(y)

# Tracer log(y) en fonction de log(x)

plt.scatter(log_x, log_y, color='red', label="log(y) vs log(x)")
plt.xlabel('log(x)')
plt.ylabel('log(y)')
plt.title('Changement de variables en log')
plt.grid(True)
plt.legend()
plt.show()

# Ajuster une droite de régression linéaire pour estimer la pente
coefficients = np.polyfit(log_x, log_y, 1)
slope = coefficients[0]
print(f"Pente estimée : {slope}")

#question 4 :


from sklearn.metrics import mean_squared_error

# Fonction pour effectuer la régression polynomiale et calculer les résidus
def polynomial_regression(x, y, degree):
    # Ajuster le polynôme
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)

    # Calculer les valeurs prédites
    y_pred = polynomial(x)

    # Calculer les résidus
    residuals = y - y_pred

    # Tracer les données et la courbe de régression
    plt.figure()
    plt.scatter(x, y, color='blue', label="Données")
    plt.plot(x, y_pred, color='red', label=f"Polynôme degré {degree}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Régression polynomiale (degré {degree})')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Retourner les résidus
    return residuals

# Régression polynomiale de degré 2, 3 et 4
residuals_deg2 = polynomial_regression(x, y, 2)
residuals_deg3 = polynomial_regression(x, y, 3)
residuals_deg4 = polynomial_regression(x, y, 4)

# Calculer l'erreur quadratique moyenne (MSE) pour chaque degré
mse_deg2 = mean_squared_error(y, np.polyval(np.polyfit(x, y, 2), x))
mse_deg3 = mean_squared_error(y, np.polyval(np.polyfit(x, y, 3), x))
mse_deg4 = mean_squared_error(y, np.polyval(np.polyfit(x, y, 4), x))

print(f"MSE (degré 2) : {mse_deg2}")
print(f"MSE (degré 3) : {mse_deg3}")
print(f"MSE (degré 4) : {mse_deg4}")



#exercice2 :