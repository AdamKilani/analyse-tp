# TP2 - Analyse de données et régression polynomiale

Ce TP vise à analyser un ensemble de données et à appliquer une régression polynomiale pour modéliser ces données. Nous utiliserons Python avec les bibliothèques `numpy` et `matplotlib` pour charger, visualiser et analyser les données.

---

## 1. Tracer un polynôme

### Code :
```python
import numpy as np
import matplotlib.pyplot as plt

# Définition du polynôme
def polynome(x):
    return x**3 + 2*x**2 + 3

# Génération des valeurs de x
x = np.arange(-10, 10, 0.01)

# Tracé de la courbe
plt.figure()
plt.plot(x, polynome(x), label="f(x) = x^3 + 2x^2 + 3")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Courbe du polynôme x^3 + 2x^2 + 3')
plt.grid(True)
plt.legend()
plt.show()

Explication :

    polynome(x): Définit la fonction polynomiale f(x)=x3+2x2+3f(x)=x3+2x2+3.
    np.arange(-10, 10, 0.01): Génère des valeurs de xx entre -10 et 10 avec un pas de 0.01.
    plt.plot(): Trace la courbe du polynôme.
    plt.xlabel(), plt.ylabel(), plt.title(): Ajoutent des étiquettes et un titre au graphique.
    plt.grid(True): Active la grille pour une meilleure lisibilité.
    plt.legend(): Affiche la légende.
    plt.show(): Affiche le graphique.

2. Importer les données et les tracer
Code :

# Charger les données depuis le fichier
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

Explication :

    np.loadtxt(): Charge les données du fichier polynome.txt dans un tableau NumPy.
    x = data[:, 0]: Extrait la première colonne (valeurs de xx).
    y = data[:, 1]: Extrait la deuxième colonne (valeurs de yy).
    plt.scatter(): Trace un nuage de points représentant les données.
    plt.xlabel(), plt.ylabel(), plt.title(): Ajoutent des étiquettes et un titre au graphique.
    plt.grid(True): Active la grille pour une meilleure lisibilité.
    plt.legend(): Affiche la légende.
    plt.show(): Affiche le graphique.

3. Changement de variables en log
Code :

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

Explication :

    np.log(x), np.log(y): Calculent les logarithmes naturels de xx et yy.
    plt.scatter(): Trace un nuage de points représentant log⁡(y)log(y) en fonction de log⁡(x)log(x).
    plt.xlabel(), plt.ylabel(), plt.title(): Ajoutent des étiquettes et un titre au graphique.
    plt.grid(True): Active la grille pour une meilleure lisibilité.
    plt.legend(): Affiche la légende.
    np.polyfit(log_x, log_y, 1): Ajuste une droite de régression pour estimer la pente.
    slope = coefficients[0]: Récupère la pente de la droite de régression.
    print(f"Pente estimée : {slope}"): Affiche la pente estimée, qui donne une indication du degré du polynôme.

4. Régression polynomiale et calcul des résidus
Code :

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

Explication :

    np.polyfit(x, y, degree): Ajuste un polynôme de degré donné aux données.
    np.poly1d(coefficients): Crée une fonction polynomiale à partir des coefficients.
    y_pred = polynomial(x): Calcule les valeurs prédites par le polynôme.
    residuals = y - y_pred: Calcule les résidus (différences entre valeurs réelles et prédites).
    plt.scatter(), plt.plot(): Tracent les données et la courbe de régression.
    mean_squared_error(y, y_pred): Calcule l'erreur quadratique moyenne (MSE).
    print(f"MSE (degré X) : {mse_X}"): Affiche l'erreur pour chaque degré.

Résumé des étapes :

    Tracer un polynôme.
    Importer les données et les afficher sous forme de nuage de points.
    Appliquer un changement de variables en log pour estimer le degré du polynôme.
    Effectuer une régression polynomiale et calculer les résidus pour les degrés 2, 3 et 4.

Résultats attendus :

    Graphiques des données et des régressions polynomiales.
    Estimation du degré du polynôme via la pente en échelle logarithmique.
    Comparaison des erreurs quadratiques moyennes pour chaque modèle.
