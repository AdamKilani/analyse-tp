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
polynome = np.loadtxt("/adhome/a/ad/adam.kilani/Bureau/analyse/polynome.txt")
print(polynome)


