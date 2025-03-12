import numpy as np
import matplotlib.pyplot as plt

# Créer un tableau de valeurs entre 0 et 10
x = np.arange(0, 10, 1)  # On peut aussi écrire x = np.linspace(debut, fin, nb_valeurs)

y = np.sqrt(x)  # Calcule la racine carrée de chaque élément du tableau x
y1 = x * x  # La fonction x^2

# Tracer la première fonction
plt.figure()
plt.plot(x, y, label="sqrt(x)")
plt.legend()
plt.show()  # Affichage de la fenêtre
plt.close()
# Tracer la deuxième fonction
plt.figure()
plt.plot(x, y1, label="x^2", color="red")
plt.legend()
plt.show()  # Affichage de la fenêtre
plt.close()
# Question 6 : Fonction somme
def somme(x, y):
    return x + y

res = somme(x, y)
print("Somme des éléments :", res)

# Question 7 : Création de la matrice
matrice = np.array([[1, -1, 2], [4, -6, 12], [-1, -5, 12]])  # Correction ici
print("Matrice :\n", matrice)

# Produit matriciel #1 methode
M = np.array([[1, -1, 2], [4, -6, 12], [-1, -5, 12]])
N = np.array([[1, -1, 2], [4, -6, 12], [-1, -5, 12]])
produit = np.dot(M, N)  # Produit matriciel
print("Produit matriciel M * N :\n", produit)


print(f"Matrice :\n{matrice}")
# 8. Addition et multiplication de la matrice par elle-même
addition = matrice + matrice
multiplication = np.dot(matrice, matrice)  # Produit matriciel #2 eme methode

# Multiplication terme à terme
multiplication_terme_a_terme = matrice * matrice

print("\nAddition de la matrice par elle-même :\n", addition)
print("\nMultiplication matricielle :\n", multiplication)
print("\nMultiplication terme à terme :\n", multiplication_terme_a_terme)

# 9. Déterminant et inverse
determinant = np.linalg.det(matrice)

# Vérification si la matrice est inversible (déterminant non nul)
if determinant != 0:
    inverse = np.linalg.inv(matrice)
    print("\nInverse de la matrice :\n", inverse)
else:
    print("\nLa matrice n'est pas inversible car son déterminant est nul.")

print("\nDéterminant de la matrice :", determinant)

# 10. Extraction de la première ligne et somme de ses éléments
premiere_ligne = matrice[0, :]
somme_premiere_ligne = np.sum(premiere_ligne)

# Calcul du vecteur des sommes ligne par ligne
somme_par_ligne = np.sum(matrice, axis=1)

print("\nPremière ligne :\n", premiere_ligne)
print("\nSomme des éléments de la première ligne :", somme_premiere_ligne)
print("\nVecteur des sommes ligne par ligne :\n", somme_par_ligne)

# 11. Application de la fonction cosinus
cos_matrice = np.cos(matrice)
print("\nCosinus de la matrice :\n", cos_matrice)






#exercice2
#1
matrice2 = np.loadtxt("/adhome/a/ad/adam.kilani/Bureau/analyse/Kangourous.txt") #transforme le fichier en matrice
print(matrice2)
print(matrice2.shape)
plt.plot()
plt.figure()
plt.show()
#2


# Extraire la longueur (colonne 0) et la largeur (colonne 1)
longueur = data[:, 0]
largeur = data[:, 1]

# Tracer la largeur en fonction de la longueur
plt.scatter(longueur, largeur, color='blue', label='Données brutes')
plt.xlabel('Longueur du nez (mm)')
plt.ylabel('Largeur du nez (mm)')
plt.title('Largeur en fonction de la longueur du nez des kangourous')
plt.legend()
plt.grid(True)
plt.show()

#3 
# Calculer la régression linéaire (degré 1)
coefficients = np.polyfit(longueur, largeur, 1)

# Afficher les coefficients
print(f"Coefficients de la régression linéaire: {coefficients}")

#4
# Calculer les valeurs prédites par la régression linéaire
largeur_predite = np.polyval(coefficients, longueur)

# Tracer les données brutes et la droite de régression
plt.scatter(longueur, largeur, color='blue', label='Données brutes')
plt.plot(longueur, largeur_predite, color='red', label='Régression linéaire')
plt.xlabel('Longueur du nez (mm)')
plt.ylabel('Largeur du nez (mm)')
plt.title('Régression linéaire sur les données des kangourous')
plt.legend()
plt.grid(True)
plt.show()




