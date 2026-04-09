import numpy as np
import matplotlib.pyplot as plt



# 1. O grafico de linha mais simples
x = np.linspace(0, 5, 6)
y = x**2
plt.figure()
plt.plot(x, y, 'o-')
plt.title("Funcao quadratica")
plt.xlabel("Temperatura (C)")
plt.ylabel("$ Resistência (\Omega) $")
plt.grid()
plt.show()


# 2. Comparando multiplas curvas
x = np.linspace(0, 2 * np.pi, 200)
plt.figure()
plt.plot(x, np.sin(x), label="seno", color="blue")
plt.plot(x, np.cos(x), label="cosseno", linestyle="--", color="red")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid(True)
plt.show()


# 3. Customizacoes basicas
plt.figure(figsize=(8, 3))
plt.plot(x, np.sin(x), marker="^", markersize=3, linewidth=1.5)
plt.title("Sinal senoidal")
plt.xlabel("tempo")
plt.ylabel("amplitude")
plt.grid(True, linestyle="dashdot")
plt.show()



# 4. Outros tipos de grafico: barras
categorias = ["A", "B", "C", "D"]
valores = [4, 7, 5, 8]
plt.figure()
plt.bar(categorias, valores)
plt.title("Comparacao por categoria")
plt.xlabel("categorias")
plt.ylabel("valores")
plt.show()
 

# 5. Outros tipos de grafico: dispersao
x_medidas = np.array([1, 2, 3, 4, 5, 6, 7])
y_medidas = np.array([2.0, 2.5, 3.7, 3.9, 5.1, 5.5, 6.8])
plt.figure()
plt.scatter(x_medidas, y_medidas)
plt.title("Dispersao entre medidas")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# 6. Outros tipos de grafico: histograma
amostras = np.array([4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11])
plt.figure()
plt.hist(amostras, bins=8, edgecolor="black")
plt.title("Distribuicao das amostras")
plt.xlabel("valor")
plt.ylabel("frequencia")
plt.show()