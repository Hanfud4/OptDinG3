import random
import math
import numpy as np
import numba as nb
from numba import njit
import gurobipy as gp
from gurobipy import GRB


@njit
def funcion_p(a, h):
    valor1 = (2 * a + 1)
    valor2 = (a + 1) ** h
    valor3 = 0
    for k in range(0, a + 1):
        valor3 += k ** h
    valor3 *= 2
    return valor1 * valor2 - valor3


@njit
def prob_triangular_discreta(a, h, c, d):
    if d < c - a or d > c + a:
        return 0
    numerador = (a + 1) ** h - abs(d - c) ** h
    denominador = funcion_p(a, h)
    prob = numerador / denominador
    return prob


@njit
def va_triangular_discreta(a, h, c):
    u = np.random.random()
    # print(u)
    p_acum = 0
    for d in range(c - a, c + a + 1):
        prob = prob_triangular_discreta(a, h, c, d)
        if p_acum <= u < p_acum + prob:
            return d
        p_acum += prob
    return d


@njit
def esperanza_triangular_discreta(a, h, c):
    esperanza = 0
    for d in range(c - a, c + a + 1):
        esperanza += d * prob_triangular_discreta(a, h, c, d)
    return round(esperanza)


def varianza_triangular_discreta(a, h, c):
    prob_a_h = funcion_p(a, h)

    varianza = (a * (2 * a + 1) * (a + 1) ** (h + 1) / 3) - \
        2 * sum(k ** (h + 2) for k in range(a + 1))

    return varianza/prob_a_h


@njit
def a_value(t):
    if t <= 15:
        return 50
    elif t <= 30:
        return 86
    elif t <= 45:
        return 34
    elif t <= 60:
        return 22
    elif t <= 75:
        return 100
    return 66


@njit
def h_value(t):
    if t <= 15:
        return 5
    elif t <= 30:
        return 4
    elif t <= 45:
        return 4
    elif t <= 60:
        return 5
    elif t <= 75:
        return 3
    return 4


@njit
def c_value(t):
    if t <= 15:
        return 50
    elif t <= 30:
        return 86
    elif t <= 45:
        return 34
    elif t <= 60:
        return 22
    elif t <= 75:
        return 100
    return 66


@njit
def K_value(t):
    if t <= 30:
        return 180
    elif t <= 60:
        return 500
    return 250


@njit
def C_value(t):
    if (t % 7 == 0):
        return 18
    return 12


@njit
def H_value(t):
    return 1


@njit
def Q_value(t):
    return 18


@njit
def politica_optima(T, Q):
    array_size = (T+1, Q + 1)
    matriz_costos = np.zeros(array_size)
    matriz_decision = np.zeros(array_size)
    matriz_costos_inmediatos = np.zeros((Q + 1, Q + 1))

    # for s in range(Q + 1):
    #     for x in range(0, Q - s + 1):
    #         costo_esperado = 0
    #         if x > 0:
    #             costo_esperado += K_value(T) + C_value(T) * x
    #         for d in range(c_value(T) - a_value(T), c_value(T) + a_value(T) + 1):
    #             costo_esperado += Q_value(T) * max(0, d - s - x) * prob_triangular_discreta(a_value(T), h_value(T), c_value(T), d)
    #             costo_esperado += H_value(T) * max(0, s + x - d) * prob_triangular_discreta(a_value(T), h_value(T), c_value(T), d)
    #         matriz_costos_inmediatos[s][x] = costo_esperado

    # print("Costos Inmediatos:")
    # for s in range(0, Q + 1):
    #     minimum = np.inf
    #     x_value = -1
    #     for x in range(0, Q - s + 1):
    #         if matriz_costos_inmediatos[s][x] < minimum:
    #             minimum = matriz_costos_inmediatos[s][x]
    #             x_value = x
    #     matriz_decision[T - 1][s] = x_value
    #     matriz_costos[T - 1][s] = matriz_costos_inmediatos[s][x_value]

    for s in range(0, Q + 1):
        matriz_costos[T][s] = 0

    # print(matriz_decision)

    for t in range(T - 1, -1, -1):
        for s in range(0, Q + 1):
            # escoge acción óptima.
            minimum = np.inf
            x_value = -1
            for x in range(0, Q - s + 1):
                # costo inmediato
                CostoAccion = 0
                if x > 0:
                    CostoAccion += K_value(t+1) + C_value(t+1) * x
                for d in range(c_value(t+1) - a_value(t+1), c_value(t+1) + a_value(t+1) + 1):
                    CostoAccion += Q_value(t+1) * max(0, d - s - x) * prob_triangular_discreta(
                        a_value(t+1), h_value(t+1), c_value(t+1), d)
                    CostoAccion += H_value(t+1) * max(0, s + x - d) * prob_triangular_discreta(
                        a_value(t+1), h_value(t+1), c_value(t+1), d)
                # valor esperado futuro
                for d in range(c_value(t+1) - a_value(t+1), c_value(t+1) + a_value(t+1) + 1):
                    sfuturo = max(s + x - d, 0)
                    CostoAccion += matriz_costos[t+1][sfuturo] * prob_triangular_discreta(
                        a_value(t+1), h_value(t+1), c_value(t+1), d)

                # Actualiza
                if CostoAccion < minimum:
                    minimum = CostoAccion
                    x_value = x

            matriz_costos[t][s] = minimum
            matriz_decision[t][s] = x_value

    return matriz_costos, matriz_decision


T = 90
Q = 120

matriz_costos, matriz_decision = politica_optima(T, Q)


# for i in range(T+1):
#    print(i+1, matriz_costos[i])
#    print(i+1, matriz_decision[i])
#
#
# print(matriz_costos[0][40])


def resolver_problema_inventario_con_quiebre_de_stock(demandas):
    # Crear el modelo
    modelo = gp.Model("Problema_de_Inventario_con_Quiebre_de_Stock")

    # Parámetros
    num_periodos = len(demandas)

    # Variables de decisión
    inventario = modelo.addVars(num_periodos, name="Inventario")
    ordenar = modelo.addVars(num_periodos, vtype=GRB.BINARY, name="Ordenar")
    cantidad_ordenar = modelo.addVars(
        num_periodos, vtype=GRB.INTEGER, name="Cantidad_Ordenar")
    quiebre_stock = modelo.addVars(
        num_periodos, vtype=GRB.INTEGER, name="Quiebre_Stock")

    # Función objetivo: Minimizar el costo total
    modelo.setObjective(
        gp.quicksum(K_value(t) * ordenar[t] + C_value(t) * cantidad_ordenar[t] + H_value(
            t) * inventario[t] + quiebre_stock[t] * Q_value(t) for t in range(num_periodos)),
        GRB.MINIMIZE
    )

    # Restricciones de inventario
    modelo.addConstr(inventario[0] == 0, "Inicial")
    for t in range(1, num_periodos):
        modelo.addConstr(inventario[t-1] + cantidad_ordenar[t] - int(
            demandas[t]) + quiebre_stock[t] == inventario[t], f"Inventario_{t}")

    # Restricción de capacidad máxima de inventario
    for t in range(num_periodos):
        modelo.addConstr(inventario[t] <= 120, f"Capacidad_{t}")

    # Vincular variable binaria "Ordenar" con "Cantidad_Ordenar"
    for t in range(num_periodos):
        # Un valor grande para M (mayor que la capacidad máxima)
        M = 120
        modelo.addConstr(
            cantidad_ordenar[t] <= M * ordenar[t], f"Ordenar_Constraint_{t}")

    modelo.Params.OutputFlag = 0

    # Resolver el modelo
    modelo.optimize()

    return modelo.objVal / num_periodos


@njit
def simular_demanda(cantidad_iteraciones, T):
    out = ""
    for i in range(cantidad_iteraciones):
        for t in range(1, T + 1):
            a = a_value(t)
            h = h_value(t)
            c = c_value(t)
            demanda = va_triangular_discreta(a, h, c)
            out += str(demanda) + ","
        out = out[:-1]
        out += "\n"
    return out


def simulacion_aplicando_politica_optima(demanda, matriz_decision):
    stock_inicial = 40
    stock = stock_inicial
    costo_total = 0
    for t in range(1, T + 1):
        # print("Tiempo:", t)
        # print("Stock:", stock)
        # print("Demanda:", demanda[t - 1])
        # print("Decision:", matriz_decision[t - 1][int(stock)])
        reposicion = matriz_decision[t - 1][stock]
        # print("Reposicion:", reposicion)
        if reposicion > 0:
            costo_total += K_value(t) + C_value(t) * reposicion
        costo_total += Q_value(t) * max(0, demanda[t - 1] - stock - reposicion)
        # print(Q_value(t) * max(0, demanda[t - 1] - stock - reposicion))
        costo_total += H_value(t) * max(0, stock + reposicion - demanda[t - 1])
        # print(H_value(t) * max(0, stock + reposicion - demanda[t - 1]))
        # print("Stock - Demanda:", stock)
        stock = int(max(stock + reposicion - demanda[t - 1], 0))
        # print("Costo:", costo_total)
        # print()
    return costo_total/T


def simulacion_aplicando_politica_conservadora(demanda):
    stock_inicial = 40
    stock = stock_inicial
    costo_total = 0
    for t in range(1, T + 1):
        # print("Tiempo:", t)
        # print("Stock:", stock)
        # print("Demanda:", demanda[t - 1])
        # print("Decision:", matriz_decision[t - 1][int(stock)])
        reposicion = 120 - stock
        costo_total += K_value(t) + C_value(t) * reposicion
        costo_total += Q_value(t) * max(0, demanda[t - 1] - stock - reposicion)
        # print(Q_value(t) * max(0, demanda[t - 1] - stock - reposicion))
        costo_total += H_value(t) * max(0, stock + reposicion - demanda[t - 1])
        # print(H_value(t) * max(0, stock + reposicion - demanda[t - 1]))
        # print("Stock - Demanda:", stock)
        stock = int(max(stock + reposicion - demanda[t - 1], 0))
        # print("Costo:", costo_total)
        # print()
    return costo_total/T


def simulacion_aplicando_politica_SS(demanda):
    stock_inicial = 40
    stock = stock_inicial
    costo_total = 0
    for t in range(1, T + 1):
        # print("Tiempo:", t)
        # print("Stock:", stock)
        # print("Demanda:", demanda[t - 1])
        # print("Decision:", matriz_decision[t - 1][int(stock)])
        SS = np.ceil(esperanza_triangular_discreta(a_value(t), h_value(t), c_value(
            t)) + 2*np.sqrt(varianza_triangular_discreta(a_value(t), h_value(t), c_value(t))))
        if stock < SS:
            reposicion = 120 - stock
        else:
            reposicion = 0
        costo_total += K_value(t) + C_value(t) * reposicion
        costo_total += Q_value(t) * max(0, demanda[t - 1] - stock - reposicion)
        # print(Q_value(t) * max(0, demanda[t - 1] - stock - reposicion))
        costo_total += H_value(t) * max(0, stock + reposicion - demanda[t - 1])
        # print(H_value(t) * max(0, stock + reposicion - demanda[t - 1]))
        # print("Stock - Demanda:", stock)
        stock = int(max(stock + reposicion - demanda[t - 1], 0))
        # print("Costo:", costo_total)
        # print()
    return costo_total/T


M = 500
# output = simular_demanda(M, T)
#
# with open("demanda.csv", "w") as f:
#    f.write(output)

with open("demanda.csv", "r") as f:
    demanda = f.readlines()

suma_po = 0
po_lista = []
suma_pc = 0
suma_pss = 0
suma_pvi = 0
pvi_lista = []
for i in range(M):
    dem = demanda[i].split(",")
    for j in range(len(dem)):
        dem[j] = int(dem[j])
    po = simulacion_aplicando_politica_optima(dem, matriz_decision)
    suma_po += po
    po_lista.append(po)
    suma_pc += simulacion_aplicando_politica_conservadora(dem)
    suma_pss += simulacion_aplicando_politica_SS(dem)
    pvi = resolver_problema_inventario_con_quiebre_de_stock(dem)
    suma_pvi += pvi
    pvi_lista.append(pvi)

print(suma_po / M)
print(suma_pc/M)
print(suma_pss/M)
print(suma_pvi/M)

po_vs_pvi = 0
for i in range(len(po_lista)):
    po_vs_pvi += (po_lista[i]-pvi_lista[i])

print(po_vs_pvi/M)
