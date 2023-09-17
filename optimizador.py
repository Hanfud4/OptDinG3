import time
import random
import math
import numpy as np
import numba as nb
from numba import njit
import gurobipy as gp
from gurobipy import GRB


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
