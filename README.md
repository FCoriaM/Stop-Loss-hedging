# Monte Carlo Hedging Simulation

Simulación de precios de activos financieros y estrategias de cobertura (Stop-Loss y Delta Hedging) usando Python.

## Instalación

```bash
git clone https://github.com/tu_usuario/Stop-Loss-hedging.git
cd Stop-Loss-hedging
```
## Preparación del entorno
Se recomienda crear un entorno virtual antes de instalar las dependencias:

```bash
python -m venv .venv
source venv/bin/activate 
pip install -r requirements.txt
```

## Módulos
### price_simulation.py
En este módulo se realiza la simulación del precio del activo subyacente mediante el método MonteCarlo.
En cada ciclo del método, se actualiza el valor $\Delta S$, cuyo valor se obtiene de la fórmula $$\Delta S = \mu S \Delta t + \sigma S \epsilon \sqrt{\Delta t}$$ la cual denota el comportamiento del precio del activo según el **Movimiento Geométrico Browniano** en intervalos de tiempo pequenios. Para esto se debe simular el valor de $\epsilon$ con una distribución $N(0,1)$.

Luego de 1000 simulaciones de precio cada $\Delta t$ intervalos de tiempo, obtenemos la tabla con las siguientes variaciones

==Agregar tabla==



## Referencias

[John Hull](https://drive.google.com/file/d/1_92BVhgf5vjEW2htlPv8nARihjLWsQTk/view)
