# Markowitz — Technology + Industrials

Primera fase de una aplicación Streamlit para una tarea de teoría de portafolios.

## Qué hace

- Consulta acciones de EE.UU. de los sectores Technology e Industrials mediante Yahoo Finance/yfinance.
- Descarga 5 años de precios diarios ajustados.
- Exige una cobertura mínima del periodo.
- Calcula rendimiento acumulado, CAGR y volatilidad anual.
- Selecciona el Top N de cada sector.
- Prepara el universo final que después se usará en Markowitz.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

## Próxima fase

Se añadirá:
- rendimientos diarios,
- matriz de covarianzas y correlaciones,
- simulación Monte Carlo,
- portafolio de mínima varianza,
- portafolio de máximo Sharpe,
- frontera eficiente,
- comparación contra Equal Weight.
