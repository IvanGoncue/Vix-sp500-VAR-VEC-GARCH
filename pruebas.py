# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:08:40 2024

@author: GonCue
"""

import pandas as pd

import yfinance as yf

import matplotlib.pyplot as plt

import numpy as np

from arch import arch_model

from statsmodels.tsa.api import VAR

import seaborn as sns

 

#############################################
#############################################
#############################################
#############################################
# US500
#############################################
#############################################
#############################################
#############################################

################### CARGAMOS LOS DATOS US500 & VIX

ruta_SP500 = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\Prediccion en economia SIMULACION PORTFOLIO (VIX)\sp500.xlsx'
sp500 = pd.read_excel(ruta_SP500, index_col=0)
print(sp500)

ruta_vix = r"C:\Users\34653\Desktop\UNED\segundo cuatrimestre\Prediccion en economia SIMULACION PORTFOLIO (VIX)\vix.xlsx"
vix = pd.read_excel(ruta_vix, index_col=0)
print(vix)

######################### REPRESENTAR EL SP500

# Resetear el índice de fecha para convertirlo en una columna regular
sp500.reset_index(inplace=True)

# Convertir las columnas de fecha y cierre ajus a arrays de NumPy
fechas = sp500['Fecha'].values
cierre_ajustado = sp500['Cierre ajus SP500'].values

# Crear el gráfico para el DataFrame sp500
plt.figure(figsize=(10, 6))
plt.plot(fechas, cierre_ajustado, color='blue')
plt.title('Evolución del SP500')
plt.xlabel('Fecha')
plt.ylabel('Cierre ajustado')
plt.grid(True)
plt.show()




#############################################
#############################################
#############################################
#############################################
#VOLATILIDAD DEL SP500 MODELO ARCH AJUSTADO AL sp500
#############################################
#############################################
#############################################
#############################################

ruta_SP500 = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\Prediccion en economia SIMULACION PORTFOLIO (VIX)\sp500.xlsx'
sp5001 = pd.read_excel(ruta_SP500, index_col=0)

# Calcular los retornos
returns = sp5001['Cierre ajus SP500'].pct_change().dropna()

# Ajustar un modelo ARCH(1)
model = arch_model(returns, vol='Garch', p=1, o=0, q=0, dist='Normal')
results = model.fit()

# Resumen del modelo
print(results.summary())

# Extraer la volatilidad condicional estimada
cond_volatility = results.conditional_volatility

# Graficar la volatilidad condicional estimada
plt.figure(figsize=(10, 6))
cond_volatility.plot()
plt.title('Volatilidad Condicional Estimada del S&P 500 - Modelo ARCH(1)')
plt.xlabel('Fecha')
plt.ylabel('Volatilidad')
plt.grid(True)
plt.show()



#############################################
#############################################
#############################################
#############################################
#VIX
#############################################
#############################################
#############################################
#############################################




vix = vix[vix['Cierre ajus VIX'] != '-']
# Resetear el índice de fecha para convertirlo en una columna regular
vix.reset_index(inplace=True)

# Convertir las columnas de fecha y cierre ajus a arrays de NumPy
fechas_vix = vix['Fecha'].values
cierre_ajustado_vix = vix['Cierre ajus VIX'].values

# Crear el gráfico para el DataFrame vix
plt.figure(figsize=(10, 6))
plt.plot(fechas_vix, cierre_ajustado_vix, color='red')
plt.title('Evolución del VIX')
plt.xlabel('Fecha')
plt.ylabel('Cierre ajustado')
plt.grid(True)
plt.show()


######################### REPRESENTAR AMBOS PARA UN PERIODO CONCRETO


# Filtrar los datos de sp500 para el rango de fechas deseado
sp500_filtered = sp500[(sp500['Fecha'] >= '2005-01-01') & (sp500['Fecha'] <= '2022-01-01')]

# Filtrar los datos de vix para el rango de fechas deseado
vix_filtered = vix[(vix['Fecha'] >= '2005-01-01') & (vix['Fecha'] <= '2022-01-01')]

# Convertir las columnas de fecha y cierre ajus a arrays de NumPy para sp500
fechas_sp500 = sp500_filtered['Fecha'].values
cierre_ajustado_sp500 = sp500_filtered['Cierre ajus SP500'].values

# Convertir las columnas de fecha y cierre ajus a arrays de NumPy para vix
fechas_vix = vix_filtered['Fecha'].values
cierre_ajustado_vix = vix_filtered['Cierre ajus VIX'].values

# Crear el gráfico
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar SP500 en el primer eje y
color = 'black'
ax1.set_xlabel('Fecha')
ax1.set_ylabel('SP500', color=color)
ax1.plot(fechas_sp500, cierre_ajustado_sp500, color=color, label='SP500')
ax1.tick_params(axis='y', labelcolor=color)

# Crear un segundo eje y para el VIX
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('VIX', color=color)
ax2.plot(fechas_vix, cierre_ajustado_vix, color=color, label='VIX')
ax2.tick_params(axis='y', labelcolor=color)

# Configuración del gráfico
fig.suptitle('Evolución de SP500 y VIX (2005-2022)')
fig.tight_layout()
fig.legend(loc='upper left')
plt.xlabel('Fecha')
plt.grid(True)

# Mostrar el gráfico
plt.show()














































#############################################
#############################################
#############################################
#############################################
#VAR vix-us500, prediccion a 10 periodos
#############################################
#############################################
#############################################
#############################################

import pandas as pd
from statsmodels.tsa.api import VAR

# Cargar los DataFrames desde los archivos
ruta_SP500 = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\Prediccion en economia SIMULACION PORTFOLIO (VIX)\sp500.xlsx'
ruta_vix = r"C:\Users\34653\Desktop\UNED\segundo cuatrimestre\Prediccion en economia SIMULACION PORTFOLIO (VIX)\vix.xlsx"

sp500 = pd.read_excel(ruta_SP500, index_col=0)
vix = pd.read_excel(ruta_vix, index_col=0)

sp500 = sp500['Cierre ajus SP500']
vix = vix['Cierre ajus VIX']

# Asegurarse de que los DataFrames estén en el mismo orden
sp500 = sp500.sort_index(ascending=True)
vix = vix.sort_index(ascending=True)

# Unir los DataFrames en uno solo
df = pd.concat([sp500, vix], axis=1)

# Verificar tipos de datos
print(df.dtypes)

# Convertir tipos de datos si es necesario
df = df.apply(pd.to_numeric, errors='coerce')

# Eliminar filas con NaNs si es necesario
df = df.dropna()

# Crear y ajustar el modelo VAR

model = VAR(df)
order = model.select_order()
results = model.fit(maxlags=order.aic)

# Ver resumen del modelo
print(results.summary())


# Obtener el número de pasos hacia adelante que deseamos predecir
forecast_steps = 10  # Por ejemplo, predeciremos los próximos 10 períodos

# Realizar las predicciones futuras
predicciones = results.forecast(df.values[-results.k_ar:], steps=forecast_steps)


# Crear un DataFrame para las predicciones
pred_df = pd.DataFrame(predicciones, index=pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq=df.index.freq)[1:], columns=df.columns)

# Imprimir las predicciones
print(pred_df)




# Obtener los últimos 10 valores de la columna "Cierre ajus SP500"
ultimos_10_valores = sp500.tail(10)

# Crear un gráfico de líneas para los últimos 10 valores con Seaborn
plt.figure(figsize=(12, 8))

# Graficar los últimos 10 valores
sns.lineplot(x=ultimos_10_valores.index, y=ultimos_10_valores, marker='o', color='red', label='Últimos 10 valores observados S&P500')

# Graficar la columna "Cierre ajus SP500" de pred_df con Seaborn
sns.lineplot(data=pred_df['Cierre ajus SP500'], marker='o', label='Predicciones VAR')

# Ajustar las etiquetas de las fechas en el eje x
plt.xticks(rotation=45)  # Rotar las etiquetas 45 grados

# Configurar título y etiquetas de ejes
plt.title('Últimos 10 valores y predicciones del cierre ajustado del S&P 500')
plt.xlabel('Fecha')
plt.ylabel('Cierre ajustado')

# Mostrar la leyenda
plt.legend()

# Mostrar la gráfica
plt.grid(True)
plt.show()


###############################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# Cargar los DataFrames desde los archivos
ruta_SP500 = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\Prediccion en economia SIMULACION PORTFOLIO (VIX)\sp500.xlsx'
ruta_vix = r"C:\Users\34653\Desktop\UNED\segundo cuatrimestre\Prediccion en economia SIMULACION PORTFOLIO (VIX)\vix.xlsx"

sp500 = pd.read_excel(ruta_SP500, index_col=0)
vix = pd.read_excel(ruta_vix, index_col=0)

# Asegurarse de que los DataFrames estén en el mismo orden
sp500 = sp500.sort_index(ascending=True)
vix = vix.sort_index(ascending=True)

# Unir los DataFrames en uno solo
df = pd.concat([sp500, vix], axis=1)

# Verificar tipos de datos
print(df.dtypes)

# Convertir tipos de datos si es necesario
df = df.apply(pd.to_numeric, errors='coerce')

# Eliminar filas con NaNs si es necesario
df = df.dropna()

# Crear y ajustar el modelo VAR con el criterio de informacion de AIC

model = VAR(df)
order = model.select_order()
results = model.fit(maxlags=order.aic)



# Ver resumen del modelo
print(results.summary())

# Obtener el número de pasos hacia adelante que deseamos predecir
forecast_steps = 10  # Por ejemplo, predeciremos los próximos 10 períodos

# Realizar las predicciones futuras
predicciones = results.forecast(df.values[-results.k_ar:], steps=forecast_steps)

# Crear un DataFrame para las predicciones
pred_df = pd.DataFrame(predicciones, index=pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq=df.index.freq)[1:], columns=df.columns)

# Obtener los últimos 10 valores de la columna "Cierre ajus SP500" y "Cierre ajus VIX"
ultimos_10_valores_sp500 = sp500['Cierre ajus SP500'].tail(10).values
ultimos_10_valores_vix = vix['Cierre ajus VIX'].tail(10).values

# Obtener los datos de las fechas, cierre ajustado del S&P 500 y del VIX
fechas_sp500 = sp500.index[-10:].values
cierre_ajustado_sp500 = sp500['Cierre ajus SP500'].tail(10).values

fechas_vix = vix.index[-10:].values
cierre_ajustado_vix = vix['Cierre ajus VIX'].tail(10).values

# Obtener las fechas de las predicciones
fechas_prediccion = pred_df.index.values

# Obtener las predicciones para SP500 y VIX
predicciones_sp500 = pred_df['Cierre ajus SP500'].values
predicciones_vix = pred_df['Cierre ajus VIX'].values

# Crear el gráfico
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar SP500 en el primer eje y
color = 'black'
ax1.set_xlabel('Fecha')
ax1.set_ylabel('SP500', color=color)
ax1.plot(fechas_sp500, cierre_ajustado_sp500, color=color, label='SP500')
ax1.plot(fechas_prediccion, predicciones_sp500, color='green', linestyle='--', label='Predicciones SP500')
ax1.tick_params(axis='y', labelcolor=color)

# Crear un segundo eje y para el VIX
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('VIX', color=color)
ax2.plot(fechas_vix, cierre_ajustado_vix, color=color, label='VIX')
ax2.plot(fechas_prediccion, predicciones_vix, color='blue', linestyle='--', label='Predicciones VIX')
ax2.tick_params(axis='y', labelcolor=color)

# Configuración del gráfico
fig.suptitle('Evolución de SP500 y VIX (últimos 10 valores y predicciones)')
fig.tight_layout()
fig.legend(loc='upper left')
plt.xlabel('Fecha')
plt.grid(True)

# Mostrar el gráfico
plt.show()





#############################################
#############################################
#############################################
#############################################
#VEC vix-us500, prediccion a 10 periodos
#############################################
#############################################
#############################################
#############################################

import pandas as pd
from statsmodels.tsa.api import VECM 
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import select_order

# Cargar los DataFrames desde los archivos
ruta_SP500 = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\Prediccion en economia SIMULACION PORTFOLIO (VIX)\sp500.xlsx'
ruta_vix = r"C:\Users\34653\Desktop\UNED\segundo cuatrimestre\Prediccion en economia SIMULACION PORTFOLIO (VIX)\vix.xlsx"

sp500 = pd.read_excel(ruta_SP500, index_col=0)
vix = pd.read_excel(ruta_vix, index_col=0)

# Asegurarse de que los DataFrames estén en el mismo orden
sp500 = sp500.sort_index(ascending=True)
vix = vix.sort_index(ascending=True)

# Unir los DataFrames en uno solo
df = pd.concat([sp500, vix], axis=1)

# Verificar tipos de datos
print(df.dtypes)

# Convertir tipos de datos si es necesario
df = df.apply(pd.to_numeric, errors='coerce')

# Eliminar filas con NaNs si es necesario
df = df.dropna()

# Utilizar select_order para encontrar el orden óptimo
model_order = select_order(df, maxlags=10)

# Crear y ajustar el modelo VEC con el orden óptimo
model = VECM(df, k_ar_diff=model_order.aic)
results = model.fit()

# Ver resumen del modelo
print(results.summary())


# Extraer los residuos del modelo VEC
residuos = results.resid


# Obtener el número de pasos hacia adelante que deseamos predecir
forecast_steps = 10  # Por ejemplo, predeciremos los próximos 10 períodos

# Realizar las predicciones futuras
predicciones = results.predict(steps=forecast_steps)

# Crear un DataFrame para las predicciones
pred_df = pd.DataFrame(predicciones, index=pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq=df.index.freq)[1:], columns=df.columns)

# Imprimir las predicciones
print(pred_df)

# Graficar los últimos 10 valores y predicciones del cierre ajustado del S&P 500
plt.figure(figsize=(12, 8))

# Graficar los últimos 10 valores
sns.lineplot(x=sp500.index[-10:], y=sp500['Cierre ajus SP500'].tail(10), marker='o', color='red', label='Últimos 10 valores observados S&P500')

# Graficar las predicciones
sns.lineplot(data=pred_df['Cierre ajus SP500'], marker='o', label='Predicciones VEC S&P500')

# Ajustar las etiquetas de las fechas en el eje x
plt.xticks(rotation=45)  # Rotar las etiquetas 45 grados

# Configurar título y etiquetas de ejes
plt.title('Últimos 10 valores y predicciones del cierre ajustado del S&P 500 con modelo VEC')
plt.xlabel('Fecha')
plt.ylabel('Cierre ajustado')

# Mostrar la leyenda
plt.legend()

# Mostrar la gráfica
plt.grid(True)
plt.show()



################################# AMBAS PREDICCIONES



# Crear figura y ejes
fig, ax1 = plt.subplots(figsize=(12, 8))

# Graficar los últimos 10 valores de S&P 500
sns.lineplot(x=sp500.index[-10:], y=sp500['Cierre ajus SP500'].tail(10), marker='o', color='red', label='Últimos 10 valores observados S&P500', ax=ax1)

# Configurar el primer eje y (izquierda) para S&P 500
ax1.set_ylabel('Cierre ajustado S&P500', color='red')

# Ajustar las etiquetas de las fechas en el eje x
plt.xticks(rotation=45)  # Rotar las etiquetas 45 grados

# Crear segundo eje y (derecha) para VIX
ax2 = ax1.twinx()

# Graficar los últimos 10 valores de VIX
sns.lineplot(x=vix.index[-10:], y=vix['Cierre ajus VIX'].tail(10), marker='o', color='blue', label='Últimos 10 valores observados VIX', ax=ax2)

# Configurar el segundo eje y (derecha) para VIX
ax2.set_ylabel('Cierre ajustado VIX', color='blue')

# Graficar las predicciones de S&P 500
sns.lineplot(data=pred_df['Cierre ajus SP500'], marker='o', label='Predicciones VEC S&P500', ax=ax1, linestyle='dotted', color='green')

# Graficar las predicciones de VIX
sns.lineplot(data=pred_df['Cierre ajus VIX'], marker='o', label='Predicciones VEC VIX', ax=ax2, linestyle='dotted', color='orange')

# Configurar título y etiquetas de ejes
plt.title('Últimos 10 valores y predicciones del cierre ajustado del S&P 500 con modelo VEC')
plt.xlabel('Fecha')

# Mostrar la leyenda
ax1.legend(loc='upper left')
ax2.legend(loc='lower right')

# Mostrar la gráfica
plt.grid(True)
plt.show()















#############################################
#############################################
#############################################
#############################################
#ANÁLISIS DE LA RELACIÓN VIX - VOLATILIDAD DATOS SCRAPEADOS YAHOO FINANCE
#############################################
#############################################
#############################################
#############################################


############################ SCRAPEAR DATOS DE YAHOO FINANCE


sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data_table = pd.read_html(sp500url)

ticker = data_table[0]['Symbol'].tolist()

snp_prices = yf.download(ticker, start='2000-01-01', end='2024-01-01')['Adj Close']

############################ REPRESENTAR UN ACTIVO CONCRETO

# Seleccionar los precios de cierre ajustados para 'MMM'
mmm_prices = snp_prices['NEM']
    
# Graficar la evolución del precio de cierre ajustado para 'MMM'
plt.figure(figsize=(10, 6))
mmm_prices.plot(label='AMP', color="orange")
plt.title('Evolución del precio de cierre ajustado para NEM')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre ajustado')
plt.legend()
plt.grid(True)
plt.show()


####################################### VIX YFINANCE

# Descargar los datos del VIX
vix_data = yf.download('^VIX', start='2000-01-01', end='2024-01-01')

# Seleccionar solo la columna 'Adj Close' (precios de cierre ajustados)
vix_prices = vix_data['Adj Close']

# Convertir los datos de fecha a una matriz de NumPy
dates = np.array(vix_data.index)

# Convertir los precios de cierre ajustados del VIX a una matriz de NumPy
vix_prices_array = np.array(vix_data['Adj Close'])

# Graficar los precios del VIX
plt.figure(figsize=(10, 6))
plt.plot(dates, vix_prices_array, color='blue')
plt.title('Precios de cierre ajustados del VIX (2020-2024)')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre ajustado')
plt.grid(True)
plt.show()


################################ CORRELACIONES RETORNOS ACTIVOS-VIX

# Calcular los rendimientos diarios de los activos del S&P 500
snp_returns = snp_prices.pct_change()

# Calcular los rendimientos diarios del VIX
vix_returns = vix_prices.pct_change()

# Calcular la correlación entre los rendimientos diarios de cada activo individual y los rendimientos diarios del VIX
correlations = snp_returns.corrwith(vix_returns)

# Imprimir las correlaciones
print(correlations)


######################################### RELACION DE LOS RENDIMIENTOS CON EL VIX TOP20

# Obtener las 10 mayores correlaciones
top_correlations = correlations.abs().nlargest(10)

# Obtener las 10 menores correlaciones
bottom_correlations = correlations.abs().nsmallest(10)

# Combinar los activos de interés
assets_of_interest = top_correlations.index.tolist() + bottom_correlations.index.tolist()


# Calcular la correlación entre los rendimientos diarios de los activos seleccionados y los rendimientos diarios del VIX
correlations_selection = snp_returns[assets_of_interest].corrwith(vix_returns)

# Graficar las correlaciones
plt.figure(figsize=(10, 6))
correlations_selection.plot(kind='bar')
plt.title('Correlación entre rendimientos diarios de activos seleccionados y el VIX')
plt.xlabel('Activo')
plt.ylabel('Correlación')
plt.grid(True)
plt.show()










# Seleccionar los precios de cierre ajustados
assets_prices = snp_prices[assets_of_interest]
    
# Graficar la evolución del precio de cierre ajustado para 'assets_of_interest'
plt.figure(figsize=(10, 6))
assets_prices.plot(label='assets_of_interest')
plt.title('Evolución del precio de cierre ajustado - Activos con mayor y menor correlación')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre ajustado')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=5) 
plt.grid(True)
plt.show()








# RELACIÓN DE LA VOLATILIDAD CONDICIONAL ARCH DE UN ACTIVO CONCRETO CON EL VIX
# Seleccionar un activo específico para modelar su volatilidad (por ejemplo, 'AMP')

# Calcular los retornos
asset = 'NEM'

# Obtener los retornos diarios del activo seleccionado
asset_returns = snp_prices[asset].pct_change().dropna()


# Ajustar un modelo ARCH(1)
model = arch_model(asset_returns, vol='Garch', p=1, o=0, q=0, dist='Normal')
results_asset = model.fit()

# Resumen del modelo
print(results_asset.summary())

# Extraer la volatilidad condicional estimada
cond_volatility_asset = results_asset.conditional_volatility

# Graficar la volatilidad condicional estimada
plt.figure(figsize=(10, 6))
cond_volatility_asset.plot()
plt.title('Volatilidad Condicional Estimada del activo seleccionado (NEM) - Modelo ARCH(1)')
plt.xlabel('Fecha')
plt.ylabel('Volatilidad')
plt.grid(True)
plt.show()



# Ajustar un modelo GARCH al VIX
vix_returns = vix_prices.pct_change().dropna()

vix_model = arch_model(vix_returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
results_vix = vix_model.fit()

# Extraer la volatilidad condicional estimada del VIX
cond_volatility_vix = results_vix.conditional_volatility

# Ajustar un modelo ARCH al activo seleccionado
asset_returns = snp_prices[asset].pct_change().dropna()

model_asset = arch_model(asset_returns, vol='Garch', p=1, o=0, q=0, dist='Normal')
results_asset = model_asset.fit()

# Extraer la volatilidad condicional estimada del activo seleccionado
cond_volatility_asset = results_asset.conditional_volatility

# Fusionar los datos de volatilidad condicional del activo y del VIX en un DataFrame único
merged_data = pd.concat([cond_volatility_asset, cond_volatility_vix], axis=1).dropna()
merged_data.columns = ['Conditional Volatility Asset', 'Conditional Volatility VIX']

# Calcular la correlación entre la volatilidad condicional del activo y la del VIX
correlation = np.corrcoef(merged_data['Conditional Volatility Asset'], merged_data['Conditional Volatility VIX'])[0, 1]

print("Correlación entre la volatilidad condicional de {} y la volatilidad del VIX: {}".format(asset, correlation))



# Graficar la volatilidad condicional estimada
plt.figure(figsize=(10, 6))
cond_volatility_vix.plot()
plt.title('Volatilidad Condicional Estimada del VIX - Modelo ARCH(1)')
plt.xlabel('Fecha')
plt.ylabel('Volatilidad')
plt.grid(True)
plt.show()








# Graficar la volatilidad condicional estimada del activo y del VIX
plt.figure(figsize=(10, 6))
merged_data.plot()
plt.title('Volatilidad Condicional Estimada del activo y del VIX')
plt.xlabel('Fecha')
plt.ylabel('Volatilidad')
plt.grid(True)
plt.show()











# Seleccionar los precios de cierre ajustados
assets_prices = snp_prices[assets_of_interest]
    
# Graficar la evolución del precio de cierre ajustado para 'assets_of_interest'
plt.figure(figsize=(10, 6))
assets_prices.plot(label='assets_of_interest')
plt.title('Evolución del precio de cierre ajustado - Activos con mayor y menor correlación')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre ajustado')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=5) 
plt.grid(True)
plt.show()






# Seleccionar los precios de cierre ajustados para 'MMM'
mmm_prices = snp_prices[['AMP', "NEM"]]
    
# Graficar la evolución del precio de cierre ajustado para 'MMM'
plt.figure(figsize=(10, 6))
mmm_prices.plot(label='AMP/NEM')
plt.title('Evolución del precio de cierre ajustado para AMP/NEM')
plt.xlabel('Fecha')
plt.ylabel('Precio de cierre ajustado')
plt.legend()
plt.grid(True)
plt.show()











# Fusionar los datos de volatilidad condicional y los valores del VIX en un DataFrame único
#merged_data = pd.concat([results_asset.conditional_volatility, vix_prices], axis=1).dropna()
#merged_data.columns = ['Conditional Volatility Asset', 'VIX Values']

# Calcular la correlación entre la volatilidad condicional y los valores del VIX
#correlation = np.corrcoef(merged_data['Conditional Volatility Asset'], merged_data['VIX Values'])[0, 1]

#print("Correlación entre la volatilidad condicional de {} y el VIX: {}".format(asset, correlation))







#############################################
#############################################
#############################################
#############################################
#SIMULACIÓN MONTECARLO DE ESCENARIOS DE VOLATILIDAD
#############################################
#############################################
#############################################
#############################################

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Descargar los datos del VIX
vix_data = yf.download('^VIX', start='2000-01-01', end='2024-01-01')

# Seleccionar solo la columna 'Adj Close' (precios de cierre ajustados)
vix_prices = vix_data['Adj Close']

# Obtener el último valor disponible de los datos históricos
ultimo_valor = vix_prices.iloc[-1]

# Calcular la media y la desviación estándar de los datos históricos del VIX
media_vix = 0
desviacion_vix = np.std(vix_prices)

# Definir el número de periodos de simulación y el número de simulaciones de Montecarlo
num_periodos = 10
num_simulaciones = 120

# Simular escenarios de volatilidad utilizando Montecarlo
resultados_simulacion = []

for _ in range(num_simulaciones):
    volatilidad_simulada = np.random.normal(media_vix, desviacion_vix, num_periodos)
    # Ajustar los valores para que todos comiencen desde el último valor disponible
    volatilidad_simulada[0] = ultimo_valor
    volatilidad_simulada = np.cumsum(volatilidad_simulada)
    resultados_simulacion.append(volatilidad_simulada)

# Convertir resultados a un array de NumPy
resultados_simulacion = np.array(resultados_simulacion)

# Calcular la media de todas las simulaciones
simulacion_media = np.mean(resultados_simulacion, axis=0)

# Visualizar los resultados de la simulación
plt.figure(figsize=(10, 6))
for i in range(num_simulaciones):
    plt.plot(range(num_periodos), resultados_simulacion[i], alpha=0.5)

# Plotear la media de las simulaciones en rojo
plt.plot(range(num_periodos), simulacion_media, color='red', label='Media de simulaciones')

# Marcar el valor del punto inicial
plt.axhline(y=ultimo_valor, color='red', linestyle='--', label='Valor inicial')
plt.text(-0.5, ultimo_valor, f' {ultimo_valor:.2f}', color='red', verticalalignment='top')

plt.title('Simulación de Escenarios de Volatilidad (VIX)')
plt.xlabel('Periodo')
plt.ylabel('Volatilidad')
plt.legend()
plt.grid(True)
plt.show()


































#############################################
#############################################
#############################################
#############################################
# REPRESENTACIÓN ALEATORIA DE PORTFOLIOS
#############################################
#############################################
#############################################
#############################################


import pandas as pd
import yfinance as yf
import random
import seaborn as sns
import matplotlib.pyplot as plt

# Obtener los tickers del S&P 500 desde Wikipedia
sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data_table = pd.read_html(sp500url)
ticker = data_table[0]['Symbol'].tolist()

# Descargar los precios ajustados de cierre del S&P 500 desde Yahoo Finance
snp_prices = yf.download(ticker, start='2000-01-01', end='2024-01-01')['Adj Close']

# Seleccionar una muestra aleatoria de 10 activos del S&P 500
random.seed(42)
portfolio_tickers = random.sample(ticker, 10)

# Seleccionar los precios ajustados de cierre solo para los activos seleccionados
portfolio_prices = snp_prices[portfolio_tickers]

# Calcular los rendimientos diarios de los activos seleccionados
portfolio_returns = portfolio_prices.pct_change()

# Calcular el rendimiento del portafolio sumando los rendimientos ponderados de cada activo
weights = [1 / len(portfolio_tickers)] * len(portfolio_tickers)
portfolio_return = (portfolio_returns * weights).sum(axis=1).dropna()

# Calcular el rendimiento diario del índice S&P 500
snp_returns = snp_prices.mean(axis=1).pct_change().dropna()

# Configurar Seaborn para que los gráficos tengan un aspecto más agradable
sns.set(style="darkgrid")

# Graficar los rendimientos del portafolio y del S&P 500
plt.figure(figsize=(10, 6))
sns.lineplot(data=portfolio_return.cumsum(), label='Portfolio')
sns.lineplot(data=snp_returns.cumsum(), label='S&P 500')
plt.title('Rendimiento del Portafolio vs S&P 500')
plt.xlabel('Fecha')
plt.ylabel('Rendimiento Acumulado')
plt.legend()
plt.show()






#############################################
#############################################
#############################################
#############################################
# 
#############################################
#############################################
#############################################
#############################################

import pandas as pd
import yfinance as yf
import random
import seaborn as sns
import matplotlib.pyplot as plt

# Obtener los tickers del S&P 500 desde Wikipedia
sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data_table = pd.read_html(sp500url)
ticker = data_table[0]['Symbol'].tolist()

# Descargar los precios ajustados de cierre del S&P 500 desde Yahoo Finance
snp_prices = yf.download(ticker, start='2000-01-01', end='2024-01-01')['Adj Close']

# Seleccionar una muestra aleatoria de 10 activos del S&P 500
random.seed(42)
portfolio_tickers = random.sample(ticker, 10)

# Seleccionar los precios ajustados de cierre solo para los activos seleccionados
portfolio_prices = snp_prices[portfolio_tickers]

# Calcular los rendimientos diarios de los activos seleccionados
portfolio_returns = portfolio_prices.pct_change()

# Calcular el rendimiento del portafolio sumando los rendimientos ponderados de cada activo
weights = [1 / len(portfolio_tickers)] * len(portfolio_tickers)
portfolio_return = (portfolio_returns * weights).sum(axis=1).dropna()

# Calcular el rendimiento diario del índice S&P 500
snp_returns = snp_prices.pct_change().dropna()

# Descargar los precios ajustados de cierre del índice S&P 500 (GCSP) desde Yahoo Finance
gcsp_prices = yf.download('^GSPC', start='2000-01-01', end='2024-01-01')['Adj Close']
gcsp_returns = gcsp_prices.pct_change().dropna()

# Configurar Seaborn para que los gráficos tengan un aspecto más agradable
sns.set(style="darkgrid")

# Graficar los rendimientos del portafolio, del S&P 500 y del GCSP en el mismo gráfico
plt.figure(figsize=(10, 6))
sns.lineplot(data=portfolio_return.cumsum(), label='Portfolio')
sns.lineplot(data=snp_returns.cumsum(), label='S&P 500')
#sns.lineplot(data=gcsp_prices.pct_change().cumsum(), label='GCSP')
plt.title('Rendimiento del Portafolio vs S&P 500 vs GCSP')
plt.xlabel('Fecha')
plt.ylabel('Rendimiento Acumulado')
plt.legend()
plt.show()

# Graficar los rendimientos del portafolio, del S&P 500 y del GCSP en el mismo gráfico
plt.figure(figsize=(10, 6))
sns.lineplot(data=portfolio_return.cumsum(), label='Portfolio')
sns.lineplot(data=gcsp_returns.cumsum(), label='GCSP')
plt.title('GCSP')
plt.xlabel('Fecha')
plt.ylabel('Rendimiento Acumulado')
plt.legend()
plt.show()







