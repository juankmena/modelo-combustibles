import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Modelo combustibles EIA",
    page_icon="⛽",
    layout="wide"
)

st.title("⛽ Modelo de predicción de precios de combustibles")
st.caption("Datos reales EIA | Clasificación supervisada | Predicción del siguiente periodo")

URL_WTI = "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"
URL_GASOLINA = "https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_NUS_DPGw.xls"
URL_DIESEL = "https://www.eia.gov/dnav/pet/hist_xls/EMD_EPD2D_PTE_NUS_DPGw.xls"
HISTORICO_PATH = "historico_predicciones_combustibles.csv"


def leer_archivo_eia(url: str, nombre_variable: str) -> pd.DataFrame:
    """Lee archivos históricos XLS de EIA desde la hoja Data 1."""
    df_raw = pd.read_excel(url, sheet_name="Data 1", skiprows=2)
    df_raw = df_raw.dropna(how="all")

    fecha_col = df_raw.columns[0]
    valor_col = df_raw.columns[1]

    df = df_raw[[fecha_col, valor_col]].copy()
    df.columns = ["fecha", nombre_variable]

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df[nombre_variable] = pd.to_numeric(df[nombre_variable], errors="coerce")

    df = df.dropna(subset=["fecha", nombre_variable])
    df = df.sort_values("fecha").reset_index(drop=True)

    return df

@st.cache_data(show_spinner=False)
def cargar_datos() -> pd.DataFrame:
    wti_diario = leer_archivo_eia(URL_WTI, "wti")
    gasolina = leer_archivo_eia(URL_GASOLINA, "gasolina_regular")
    diesel = leer_archivo_eia(URL_DIESEL, "diesel")

    gasolina = gasolina.sort_values("fecha")
    diesel = diesel.sort_values("fecha")
    wti_diario = wti_diario.sort_values("fecha")

    df = pd.merge_asof(
        gasolina,
        diesel,
        on="fecha",
        direction="nearest",
        tolerance=pd.Timedelta(days=4)
    )

    df = pd.merge_asof(
        df.sort_values("fecha"),
        wti_diario[["fecha", "wti"]].sort_values("fecha"),
        on="fecha",
        direction="backward"
    )

    df = df.dropna(subset=["gasolina_regular", "diesel", "wti"])
    df = df.sort_values("fecha").reset_index(drop=True)

    return df


def preparar_modelo(df: pd.DataFrame):
    datos = df.copy().sort_values("fecha").reset_index(drop=True)

    # Variables explicativas derivadas del comportamiento reciente del mercado.
    datos["gasolina_lag1"] = datos["gasolina_regular"].shift(1)
    datos["gasolina_lag2"] = datos["gasolina_regular"].shift(2)
    datos["diesel_lag1"] = datos["diesel"].shift(1)
    datos["wti_lag1"] = datos["wti"].shift(1)
    datos["var_gasolina"] = datos["gasolina_regular"].pct_change()
    datos["var_wti"] = datos["wti"].pct_change()
    datos["media_gasolina_4s"] = datos["gasolina_regular"].rolling(4).mean()
    datos["volatilidad_gasolina_4s"] = datos["gasolina_regular"].rolling(4).std()
    datos["mes"] = datos["fecha"].dt.month

    # Variable objetivo: 1 si la gasolina sube en la siguiente observación.
    datos["sube_gasolina"] = (
        datos["gasolina_regular"].shift(-1) > datos["gasolina_regular"]
    ).astype(int)

    # La última fila no tiene resultado real observable del siguiente periodo.
    datos_modelo = datos.iloc[:-1].dropna().copy()
    datos_prediccion = datos.dropna().copy()

    if len(datos_modelo) < 30:
        raise ValueError(
            f"Después de preparar los datos quedaron solo {len(datos_modelo)} registros. "
            "Revise la descarga de EIA y la unión de fechas."
        )

    columnas_x = [
        "gasolina_regular",
        "diesel",
        "wti",
        "gasolina_lag1",
        "gasolina_lag2",
        "diesel_lag1",
        "wti_lag1",
        "var_gasolina",
        "var_wti",
        "media_gasolina_4s",
        "volatilidad_gasolina_4s",
        "mes",
    ]

    X = datos_modelo[columnas_x]
    y = datos_modelo["sube_gasolina"]

    train_size = int(len(datos_modelo) * 0.8)

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    fechas_test = datos_modelo.iloc[train_size:]["fecha"]

    param_grid = {
        "max_depth": [2, 3, 4, 5, 6],
        "min_samples_leaf": [1, 3, 5, 10],
    }

    base_model = DecisionTreeClassifier(random_state=42)

    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    modelo = grid.best_estimator_

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5)

    # Predicción del último periodo disponible.
    ultima_fila = datos_prediccion.iloc[[-1]].copy()
    X_ultimo = ultima_fila[columnas_x]

    pred_final = int(modelo.predict(X_ultimo)[0])
    prob_final = float(modelo.predict_proba(X_ultimo)[0][1])

    frecuencia = datos["fecha"].diff().mode()[0]
    fecha_base = ultima_fila["fecha"].iloc[0]
    periodo_predicho = fecha_base + frecuencia

    resultados = pd.DataFrame({
        "fecha": fechas_test.values,
        "real": y_test.values,
        "prediccion": y_pred,
        "probabilidad_sube": y_proba,
    })

    resultados["acierto"] = resultados["real"] == resultados["prediccion"]

    return {
        "datos": datos,
        "datos_modelo": datos_modelo,
        "modelo": modelo,
        "mejores_parametros": grid.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "matriz": confusion_matrix(y_test, y_pred),
        "reporte": classification_report(
            y_test,
            y_pred,
            target_names=["No sube", "Sube"],
            output_dict=True
        ),
        "cv_scores": cv_scores,
        "resultados": resultados,
        "fecha_base": fecha_base,
        "periodo_predicho": periodo_predicho,
        "pred_final": pred_final,
        "prob_final": prob_final,
        "frecuencia": frecuencia,
    }


def guardar_historico(fecha_base, periodo_predicho, prediccion, probabilidad):
    nueva = pd.DataFrame({
        "fecha_ejecucion": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "fecha_base_informacion": [fecha_base],
        "periodo_predicho": [periodo_predicho],
        "prediccion": ["Sube" if prediccion == 1 else "No sube"],
        "probabilidad_sube": [probabilidad]
    })

    if os.path.exists(HISTORICO_PATH):
        historico = pd.read_csv(HISTORICO_PATH)
        historico = pd.concat([historico, nueva], ignore_index=True)
    else:
        historico = nueva

    historico = historico.drop_duplicates(
        subset=["fecha_base_informacion", "periodo_predicho"],
        keep="last"
    )
    historico.to_csv(HISTORICO_PATH, index=False)
    return historico


with st.sidebar:
    st.header("Opciones")
    st.write("Presioná el botón para descargar datos EIA, entrenar el modelo y generar la predicción.")
    ejecutar = st.button("Actualizar datos y entrenar modelo", type="primary")
    st.divider()
    st.caption("Fuente: archivos históricos públicos de EIA.")

if ejecutar:
    with st.spinner("Descargando datos EIA y entrenando modelo..."):
        df = cargar_datos()
        salida = preparar_modelo(df)
        historico = guardar_historico(
            salida["fecha_base"],
            salida["periodo_predicho"],
            salida["pred_final"],
            salida["prob_final"]
        )

    pred_texto = "Sube" if salida["pred_final"] == 1 else "No sube"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Última fecha EIA", str(salida["fecha_base"].date()))
    col2.metric("Periodo predicho", str(salida["periodo_predicho"].date()))
    col3.metric("Predicción", pred_texto)
    col4.metric("Probabilidad de subida", f"{salida['prob_final']:.2%}")

    st.subheader("Evaluación del modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy test temporal", f"{salida['accuracy']:.2%}")
    c2.metric("Promedio validación cruzada", f"{salida['cv_scores'].mean():.2%}")
    c3.metric("Frecuencia detectada", str(salida["frecuencia"]))

    st.write("Mejores hiperparámetros:", salida["mejores_parametros"])

    st.subheader("Matriz de confusión")
    matriz_df = pd.DataFrame(
        salida["matriz"],
        index=["Real: No sube", "Real: Sube"],
        columns=["Pred: No sube", "Pred: Sube"]
    )
    st.dataframe(matriz_df, use_container_width=True)

    st.subheader("Resultados del periodo de prueba")
    st.dataframe(salida["resultados"].sort_values("fecha", ascending=False), use_container_width=True)

    fig = px.line(
        salida["resultados"],
        x="fecha",
        y="probabilidad_sube",
        title="Probabilidad estimada de subida en el periodo de prueba"
    )
    fig.add_hline(y=0.5, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Histórico de predicciones")
    st.dataframe(historico.sort_values("fecha_ejecucion", ascending=False), use_container_width=True)

    st.download_button(
        "Descargar histórico CSV",
        historico.to_csv(index=False),
        file_name="historico_predicciones_combustibles.csv",
        mime="text/csv"
    )

    st.subheader("Últimos datos descargados")
    st.dataframe(df.tail(15).sort_values("fecha", ascending=False), use_container_width=True)
else:
    st.info("Presioná el botón de la izquierda para ejecutar el modelo.")

st.markdown("---")
st.markdown("""
### Glosario técnico

- **MAE (Mean Absolute Error):** Promedio absoluto del error entre el valor estimado y el valor real.
- **MAPE (Mean Absolute Percentage Error):** Error porcentual promedio del modelo.
- **Accuracy:** Porcentaje de aciertos del modelo supervisado.
- **Proxy EIA:** Aproximación internacional basada en datos públicos de EIA.
""")

st.info("""
Interpretación técnica:

El comportamiento histórico evidencia una relación observable entre el proxy internacional EIA y las compras reales de importación de Recope.

Aunque las compras reales presentan menor volatilidad y ciertos rezagos temporales, la tendencia general muestra una asociación importante con el comportamiento internacional de los combustibles refinados.
""")

st.markdown("""
---
<div style='text-align:center;color:gray;font-size:14px'>
Observatorio experimental del mercado internacional de combustibles.<br>
Datos públicos EIA y contraste con compras reales de importación.<br><br>
<b>Hecho por: Lic. Juan Carlos Mena</b>
</div>
""", unsafe_allow_html=True)
