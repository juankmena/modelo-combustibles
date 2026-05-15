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
st.caption("Datos reales EIA | Clasificación supervisada | Predicción del siguiente periodo | Contraste con compras Recope")

URL_WTI = "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"
URL_GASOLINA = "https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_NUS_DPGw.xls"
URL_DIESEL = "https://www.eia.gov/dnav/pet/hist_xls/EMD_EPD2D_PTE_NUS_DPGw.xls"
HISTORICO_PATH = "historico_predicciones_combustibles.csv"
RECOPE_PATH = "2016-2026_abril.csv"


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
        "gasolina_regular", "diesel", "wti", "gasolina_lag1", "gasolina_lag2",
        "diesel_lag1", "wti_lag1", "var_gasolina", "var_wti", "media_gasolina_4s",
        "volatilidad_gasolina_4s", "mes",
    ]

    X = datos_modelo[columnas_x]
    y = datos_modelo["sube_gasolina"]

    train_size = int(len(datos_modelo) * 0.8)

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    fechas_test = datos_modelo.iloc[train_size:]["fecha"]

    param_grid = {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [1, 3, 5, 10]}
    base_model = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(base_model, param_grid, cv=5, scoring="accuracy")
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
        "reporte": classification_report(y_test, y_pred, target_names=["No sube", "Sube"], output_dict=True),
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

    historico = historico.drop_duplicates(subset=["fecha_base_informacion", "periodo_predicho"], keep="last")
    historico.to_csv(HISTORICO_PATH, index=False)
    return historico


def _limpiar_numero_esp(x):
    """Convierte números con coma decimal y/o separadores extraños a float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("\xa0", "").replace(" ", "")
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    # Si hay coma y punto, asumimos punto como miles y coma como decimal.
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    return pd.to_numeric(s, errors="coerce")


def normalizar_producto(x):
    x = str(x).upper().strip()
    if "GAS" in x and ("91" in x or "REG" in x):
        return "gasolina_regular"
    if "GAS" in x and ("95" in x or "SUPER" in x or "SÚPER" in x):
        return "gasolina_super"
    if "DIESEL" in x or "DIÉSEL" in x:
        return "diesel"
    if "JET" in x or "AV TUR" in x:
        return "jet"
    if "LPG" in x or "GLP" in x or "PROPANO" in x or "BUTANO" in x:
        return "glp"
    if "ASFALTO" in x:
        return "asfalto"
    if "AV-GAS" in x or "AVGAS" in x:
        return "avgas"
    if "BUNKER" in x:
        return "bunker"
    return "otro"


@st.cache_data(show_spinner=False)
def cargar_compras_recope(path_o_buffer=None) -> pd.DataFrame:
    """Carga y prepara las compras Recope 2016-2026."""
    if path_o_buffer is None:
        if not os.path.exists(RECOPE_PATH):
            return pd.DataFrame()
        path_o_buffer = RECOPE_PATH

    # Los archivos históricos del usuario vienen normalmente en Latin-1 y separados por punto y coma.
    try:
        df = pd.read_csv(path_o_buffer, sep=";", encoding="latin1")
    except Exception:
        df = pd.read_csv(path_o_buffer, sep=",", encoding="utf-8")

    # Normaliza nombres de columnas para evitar problemas con tildes, espacios y símbolos.
    renombrar = {
        "AÑO": "anio",
        "MES": "mes",
        "FECHA_BL": "fecha_bl",
        "PRODUCTO": "producto",
        "BARRILES": "barriles",
        "COSTO_FOB": "costo_fob",
        "COSTO_FLETE": "costo_flete",
        "COSTO_SEGURO": "costo_seguro",
        "COSTO_CIF": "costo_cif",
        "COSTO_CFR": "costo_cfr",
        "PROVEEDOR": "proveedor",
        "NÚMERO_EMBARQUE": "numero_embarque",
        "NÚMERO_CONTRATO": "numero_contrato",
    }
    df = df.rename(columns={c: renombrar.get(c, c) for c in df.columns})

    for col in ["barriles", "costo_fob", "costo_flete", "costo_seguro", "costo_cif", "costo_cfr"]:
        if col in df.columns:
            df[col] = df[col].apply(_limpiar_numero_esp)

    df["fecha_bl"] = pd.to_datetime(df.get("fecha_bl"), errors="coerce", dayfirst=True)
    df["fecha_operacion"] = df["fecha_bl"]
    df["producto_normalizado"] = df.get("producto", "").apply(normalizar_producto)

    # Precios unitarios por barril. Se calculan desde los montos totales para mantener consistencia.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["precio_fob_bbl"] = df["costo_fob"] / df["barriles"]
        df["precio_cif_bbl"] = df["costo_cif"] / df["barriles"]
        df["precio_cfr_bbl"] = df["costo_cfr"] / df["barriles"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["fecha_operacion", "producto_normalizado", "barriles", "precio_cfr_bbl"])
    df = df[df["barriles"] > 0]

    df["anio"] = df["fecha_operacion"].dt.year
    df["mes"] = df["fecha_operacion"].dt.month
    df["semana_iso"] = df["fecha_operacion"].dt.isocalendar().week.astype(int)

    return df.sort_values("fecha_operacion").reset_index(drop=True)


def agregar_compras_periodo(df_recope: pd.DataFrame, producto: str, frecuencia="W-MON") -> pd.DataFrame:
    """Agrupa compras por producto y periodo con promedio ponderado por barriles."""
    base = df_recope[df_recope["producto_normalizado"] == producto].copy()
    if base.empty:
        return pd.DataFrame()

    base["periodo"] = base["fecha_operacion"].dt.to_period(frecuencia).dt.start_time

    def wavg(g, col):
        return np.average(g[col], weights=g["barriles"])

    out = base.groupby("periodo", as_index=False).apply(
        lambda g: pd.Series({
            "barriles": g["barriles"].sum(),
            "precio_real_fob_bbl": wavg(g, "precio_fob_bbl"),
            "precio_real_cif_bbl": wavg(g, "precio_cif_bbl"),
            "precio_real_cfr_bbl": wavg(g, "precio_cfr_bbl"),
            "compras": len(g),
        })
    ).reset_index(drop=True)
    out = out.rename(columns={"periodo": "fecha"})
    return out.sort_values("fecha")


def serie_eia_para_producto(df_eia: pd.DataFrame, producto: str) -> tuple[pd.DataFrame, str]:
    """Devuelve una serie EIA en USD/bbl para comparar como proxy de mercado."""
    eia = df_eia[["fecha", "gasolina_regular", "diesel", "wti"]].copy().sort_values("fecha")
    if producto in ["gasolina_regular", "gasolina_super"]:
        eia["precio_eia_bbl"] = eia["gasolina_regular"] * 42
        etiqueta = "Gasolina regular EIA × 42"
    elif producto == "diesel":
        eia["precio_eia_bbl"] = eia["diesel"] * 42
        etiqueta = "Diésel EIA × 42"
    else:
        eia["precio_eia_bbl"] = eia["wti"]
        etiqueta = "WTI EIA como proxy"
    return eia[["fecha", "precio_eia_bbl"]].dropna(), etiqueta


def construir_backtesting_recope(df_eia: pd.DataFrame, df_recope: pd.DataFrame, producto: str, objetivo: str):
    compras = agregar_compras_periodo(df_recope, producto)
    if compras.empty:
        return pd.DataFrame(), "Sin compras para el producto seleccionado"

    eia, etiqueta = serie_eia_para_producto(df_eia, producto)
    comp = pd.merge_asof(
        compras.sort_values("fecha"),
        eia.sort_values("fecha"),
        on="fecha",
        direction="nearest",
        tolerance=pd.Timedelta(days=10)
    ).dropna(subset=["precio_eia_bbl"])

    if comp.empty:
        return comp, etiqueta

    comp["precio_real_bbl"] = comp[objetivo]
    comp["diferencia_usd_bbl"] = comp["precio_real_bbl"] - comp["precio_eia_bbl"]
    comp["error_abs_usd_bbl"] = comp["diferencia_usd_bbl"].abs()
    comp["error_pct"] = np.where(comp["precio_real_bbl"] != 0, comp["error_abs_usd_bbl"] / comp["precio_real_bbl"] * 100, np.nan)
    comp["cercania_pct"] = (100 - comp["error_pct"]).clip(lower=0)
    comp["sesgo"] = np.where(comp["diferencia_usd_bbl"] > 0, "Recope > EIA", "Recope <= EIA")
    return comp.sort_values("fecha"), etiqueta


def evaluar_cercania(cercania):
    if pd.isna(cercania):
        return "Sin dato"
    if cercania >= 95:
        return "Muy cercana"
    if cercania >= 90:
        return "Aceptable"
    if cercania >= 80:
        return "Moderada"
    return "Lejana"


with st.sidebar:
    st.header("Opciones")
    st.write("Presioná el botón para descargar datos EIA, entrenar el modelo y generar la predicción.")
    ejecutar = st.button("Actualizar datos y entrenar modelo", type="primary")
    st.divider()
    st.subheader("Compras Recope")
    archivo_recope = st.file_uploader("CSV compras Recope", type=["csv"], help="Opcional. Si no se carga archivo, la app intenta usar 2016-2026_abril.csv en la carpeta del proyecto.")
    producto_recope = st.selectbox(
        "Producto para contraste",
        ["gasolina_regular", "gasolina_super", "diesel", "jet", "glp", "asfalto", "avgas", "bunker"],
        index=2,
    )
    objetivo_recope = st.selectbox(
        "Precio real Recope",
        ["precio_real_cfr_bbl", "precio_real_cif_bbl", "precio_real_fob_bbl"],
        index=0,
        help="CFR se usa por defecto porque aproxima mejor FOB + flete. CIF y FOB quedan disponibles para análisis."
    )
    st.divider()
    st.caption("Fuente EIA: archivos históricos públicos. Fuente Recope: CSV suministrado por el usuario.")

if ejecutar:
    with st.spinner("Descargando datos EIA y entrenando modelo..."):
        df = cargar_datos()
        salida = preparar_modelo(df)
        historico = guardar_historico(salida["fecha_base"], salida["periodo_predicho"], salida["pred_final"], salida["prob_final"])

    pred_texto = "Sube" if salida["pred_final"] == 1 else "No sube"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Última fecha EIA", str(salida["fecha_base"].date()))
    col2.metric("Periodo predicho", str(salida["periodo_predicho"].date()))
    col3.metric("Predicción", pred_texto)
    col4.metric("Probabilidad de subida", f"{salida['prob_final']:.2%}")

    # -------------------------------------------------------------------------
    # Nuevo módulo: contraste con compras reales Recope.
    # -------------------------------------------------------------------------
    st.subheader("Contraste con compras reales Recope")
    try:
        df_recope = cargar_compras_recope(archivo_recope if archivo_recope is not None else None)
    except Exception as exc:
        df_recope = pd.DataFrame()
        st.error(f"No fue posible cargar el CSV de compras Recope: {exc}")

    if df_recope.empty:
        st.info("No se encontró un CSV de compras Recope. Cargá el archivo en la barra lateral o colocá 2016-2026_abril.csv en la carpeta del proyecto.")
    else:
        backtest, etiqueta_eia = construir_backtesting_recope(df, df_recope, producto_recope, objetivo_recope)
        if backtest.empty:
            st.warning("No hay datos suficientes para cruzar el producto seleccionado contra la serie EIA.")
        else:
            ultimo = backtest.iloc[-1]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Última compra Recope", str(pd.to_datetime(ultimo["fecha"]).date()))
            c2.metric("Proxy EIA", f"{ultimo['precio_eia_bbl']:,.2f} USD/bbl")
            c3.metric("Precio real Recope", f"{ultimo['precio_real_bbl']:,.2f} USD/bbl")
            c4.metric("Diferencia", f"{ultimo['diferencia_usd_bbl']:,.2f} USD/bbl")
            c5.metric("Cercanía", f"{ultimo['cercania_pct']:,.2f}%")

            evaluacion = evaluar_cercania(ultimo["cercania_pct"])
            if ultimo["cercania_pct"] >= 95:
                st.success(f"Evaluación de la última compra: {evaluacion}")
            elif ultimo["cercania_pct"] >= 90:
                st.warning(f"Evaluación de la última compra: {evaluacion}")
            else:
                st.error(f"Evaluación de la última compra: {evaluacion}")

            st.caption(
                f"Comparación contra {etiqueta_eia}. La cercanía se calcula como 100 - error porcentual absoluto. "
                "Este bloque mide cercanía contra un proxy EIA; no sustituye una liquidación tarifaria ni una comparación contractual detallada."
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cercanía histórica promedio", f"{backtest['cercania_pct'].mean():,.2f}%")
            m2.metric("MAE", f"{backtest['error_abs_usd_bbl'].mean():,.2f} USD/bbl")
            m3.metric("MAPE", f"{backtest['error_pct'].mean():,.2f}%")
            m4.metric("Observaciones cruzadas", f"{len(backtest):,.0f}")

            graf = backtest[["fecha", "precio_eia_bbl", "precio_real_bbl"]].melt(
                id_vars="fecha",
                value_vars=["precio_eia_bbl", "precio_real_bbl"],
                var_name="serie",
                value_name="USD/bbl"
            )
            fig_recope = px.line(
                graf,
                x="fecha",
                y="USD/bbl",
                color="serie",
                title=f"Precio proxy EIA vs compra real Recope - {producto_recope}"
            )
            st.plotly_chart(fig_recope, use_container_width=True)

            with st.expander("Ver detalle del contraste Recope"):
                st.dataframe(backtest.sort_values("fecha", ascending=False), use_container_width=True)
                st.download_button(
                    "Descargar contraste Recope CSV",
                    backtest.to_csv(index=False),
                    file_name=f"contraste_recope_{producto_recope}.csv",
                    mime="text/csv"
                )

    st.subheader("Evaluación del modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy test temporal", f"{salida['accuracy']:.2%}")
    c2.metric("Promedio validación cruzada", f"{salida['cv_scores'].mean():.2%}")
    c3.metric("Frecuencia detectada", str(salida["frecuencia"]))

    st.write("Mejores hiperparámetros:", salida["mejores_parametros"])

    st.subheader("Matriz de confusión")
    matriz_df = pd.DataFrame(salida["matriz"], index=["Real: No sube", "Real: Sube"], columns=["Pred: No sube", "Pred: Sube"])
    st.dataframe(matriz_df, use_container_width=True)

    st.subheader("Resultados del periodo de prueba")
    st.dataframe(salida["resultados"].sort_values("fecha", ascending=False), use_container_width=True)

    fig = px.line(salida["resultados"], x="fecha", y="probabilidad_sube", title="Probabilidad estimada de subida en el periodo de prueba")
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
