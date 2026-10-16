import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Visualizador PCA – Vinos",
    page_icon="🍷",
    layout="wide"
)

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Cabecera ─────────────────────────────────────────────────────────────────
st.title("🍷 Visualizador Interactivo PCA – Dataset Wine")
st.markdown("""
Esta aplicación aplica **Análisis de Componentes Principales (PCA)** al dataset Wine
de scikit-learn para reducir 13 variables químicas a 2 o 3 dimensiones y visualizar
cómo se agrupan los distintos cultivares de vino.
""")

with st.expander("ℹ️ ¿Qué es PCA y cómo usar esta app?"):
    st.markdown("""
    **PCA (Principal Component Analysis)** es una técnica de reducción de dimensionalidad
    que transforma variables correlacionadas en un conjunto menor de variables no correlacionadas
    llamadas **componentes principales**, ordenadas de mayor a menor varianza explicada.

    **Cómo usar esta app:**
    1. Ajusta los parámetros en el panel izquierdo.
    2. En la pestaña **Visualización** verás los vinos proyectados en 2D o 3D.
    3. En la pestaña **Análisis** encontrarás la varianza explicada y los loadings.
    4. En la pestaña **Predicción** puedes ingresar un vino nuevo y ver dónde cae en el espacio PCA.
    """)

# ── Cargar datos ─────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['clase'] = wine.target
    df['cultivar'] = df['clase'].map({
        0: 'Cultivar 1', 1: 'Cultivar 2', 2: 'Cultivar 3'
    })
    return wine, df

wine, df = cargar_datos()
COLORES = {'Cultivar 1': '#e74c3c', 'Cultivar 2': '#2ecc71', 'Cultivar 3': '#3498db'}

# ── Panel lateral ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Parámetros del modelo")

n_componentes = st.sidebar.radio(
    "Número de componentes principales",
    options=[2, 3],
    index=0,
    help="2 componentes para un gráfico 2D, 3 para un gráfico 3D."
)

estandarizar = st.sidebar.checkbox(
    "Estandarizar los datos (recomendado)",
    value=True,
    help="Escala todas las variables a media=0 y desv. estándar=1 antes de aplicar PCA."
)

cultivares_mostrar = st.sidebar.multiselect(
    "Cultivares a mostrar",
    options=['Cultivar 1', 'Cultivar 2', 'Cultivar 3'],
    default=['Cultivar 1', 'Cultivar 2', 'Cultivar 3']
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Wine – scikit-learn**")
st.sidebar.markdown(f"- 🍾 {df.shape[0]} muestras de vino")
st.sidebar.markdown(f"- 🔬 {len(wine.feature_names)} variables químicas")
st.sidebar.markdown(f"- 🏷️ 3 cultivares (clases)")

# ── Preprocesamiento y PCA ────────────────────────────────────────────────────
X = df[wine.feature_names].values
y = df['clase'].values

if estandarizar:
    scaler = StandardScaler()
    X_proc = scaler.fit_transform(X)
else:
    X_proc = X.copy()
    scaler = None

pca = PCA(n_components=n_componentes)
X_pca = pca.fit_transform(X_proc)

# PCA completo para el scree plot
pca_full = PCA()
pca_full.fit(X_proc)

varianza_exp = pca.explained_variance_ratio_
varianza_total = varianza_exp.sum() * 100

# DataFrame con resultados
cols_pca = [f'PC{i+1}' for i in range(n_componentes)]
df_pca = pd.DataFrame(X_pca, columns=cols_pca)
df_pca['cultivar'] = df['cultivar'].values
df_pca['clase'] = y

# Filtrar cultivares seleccionados
df_vis = df_pca[df_pca['cultivar'].isin(cultivares_mostrar)]

# ── Métricas rápidas ──────────────────────────────────────────────────────────
st.markdown("### 📊 Métricas principales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Variables originales", f"{X.shape[1]}")
col2.metric("Componentes usados", f"{n_componentes}")
col3.metric("Varianza explicada", f"{varianza_total:.1f}%")
col4.metric("Variables reducidas", f"{X.shape[1] - n_componentes} (-{((X.shape[1]-n_componentes)/X.shape[1]*100):.0f}%)")

st.markdown("---")

# ── Pestañas ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Visualización", "🔬 Análisis de varianza", "🧪 Predicción de nuevo vino"])

# ════════════════════════════════════════════════════════════════════════════
# PESTAÑA 1 – VISUALIZACIÓN
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Vinos proyectados en {n_componentes} componentes principales")
    st.caption(f"Varianza total explicada: **{varianza_total:.2f}%**")

    if not cultivares_mostrar:
        st.warning("Selecciona al menos un cultivar en el panel lateral.")
    elif n_componentes == 2:
        fig, ax = plt.subplots(figsize=(9, 6))
        for cultivar, grupo in df_vis.groupby('cultivar'):
            ax.scatter(grupo['PC1'], grupo['PC2'],
                       label=cultivar, color=COLORES[cultivar],
                       alpha=0.85, s=90, edgecolors='white', linewidth=0.6)
        ax.set_xlabel(f"PC1 — {varianza_exp[0]*100:.2f}% varianza", fontsize=12)
        ax.set_ylabel(f"PC2 — {varianza_exp[1]*100:.2f}% varianza", fontsize=12)
        ax.set_title("PCA 2D – Dataset Wine", fontsize=14)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.legend(title="Cultivar", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

        st.info(f"""
        **Interpretación:** Con 2 componentes se captura el **{varianza_total:.1f}%** de la información 
        original. Los cultivares forman grupos diferenciados, lo que indica que las variables 
        químicas permiten distinguir los tipos de vino incluso en 2 dimensiones.
        """)

    else:  # 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        for cultivar, grupo in df_vis.groupby('cultivar'):
            ax.scatter(grupo['PC1'], grupo['PC2'], grupo['PC3'],
                       label=cultivar, color=COLORES[cultivar],
                       alpha=0.85, s=70, edgecolors='white', linewidth=0.4)
        ax.set_xlabel(f"PC1 ({varianza_exp[0]*100:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC2 ({varianza_exp[1]*100:.1f}%)", fontsize=10)
        ax.set_zlabel(f"PC3 ({varianza_exp[2]*100:.1f}%)", fontsize=10)
        ax.set_title(f"PCA 3D – Dataset Wine\nVarianza explicada: {varianza_total:.2f}%", fontsize=13)
        ax.legend(title="Cultivar", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

        st.info(f"""
        **Interpretación:** Con 3 componentes se captura el **{varianza_total:.1f}%** de la información. 
        La tercera dimensión ayuda a separar mejor los grupos que se solapaban en 2D.
        """)

# ════════════════════════════════════════════════════════════════════════════
# PESTAÑA 2 – ANÁLISIS DE VARIANZA
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Varianza explicada por componente")

        varianza_full = pca_full.explained_variance_ratio_
        varianza_acum = np.cumsum(varianza_full)
        n_total = len(varianza_full)

        fig2, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].bar(range(1, n_total + 1), varianza_full * 100,
                    color=['#3498db' if i < n_componentes else '#bdc3c7'
                           for i in range(n_total)],
                    edgecolor='white', alpha=0.9)
        axes[0].set_xlabel("Componente Principal", fontsize=11)
        axes[0].set_ylabel("Varianza explicada (%)", fontsize=11)
        axes[0].set_title("Scree Plot", fontsize=12)
        axes[0].set_xticks(range(1, n_total + 1))
        for i, v in enumerate(varianza_full):
            axes[0].text(i + 1, v * 100 + 0.3, f'{v*100:.1f}%',
                         ha='center', fontsize=7.5)

        axes[1].plot(range(1, n_total + 1), varianza_acum * 100,
                     'o-', color='darkorange', linewidth=2, markersize=6)
        axes[1].axhline(80, color='gray', linestyle='--', alpha=0.6, label='80%')
        axes[1].axhline(95, color='red', linestyle='--', alpha=0.6, label='95%')
        axes[1].axvline(n_componentes, color='#3498db', linestyle='--',
                        alpha=0.7, label=f'Seleccionados: {n_componentes}')
        axes[1].set_xlabel("Número de componentes", fontsize=11)
        axes[1].set_ylabel("Varianza acumulada (%)", fontsize=11)
        axes[1].set_title("Varianza acumulada", fontsize=12)
        axes[1].set_xticks(range(1, n_total + 1))
        axes[1].legend(fontsize=9)
        axes[1].set_ylim([0, 105])

        plt.tight_layout()
        st.pyplot(fig2)

        tabla_var = pd.DataFrame({
            'Componente': [f'PC{i+1}' for i in range(n_total)],
            'Varianza explicada (%)': (varianza_full * 100).round(2),
            'Varianza acumulada (%)': (varianza_acum * 100).round(2)
        })
        st.dataframe(tabla_var, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Pesos de variables originales (Loadings)")
        st.caption("Indica cuánto contribuye cada variable original a cada componente.")

        loadings_df = pd.DataFrame(
            pca.components_.T,
            index=wine.feature_names,
            columns=[f'PC{i+1}' for i in range(n_componentes)]
        )

        fig3, ax3 = plt.subplots(figsize=(6, 8))
        sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, linewidths=0.5, ax=ax3,
                    cbar_kws={'label': 'Peso (loading)', 'shrink': 0.8})
        ax3.set_title("Heatmap de Loadings", fontsize=13)
        ax3.set_ylabel("Variable química", fontsize=11)
        ax3.set_xlabel("Componente Principal", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig3)

        top_vars = loadings_df['PC1'].abs().sort_values(ascending=False).head(5)
        st.markdown("**Top 5 variables que más influyen en PC1:**")
        for var, peso in top_vars.items():
            st.markdown(f"- `{var}`: {peso:.3f}")

# ════════════════════════════════════════════════════════════════════════════
# PESTAÑA 3 – PREDICCIÓN DE NUEVO VINO
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🧪 Ingresa un vino nuevo y ubícalo en el espacio PCA")
    st.markdown("""
    Ajusta los valores químicos del vino que quieres analizar.
    Los valores por defecto corresponden al **promedio** de todos los vinos del dataset.
    """)

    medias = df[wine.feature_names].mean()

    col_i1, col_i2 = st.columns(2)
    nuevo_vino = {}

    features_mitad1 = wine.feature_names[:7]
    features_mitad2 = wine.feature_names[7:]

    with col_i1:
        for feat in features_mitad1:
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            default  = float(medias[feat])
            nuevo_vino[feat] = st.slider(
                label=feat,
                min_value=round(min_val, 3),
                max_value=round(max_val, 3),
                value=round(default, 3),
                step=round((max_val - min_val) / 100, 4),
                key=f"slider_{feat}"
            )

    with col_i2:
        for feat in features_mitad2:
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            default  = float(medias[feat])
            nuevo_vino[feat] = st.slider(
                label=feat,
                min_value=round(min_val, 3),
                max_value=round(max_val, 3),
                value=round(default, 3),
                step=round((max_val - min_val) / 100, 4),
                key=f"slider_{feat}"
            )

    if st.button("🔍 Proyectar vino en el espacio PCA", type="primary"):
        X_nuevo = np.array([[nuevo_vino[f] for f in wine.feature_names]])

        if estandarizar and scaler is not None:
            X_nuevo_proc = scaler.transform(X_nuevo)
        else:
            X_nuevo_proc = X_nuevo

        coords_nuevo = pca.transform(X_nuevo_proc)[0]

        st.success(
            "**Coordenadas del vino en el espacio PCA:**  " +
            "  |  ".join([f"**PC{i+1}** = {coords_nuevo[i]:.3f}" for i in range(n_componentes)])
        )

        if n_componentes == 2:
            fig4, ax4 = plt.subplots(figsize=(9, 6))
            for cultivar, grupo in df_pca.groupby('cultivar'):
                ax4.scatter(grupo['PC1'], grupo['PC2'],
                            label=cultivar, color=COLORES[cultivar],
                            alpha=0.5, s=70, edgecolors='white', linewidth=0.5)
            ax4.scatter(coords_nuevo[0], coords_nuevo[1],
                        color='gold', s=250, marker='*', zorder=5,
                        edgecolors='black', linewidth=1.5, label='Tu vino ⭐')
            ax4.set_xlabel(f"PC1 — {varianza_exp[0]*100:.2f}% varianza", fontsize=12)
            ax4.set_ylabel(f"PC2 — {varianza_exp[1]*100:.2f}% varianza", fontsize=12)
            ax4.set_title("Posición del nuevo vino en el espacio PCA 2D", fontsize=13)
            ax4.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax4.axvline(0, color='gray', linewidth=0.5, linestyle='--')
            ax4.legend(title="Cultivar", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig4)

        else:
            fig4 = plt.figure(figsize=(10, 7))
            ax4 = fig4.add_subplot(111, projection='3d')
            for cultivar, grupo in df_pca.groupby('cultivar'):
                ax4.scatter(grupo['PC1'], grupo['PC2'], grupo['PC3'],
                            label=cultivar, color=COLORES[cultivar],
                            alpha=0.5, s=60, edgecolors='white', linewidth=0.3)
            ax4.scatter(coords_nuevo[0], coords_nuevo[1], coords_nuevo[2],
                        color='gold', s=200, marker='*', zorder=5,
                        edgecolors='black', linewidth=1.2, label='Tu vino ⭐')
            ax4.set_xlabel(f"PC1 ({varianza_exp[0]*100:.1f}%)", fontsize=10)
            ax4.set_ylabel(f"PC2 ({varianza_exp[1]*100:.1f}%)", fontsize=10)
            ax4.set_zlabel(f"PC3 ({varianza_exp[2]*100:.1f}%)", fontsize=10)
            ax4.set_title("Posición del nuevo vino en el espacio PCA 3D", fontsize=13)
            ax4.legend(title="Cultivar", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig4)

        # Distancias a los centroides
        st.markdown("#### ¿A qué cultivar se parece más?")
        centroides = df_pca.groupby('cultivar')[cols_pca].mean()
        distancias = {}
        for cultivar, centroide in centroides.iterrows():
            dist = np.linalg.norm(coords_nuevo - centroide.values)
            distancias[cultivar] = dist

        cultivar_cercano = min(distancias, key=distancias.get)
        df_dist = pd.DataFrame({
            'Cultivar': list(distancias.keys()),
            'Distancia al centroide': [f"{v:.4f}" for v in distancias.values()],
            'Más cercano': ['✅' if k == cultivar_cercano else '' for k in distancias.keys()]
        })
        st.dataframe(df_dist, use_container_width=True, hide_index=True)
        st.info(f"🍷 El vino ingresado se parece más al **{cultivar_cercano}** "
                f"(distancia euclidiana: {distancias[cultivar_cercano]:.4f})")

# ── Pie de página ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Dataset: Wine Dataset · scikit-learn `load_wine()` · "
           "Herramientas: scikit-learn, pandas, matplotlib, seaborn, Streamlit")
