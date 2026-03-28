# app.py
# Ejecutar con: streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from io import BytesIO

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Observatorio LATAM 2025",
    page_icon="🗳️",
    layout="wide",
)

COLORES_CANDIDATO = {
    "Jeannette Jara Román":   "#E45756",
    "José Antonio Kast":      "#B22222",
    "Franco Parisi":          "#FF7F0E",
    "Johannes Kaiser":        "#8B4513",
    "Evelyn Matthei":         "#F4A9C0",
    'Jorge "Tuto" Quiroga':   "#2CA02C",
    "Rodrigo Paz Pereira":    "#98DF8A",
    "Daniel Noboa":           "#1F77B4",
    "Luisa González":         "#AEC7E8",
}
COLORES_PAIS = {"Chile": "#D62728", "Bolivia": "#2CA02C", "Ecuador": "#1F77B4"}
COLORES_TIPO = {
    "Original": "#1F77B4",
    "Reply":    "#FF7F0E",
    "Retweet":  "#2CA02C",
    "Quote":    "#9467BD",
}

NOTA_METODOLOGICA = """
**Nota metodológica · Comparación intra-país**

Los análisis por candidato se realizan dentro de cada país por razones metodológicas:
los contextos electorales son distintos (1ra/2da vuelta, duración del período, sistema político),
los niveles de base de seguidores difieren ampliamente entre países, y el uso de Twitter/X varía
estructuralmente entre Chile, Bolivia y Ecuador. Comparar candidatos entre países introduciría
sesgos difíciles de controlar. El único análisis multi-país es el gráfico de **Volumen vs Eficiencia**,
que normaliza por tweet y permite una lectura comparada sin esos sesgos.
"""

# ── CARGA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/tweets_consolidado_20260328_204225.csv")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["semana"] = (df["created_at"]
                    .dt.to_period("W")
                    .dt.start_time
                    .dt.tz_localize(None))
    df["tipo"] = df.apply(lambda r:
        "Retweet"  if r["is_retweet"] else
        "Reply"    if r["is_reply"]   else
        "Quote"    if r["is_quote"]   else
        "Original", axis=1)
    return df

df      = load_data()
df_prop = df[~df["is_retweet"]].copy()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fmt(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}k"
    return str(int(n))

def top_hashtags(series, n=10):
    items = []
    for s in series.dropna():
        items.extend([x.strip().lower() for x in str(s).split("|") if x.strip()])
    return pd.DataFrame(Counter(items).most_common(n), columns=["hashtag", "frecuencia"])

def metricas_candidatos(data):
    return (data.groupby(["candidato_nombre", "candidato_pais", "candidato_vuelta"])
            .agg(
                tweets     = ("tweet_id",        "count"),
                likes      = ("like_count",       "sum"),
                rts        = ("retweet_count",    "sum"),
                replies    = ("reply_count",      "sum"),
                views      = ("view_count",       "sum"),
                engagement = ("engagement_total", "sum"),
            )
            .reset_index()
            .assign(
                likes_x_tweet = lambda d: (d.likes  / d.tweets).round(0).astype(int),
                views_x_tweet = lambda d: (d.views  / d.tweets).round(0).astype(int),
                eng_x_tweet   = lambda d: (d.engagement / d.tweets).round(0).astype(int),
            ))

def insight_box(texto):
    st.info(f"📌 {texto}")

def seccion(titulo, descripcion):
    """Encabezado de sección con título y descripción interpretativa."""
    st.subheader(titulo)
    st.markdown(f"<p style='color:#666;font-size:0.95em;margin-top:-8px'>{descripcion}</p>",
                unsafe_allow_html=True)

def nota_lectura(texto):
    """Nota de lectura destacada bajo un gráfico."""
    st.markdown(
        f"<div style='background:#f0f2f6;border-left:3px solid #4a90d9;"
        f"padding:8px 12px;border-radius:4px;font-size:0.88em;color:#444;"
        f"margin-bottom:12px'>💡 <b>Cómo leer este gráfico:</b> {texto}</div>",
        unsafe_allow_html=True,
    )

# ── TABS PRINCIPALES ──────────────────────────────────────────────────────────
tab_general, tab_chile, tab_bolivia, tab_ecuador, tab_candidato = st.tabs([
    "🌎 Vista general",
    "🇨🇱 Chile",
    "🇧🇴 Bolivia",
    "🇪🇨 Ecuador",
    "🧠 Perfil de candidato",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 · VISTA GENERAL
# ════════════════════════════════════════════════════════════════════════════
with tab_general:
    st.title("🗳️ Observatorio de Comunicación Política · LATAM 2025")
    st.markdown("""
    Este observatorio analiza la actividad en **Twitter/X** de los principales candidatos
    presidenciales de Chile, Bolivia y Ecuador durante sus respectivos períodos electorales de 2025.
    El objetivo es describir patrones de comunicación digital: ¿quién publica más?, ¿quién genera
    más impacto?, ¿de qué temas habla cada candidato?
    """)
    st.markdown(NOTA_METODOLOGICA)
    st.divider()

    # ── KPIs ──────────────────────────────────────────────────────────────
    seccion(
        "Totales capturados",
        "Resumen agregado de toda la base de datos. Incluye solo tweets propios "
        "(se excluyen retweets) para reflejar la producción original de cada candidato."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tweets propios",  fmt(len(df_prop)))
    c2.metric("Likes",           fmt(df_prop["like_count"].sum()))
    c3.metric("Views",           fmt(df_prop["view_count"].sum()))
    c4.metric("Retweets",        fmt(df_prop["retweet_count"].sum()))
    c5.metric("Candidatos",      df["candidato_nombre"].nunique())

    st.divider()

    # ── Insights ──────────────────────────────────────────────────────────
    met_global = metricas_candidatos(df_prop)
    top_ef  = met_global.sort_values("likes_x_tweet", ascending=False).iloc[0]
    top_vol = met_global.sort_values("tweets",         ascending=False).iloc[0]
    top_vw  = met_global.sort_values("views",          ascending=False).iloc[0]

    seccion(
        "📌 Claves del período",
        "Tres hallazgos destacados calculados automáticamente a partir de los datos. "
        "Son un punto de entrada para orientar la lectura del resto del dashboard."
    )
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        insight_box(
            f"**Mayor eficiencia:** {top_ef['candidato_nombre']} "
            f"({fmt(top_ef['likes_x_tweet'])} likes/tweet)"
        )
    with col_i2:
        insight_box(
            f"**Mayor volumen:** {top_vol['candidato_nombre']} "
            f"({fmt(top_vol['tweets'])} tweets capturados)"
        )
    with col_i3:
        insight_box(
            f"**Mayor alcance:** {top_vw['candidato_nombre']} "
            f"({fmt(top_vw['views'])} views totales)"
        )

    st.divider()

    # ── Wordcloud ─────────────────────────────────────────────────────────
    seccion(
        "¿De qué habla cada candidato?",
        "La nube de palabras muestra los términos más frecuentes en los tweets propios "
        "de cada candidato. El tamaño de cada palabra es proporcional a su frecuencia: "
        "palabras más grandes aparecen más veces. Se han eliminado stopwords (palabras "
        "vacías como artículos y preposiciones), URLs, menciones y el propio nombre del "
        "candidato para revelar los temas sustantivos de su comunicación."
    )

    candidatos_lista = sorted(df_prop["candidato_nombre"].unique().tolist())
    cand_wc = st.selectbox(
        "Selecciona un candidato",
        candidatos_lista,
        key="wc_candidato",
    )

    STOPWORDS_ES = {
        "de","la","el","en","y","a","los","del","se","las","por","un","para",
        "con","una","su","al","lo","como","más","pero","sus","le","ya","o",
        "este","sí","porque","esta","entre","cuando","muy","sin","sobre",
        "también","me","hasta","hay","donde","han","yo","él","ella","nos",
        "todo","esta","estos","estas","fue","son","ser","tiene","tenemos",
        "que","es","no","si","te","mi","tu","http","https","t","co","amp",
        "rt","via","hoy","ayer","así","bien","gran","cada","hacer","puede",
        "nuestro","nuestra","nuestros","nuestras","está","están","tiene",
        "tener","solo","todos","todas","otro","otra","años","Chile","Bolivia",
        "Ecuador","país","gobierno","presidente","presidenta",
    }

    @st.cache_data
    def generar_wordcloud(candidato: str, color: str) -> BytesIO:
        textos = df_prop[df_prop["candidato_nombre"] == candidato]["text"].dropna()
        def limpiar(texto):
            texto = re.sub(r"http\S+", "", texto)
            texto = re.sub(r"@\w+", "", texto)
            texto = re.sub(r"#(\w+)", r"\1", texto)
            texto = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", texto)
            return texto.lower()
        corpus = " ".join(limpiar(t) for t in textos)
        wc = WordCloud(
            width=900, height=420,
            background_color="white",
            color_func=lambda *args, **kwargs: color,
            stopwords=STOPWORDS_ES,
            min_word_length=4,
            max_words=80,
            collocations=False,
            prefer_horizontal=0.85,
        )
        wc.generate(corpus)
        buf = BytesIO()
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    color_wc  = COLORES_CANDIDATO.get(cand_wc, "#1F77B4")
    buf       = generar_wordcloud(cand_wc, color_wc)
    pais_wc   = df_prop[df_prop["candidato_nombre"] == cand_wc]["candidato_pais"].iloc[0]
    tweets_wc = len(df_prop[df_prop["candidato_nombre"] == cand_wc])
    st.caption(f"🗳️ {pais_wc} · {tweets_wc} tweets analizados")
    st.image(buf, use_container_width=True)
    nota_lectura(
        "Compara las nubes de distintos candidatos para identificar diferencias temáticas. "
        "Un candidato con palabras como 'seguridad', 'orden', 'frontera' tiene un perfil "
        "diferente a uno cuyas palabras dominantes son 'trabajo', 'salud' o 'pensiones'."
    )

    st.divider()

    # ── Tabla de contexto electoral ────────────────────────────────────────
    seccion(
        "Contexto electoral y cobertura",
        "Esta tabla resume las condiciones de cada candidato en el análisis: el período "
        "cubierto, la etapa electoral (1ra o 2da vuelta) y las métricas básicas capturadas. "
        "Es el punto de partida para interpretar correctamente cualquier comparación posterior, "
        "ya que los períodos y contextos no son idénticos entre países."
    )
    contexto = (df.groupby(
        ["candidato_pais","candidato_nombre","candidato_vuelta","periodo_busqueda"])
        .agg(
            tweets_capturados = ("tweet_id",        "count"),
            likes_totales     = ("like_count",      "sum"),
            views_totales     = ("view_count",      "sum"),
            seguidores_hoy    = ("author_followers","first"),
        )
        .reset_index()
        .sort_values(["candidato_pais","candidato_nombre"])
        .rename(columns={
            "candidato_pais":    "País",
            "candidato_nombre":  "Candidato",
            "candidato_vuelta":  "Vuelta",
            "periodo_busqueda":  "Período",
            "tweets_capturados": "Tweets",
            "likes_totales":     "Likes",
            "views_totales":     "Views",
            "seguidores_hoy":    "Seguidores (hoy)",
        })
    )
    for col in ["Likes","Views","Seguidores (hoy)"]:
        contexto[col] = contexto[col].apply(lambda x: f"{int(x):,}")
    st.dataframe(contexto, use_container_width=True, hide_index=True)

    st.divider()

    # ── Cobertura por país ─────────────────────────────────────────────────
    seccion(
        "Cobertura por país",
        "Volumen total de tweets propios capturados y views generados en cada país. "
        "Esta vista permite dimensionar el peso relativo de cada contexto electoral "
        "en el conjunto del dataset."
    )
    tweets_pais = (df_prop.groupby("candidato_pais")
                   .agg(tweets=("tweet_id","count"),
                        likes=("like_count","sum"),
                        views=("view_count","sum"))
                   .reset_index())

    col_a, col_b = st.columns(2)
    with col_a:
        fig_tp = px.bar(
            tweets_pais, x="candidato_pais", y="tweets",
            color="candidato_pais", color_discrete_map=COLORES_PAIS,
            text="tweets", title="Tweets propios capturados por país",
            labels={"candidato_pais":"","tweets":"Tweets"},
        )
        fig_tp.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_tp.update_layout(showlegend=False)
        st.plotly_chart(fig_tp, use_container_width=True)

    with col_b:
        fig_vp = px.bar(
            tweets_pais, x="candidato_pais", y="views",
            color="candidato_pais", color_discrete_map=COLORES_PAIS,
            text="views", title="Views totales por país",
            labels={"candidato_pais":"","views":"Views"},
        )
        fig_vp.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_vp.update_layout(showlegend=False)
        st.plotly_chart(fig_vp, use_container_width=True)

    nota_lectura(
        "Un número alto de tweets no implica necesariamente mayor impacto. "
        "Observa si el país con más tweets también lidera en views, o si hay "
        "países donde se publica menos pero se genera más alcance por tweet."
    )

    st.divider()

    # ── Volumen vs Eficiencia ──────────────────────────────────────────────
    seccion(
        "Volumen vs Eficiencia · comparación multi-país",
        "Este es el único gráfico que permite comparar candidatos entre países. "
        "Al usar el promedio de likes por tweet en lugar de totales, se neutraliza "
        "el efecto del período de análisis y el nivel de base de seguidores, "
        "haciendo la comparación más justa metodológicamente."
    )
    fig_burbuja = px.scatter(
        met_global,
        x="tweets", y="likes_x_tweet",
        size="views", color="candidato_nombre",
        color_discrete_map=COLORES_CANDIDATO,
        symbol="candidato_pais",
        hover_name="candidato_nombre",
        hover_data={"candidato_pais":True,"tweets":True,
                    "likes_x_tweet":True,"views_x_tweet":True,"views":False},
        title="Volumen (tweets) vs Eficiencia (likes/tweet)",
        labels={"tweets":"N° tweets capturados",
                "likes_x_tweet":"Likes promedio por tweet",
                "candidato_pais":"País"},
        size_max=60,
    )
    fig_burbuja.update_layout(legend_title="Candidato", height=480)
    st.plotly_chart(fig_burbuja, use_container_width=True)
    nota_lectura(
        "El eje X muestra cuánto publica cada candidato; el eje Y, qué tan bien "
        "responde la audiencia a cada tweet. El tamaño de la burbuja indica el "
        "alcance total en views. El candidato ideal aparece arriba a la derecha "
        "(mucho volumen y alta respuesta), pero lo más informativo suele ser "
        "encontrar candidatos en los extremos: alta eficiencia con poco volumen "
        "(estrategia selectiva) o mucho volumen con baja respuesta (estrategia de presencia)."
    )


# ════════════════════════════════════════════════════════════════════════════
# FUNCIÓN REUTILIZABLE POR PAÍS
# ════════════════════════════════════════════════════════════════════════════
def render_pais(pais: str):
    data     = df_prop[df_prop["candidato_pais"] == pais].copy()
    data_all = df[df["candidato_pais"] == pais].copy()
    met      = metricas_candidatos(data).sort_values("likes", ascending=False)
    candidatos = met["candidato_nombre"].tolist()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidatos",     len(candidatos))
    c2.metric("Tweets propios", fmt(len(data)))
    c3.metric("Likes totales",  fmt(data["like_count"].sum()))
    c4.metric("Views totales",  fmt(data["view_count"].sum()))

    st.divider()

    # ── Insights ──────────────────────────────────────────────────────────
    top_ef  = met.sort_values("likes_x_tweet", ascending=False).iloc[0]
    top_vol = met.sort_values("tweets",         ascending=False).iloc[0]
    low_ef  = met.sort_values("likes_x_tweet").iloc[0]
    ratio   = top_ef["likes_x_tweet"] / max(low_ef["likes_x_tweet"], 1)

    seccion(
        "📌 Claves del período",
        "Hallazgos principales calculados a partir de los datos de este país. "
        "Sirven como hipótesis de lectura para explorar el resto de las secciones."
    )
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        insight_box(
            f"**Mayor eficiencia:** {top_ef['candidato_nombre']} "
            f"con {fmt(top_ef['likes_x_tweet'])} likes/tweet"
        )
    with col_i2:
        insight_box(
            f"**Brecha de eficiencia:** {top_ef['candidato_nombre'].split()[0]} "
            f"genera {ratio:.1f}x más likes/tweet que {low_ef['candidato_nombre'].split()[0]}"
        )
    with col_i3:
        insight_box(
            f"**Mayor volumen:** {top_vol['candidato_nombre'].split()[0]} "
            f"con {fmt(top_vol['tweets'])} tweets capturados"
        )

    st.divider()

    # ── Engagement total ───────────────────────────────────────────────────
    seccion(
        "Engagement total",
        "Suma acumulada de likes y views generados por cada candidato durante "
        "todo el período analizado. Mide el **impacto bruto** de la comunicación: "
        "a mayor barra, mayor respuesta total de la audiencia. Esta métrica "
        "favorece a quienes publicaron más tweets, por eso debe leerse junto "
        "con los promedios por tweet de la sección siguiente."
    )
    col1, col2 = st.columns(2)
    with col1:
        fig_likes = px.bar(
            met.sort_values("likes"),
            x="likes", y="candidato_nombre", orientation="h",
            color="candidato_nombre", color_discrete_map=COLORES_CANDIDATO,
            text="likes", title="Likes totales",
            labels={"likes":"Likes","candidato_nombre":""},
        )
        fig_likes.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_likes.update_layout(showlegend=False)
        st.plotly_chart(fig_likes, use_container_width=True)

    with col2:
        fig_views = px.bar(
            met.sort_values("views"),
            x="views", y="candidato_nombre", orientation="h",
            color="candidato_nombre", color_discrete_map=COLORES_CANDIDATO,
            text="views", title="Views totales",
            labels={"views":"Views","candidato_nombre":""},
        )
        fig_views.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_views.update_layout(showlegend=False)
        st.plotly_chart(fig_views, use_container_width=True)

    nota_lectura(
        "Los likes indican aprobación activa: el usuario decidió interactuar. "
        "Los views miden alcance pasivo: cuántas veces fue visto el tweet, "
        "independientemente de si generó reacción. Una brecha grande entre "
        "views y likes puede indicar contenido que se consume pero no convence."
    )

    st.divider()

    # ── Eficiencia ─────────────────────────────────────────────────────────
    seccion(
        "Eficiencia por tweet",
        "Promedio de likes y views **por cada tweet publicado**. A diferencia del "
        "engagement total, esta métrica elimina el efecto del volumen: un candidato "
        "que publica 10 tweets y genera 5.000 likes tiene una eficiencia de 500 likes/tweet, "
        "superior a otro que publica 100 tweets y genera 10.000 likes (100 likes/tweet). "
        "Es la métrica más relevante para comparar estrategias comunicacionales."
    )
    col3, col4 = st.columns(2)
    with col3:
        fig_lpt = px.bar(
            met.sort_values("likes_x_tweet"),
            x="likes_x_tweet", y="candidato_nombre", orientation="h",
            color="candidato_nombre", color_discrete_map=COLORES_CANDIDATO,
            text="likes_x_tweet", title="Likes promedio por tweet",
            labels={"likes_x_tweet":"Likes/tweet","candidato_nombre":""},
        )
        fig_lpt.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_lpt.update_layout(showlegend=False)
        st.plotly_chart(fig_lpt, use_container_width=True)

    with col4:
        fig_vpt = px.bar(
            met.sort_values("views_x_tweet"),
            x="views_x_tweet", y="candidato_nombre", orientation="h",
            color="candidato_nombre", color_discrete_map=COLORES_CANDIDATO,
            text="views_x_tweet", title="Views promedio por tweet",
            labels={"views_x_tweet":"Views/tweet","candidato_nombre":""},
        )
        fig_vpt.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_vpt.update_layout(showlegend=False)
        st.plotly_chart(fig_vpt, use_container_width=True)

    nota_lectura(
        "Observa si el ranking de eficiencia coincide con el de engagement total. "
        "Cuando no coinciden, hay una historia interesante: el candidato más activo "
        "no es necesariamente el que mejor conecta con su audiencia por publicación."
    )

    st.divider()

    # ── Evolución temporal ─────────────────────────────────────────────────
    seccion(
        "Evolución temporal",
        "Actividad y respuesta semana a semana durante el período electoral. "
        "Permite identificar momentos de mayor intensidad comunicacional: debates, "
        "lanzamientos de campaña, eventos polémicos o hitos electorales que generaron "
        "picos de actividad o engagement."
    )
    semanal = (data.groupby(["semana","candidato_nombre"])
               .agg(tweets=("tweet_id","count"),
                    likes=("like_count","sum"),
                    views=("view_count","sum"))
               .reset_index())

    metrica_temp = st.radio(
        "Métrica a visualizar",
        ["likes","tweets","views"],
        horizontal=True,
        key=f"radio_temp_{pais}",
        help="'Tweets' muestra el volumen de publicación. 'Likes' y 'Views' muestran la respuesta de la audiencia."
    )
    fig_temp = px.line(
        semanal, x="semana", y=metrica_temp,
        color="candidato_nombre", color_discrete_map=COLORES_CANDIDATO,
        markers=True,
        title=f"{metrica_temp.capitalize()} por semana — {pais}",
        labels={"semana":"Semana", metrica_temp:metrica_temp.capitalize(),
                "candidato_nombre":"Candidato"},
    )
    fig_temp.update_layout(legend_title="Candidato", height=400)
    st.plotly_chart(fig_temp, use_container_width=True)
    nota_lectura(
        "Busca picos simultáneos en varios candidatos: suelen coincidir con debates "
        "o eventos electorales relevantes. Un candidato con picos aislados puede "
        "haber protagonizado un momento viral propio. Cambia entre 'tweets' y 'likes' "
        "para ver si los picos de publicación se traducen en picos de respuesta."
    )

    st.divider()

    # ── Tipo de tweet ──────────────────────────────────────────────────────
    seccion(
        "Composición de tweets",
        "Desglose del tipo de publicaciones de cada candidato: tweets originales "
        "(contenido propio), replies (respuestas a otros usuarios), retweets (difusión "
        "de contenido ajeno) y quotes (retweet con comentario propio). La composición "
        "revela la **estrategia conversacional**: un candidato con muchos replies es "
        "más dialógico; uno con muchos originales tiene una comunicación más broadcast."
    )
    tipos_cand = (data_all.groupby(["candidato_nombre","tipo"])
                  .size().reset_index(name="n"))
    col5, col6 = st.columns(2)

    with col5:
        fig_tipo = px.bar(
            tipos_cand, x="candidato_nombre", y="n",
            color="tipo", barmode="stack",
            color_discrete_map=COLORES_TIPO,
            title="Composición por candidato",
            labels={"candidato_nombre":"","n":"N° tweets","tipo":"Tipo"},
        )
        fig_tipo.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig_tipo, use_container_width=True)

    with col6:
        tipos_pais_df = data_all["tipo"].value_counts().reset_index()
        tipos_pais_df.columns = ["tipo","n"]
        fig_pie = px.pie(
            tipos_pais_df, values="n", names="tipo",
            title=f"Proporción global — {pais}",
            color_discrete_map=COLORES_TIPO,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    nota_lectura(
        "Los tweets originales son el contenido de campaña puro. "
        "Un alto porcentaje de replies sugiere un candidato que debate y responde; "
        "puede ser una fortaleza (cercano, accesible) o una debilidad (reactivo, "
        "sin agenda propia). Los retweets indican qué voces amplifican."
    )

    st.divider()

    # ── Hashtags ───────────────────────────────────────────────────────────
    seccion(
        "Hashtags más usados",
        "Los hashtags son las etiquetas temáticas que los candidatos usan para "
        "posicionar sus mensajes en conversaciones más amplias o para crear narrativas "
        "propias de campaña. Su análisis revela los **ejes temáticos** de la comunicación "
        "y el grado de coordinación con otros actores del ecosistema político."
    )
    col7, col8 = st.columns([1, 2])
    with col7:
        candidato_sel = st.selectbox(
            "Ver hashtags de:",
            ["Todos"] + candidatos,
            key=f"ht_sel_{pais}",
        )
    with col8:
        n_tags = st.slider("N° de hashtags a mostrar", 5, 20, 10, key=f"ht_n_{pais}")

    ht_data = (data if candidato_sel == "Todos"
               else data[data["candidato_nombre"] == candidato_sel])
    ht_df = top_hashtags(ht_data["hashtags"], n=n_tags)

    if not ht_df.empty:
        color_ht = (COLORES_PAIS[pais] if candidato_sel == "Todos"
                    else COLORES_CANDIDATO.get(candidato_sel, "#1F77B4"))
        fig_ht = px.bar(
            ht_df.sort_values("frecuencia"),
            x="frecuencia", y="hashtag", orientation="h",
            text="frecuencia",
            title=f"Top {n_tags} hashtags — {candidato_sel}",
            labels={"frecuencia":"Frecuencia","hashtag":""},
            color_discrete_sequence=[color_ht],
        )
        fig_ht.update_traces(textposition="outside")
        fig_ht.update_layout(height=max(300, n_tags * 32))
        st.plotly_chart(fig_ht, use_container_width=True)
        nota_lectura(
            "Distingue entre hashtags propios de campaña (suelen incluir el nombre del "
            "candidato o una consigna propia) y hashtags de agenda pública (debates, "
            "eventos). Los hashtags compartidos entre candidatos revelan las temáticas "
            "que dominaron el debate electoral."
        )

    st.divider()

    # ── Top tweets ─────────────────────────────────────────────────────────
    seccion(
        "Tweets más virales",
        "Los tweets con mayor impacto del período, ordenables por likes, views o retweets. "
        "Son el material empírico más directo: muestran exactamente qué tipo de contenido "
        "generó la mayor respuesta en la audiencia, en las propias palabras del candidato."
    )
    col9, col10 = st.columns([1, 2])
    with col9:
        cand_top = st.selectbox(
            "Filtrar por candidato:",
            ["Todos"] + candidatos,
            key=f"top_sel_{pais}",
        )
    with col10:
        metrica_top = st.radio(
            "Ordenar por:",
            ["like_count","view_count","retweet_count"],
            format_func=lambda x: {"like_count":"Likes",
                                   "view_count":"Views",
                                   "retweet_count":"Retweets"}[x],
            horizontal=True,
            key=f"top_met_{pais}",
        )

    top_data = (data if cand_top == "Todos"
                else data[data["candidato_nombre"] == cand_top])
    top5 = (top_data.sort_values(metrica_top, ascending=False)
            .head(5).reset_index(drop=True))

    for _, row in top5.iterrows():
        with st.container(border=True):
            ca, cb, cc, cd = st.columns([2, 1, 1, 1])
            ca.markdown(
                f"**{row['candidato_nombre']}**  \n"
                f"🗓️ {str(row['created_at'])[:10]}"
            )
            cb.metric("❤️ Likes",  fmt(row["like_count"]))
            cc.metric("🔁 RTs",    fmt(row["retweet_count"]))
            cd.metric("👁️ Views",  fmt(row["view_count"]))
            st.markdown(
                f"> {str(row['text'])[:220]}"
                f"{'…' if len(str(row['text'])) > 220 else ''}"
            )
            st.markdown(f"[Ver tweet en X ↗]({row['tweet_url']})")


# ── TABS DE PAÍS ──────────────────────────────────────────────────────────────
with tab_chile:
    st.title("🇨🇱 Chile — Elecciones presidenciales 2025")
    st.caption("Período analizado: 17 sept – 14 dic 2025 · 1ra y 2da vuelta")
    st.markdown("""
    Chile celebró elecciones presidenciales en dos vueltas. El análisis cubre desde
    el inicio de la campaña oficial hasta el balotaje, permitiendo observar cómo
    evolucionó la comunicación digital de los candidatos a lo largo del proceso.
    Se incluyen cinco candidatos con presencia activa en Twitter/X durante el período.
    """)
    render_pais("Chile")

with tab_bolivia:
    st.title("🇧🇴 Bolivia — Elecciones presidenciales 2025")
    st.caption("Período analizado: 13 jul – 19 oct 2025 · 2da vuelta")
    st.markdown("""
    El análisis de Bolivia se concentra en la segunda vuelta electoral,
    comparando la comunicación digital de los dos candidatos finalistas.
    El período más acotado (3 meses) y el menor volumen de tweets reflejan
    un ecosistema digital diferente al de Chile o Ecuador.
    """)
    render_pais("Bolivia")

with tab_ecuador:
    st.title("🇪🇨 Ecuador — Elecciones presidenciales 2025")
    st.caption("Período analizado: 5 ene – 13 abr 2025 · 2da vuelta")
    st.markdown("""
    Ecuador fue el primer país del estudio en celebrar elecciones en 2025.
    El análisis cubre la segunda vuelta entre Daniel Noboa, presidente en ejercicio
    que buscaba la reelección, y Luisa González. El contexto de incumbencia
    es relevante para interpretar las diferencias de alcance entre ambos candidatos.
    """)
    render_pais("Ecuador")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 · PERFIL DE CANDIDATO
# ════════════════════════════════════════════════════════════════════════════
with tab_candidato:
    st.title("🧠 Perfil de candidato")
    st.markdown("""
    Análisis individual y detallado de cada candidato. Incluye una clasificación
    automática de su estrategia comunicacional, la evolución de su actividad,
    la distribución estadística de su engagement y sus tweets más virales.
    """)

    cand = st.selectbox(
        "Selecciona un candidato",
        df["candidato_nombre"].unique(),
    )

    data_c     = df_prop[df_prop["candidato_nombre"] == cand].copy()
    data_c_all = df[df["candidato_nombre"] == cand].copy()
    pais_c     = data_c["candidato_pais"].iloc[0]

    tweets = len(data_c)
    likes  = data_c["like_count"].sum()
    views  = data_c["view_count"].sum()
    lpt    = likes / tweets if tweets > 0 else 0
    vpt    = views / tweets if tweets > 0 else 0

    media_lpt_pais = (df_prop[df_prop["candidato_pais"] == pais_c]["like_count"].sum() /
                      len(df_prop[df_prop["candidato_pais"] == pais_c]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tweets propios", fmt(tweets))
    c2.metric("Likes totales",  fmt(likes))
    c3.metric("Views totales",  fmt(views))
    c4.metric("Likes/tweet",    fmt(int(lpt)),
              delta=f"{((lpt/media_lpt_pais)-1)*100:+.0f}% vs media {pais_c}")

    st.divider()

    # Perfil comunicacional
    vol_label = ("alto volumen" if tweets > df_prop.shape[0] / df["candidato_nombre"].nunique()
                 else "bajo volumen")
    eng_label = ("alto engagement" if lpt > media_lpt_pais else "engagement moderado")
    estrategia = {
        ("alto volumen",  "alto engagement"):
            "**Estrategia dominante:** publica con alta frecuencia y cada publicación "
            "genera una respuesta considerable. Es la combinación más efectiva.",
        ("alto volumen",  "engagement moderado"):
            "**Estrategia de presencia:** mantiene alta visibilidad pero el impacto "
            "promedio por tweet es inferior a la media. Cantidad sobre calidad.",
        ("bajo volumen",  "alto engagement"):
            "**Estrategia selectiva:** publica con moderación pero cada tweet resuena "
            "más que el promedio. Apuesta por la calidad sobre la cantidad.",
        ("bajo volumen",  "engagement moderado"):
            "**Estrategia limitada:** baja frecuencia de publicación y respuesta "
            "inferior al promedio del país. Menor presencia digital relativa.",
    }

    seccion(
        "📌 Perfil comunicacional",
        "Clasificación automática basada en dos dimensiones: volumen de publicación "
        "(tweets propios respecto a la media del conjunto) y engagement (likes por tweet "
        "respecto a la media del país). Genera cuatro perfiles estratégicos posibles."
    )
    insight_box(
        f"**{cand}** · {pais_c} · {data_c['candidato_vuelta'].iloc[0]}  \n"
        f"Publicó **{tweets} tweets propios** · promedio de **{int(lpt):,} likes/tweet** "
        f"y **{int(vpt):,} views/tweet** · "
        f"Clasificación: **{vol_label}** · **{eng_label}**  \n\n"
        f"{estrategia.get((vol_label, eng_label), '')}"
    )

    st.divider()

    # Evolución temporal
    seccion(
        "Evolución semanal",
        "Actividad semana a semana del candidato. El gráfico de área permite ver "
        "tanto la tendencia general como la magnitud de cada semana. Útil para "
        "identificar en qué momentos concentró su comunicación."
    )
    semanal_c = (data_c.groupby("semana")
                 .agg(tweets=("tweet_id","count"),
                      likes=("like_count","sum"),
                      views=("view_count","sum"))
                 .reset_index())
    metrica_c = st.radio(
        "Métrica", ["likes","tweets","views"],
        horizontal=True, key="radio_cand",
    )
    color_c = COLORES_CANDIDATO.get(cand, "#1F77B4")
    fig_ev = px.area(
        semanal_c, x="semana", y=metrica_c,
        title=f"Evolución semanal — {cand}",
        labels={"semana":"Semana", metrica_c:metrica_c.capitalize()},
        color_discrete_sequence=[color_c],
    )
    st.plotly_chart(fig_ev, use_container_width=True)
    nota_lectura(
        "Los picos pueden corresponder a debates, anuncios de campaña o momentos "
        "de controversia. Cambia entre 'tweets' y 'likes' para ver si publicó más "
        "en esas semanas o si el contenido existente simplemente tuvo más repercusión."
    )

    st.divider()

    # Composición y distribución
    seccion(
        "Composición y distribución de engagement",
        "Dos vistas complementarias: el desglose por tipo de publicación revela la "
        "estrategia conversacional; el histograma de likes muestra si el engagement "
        "está concentrado en pocos tweets virales o distribuido de forma homogénea."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        tipos_c = data_c_all["tipo"].value_counts().reset_index()
        tipos_c.columns = ["tipo","n"]
        fig_pie_c = px.pie(
            tipos_c, values="n", names="tipo",
            title="Tipos de publicación",
            color_discrete_map=COLORES_TIPO,
        )
        st.plotly_chart(fig_pie_c, use_container_width=True)
    with col_b:
        fig_hist = px.histogram(
            data_c, x="like_count", nbins=20,
            title="Distribución de likes por tweet",
            labels={"like_count":"Likes por tweet","count":"N° tweets"},
            color_discrete_sequence=[color_c],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    nota_lectura(
        "En el histograma, una distribución muy sesgada a la derecha (cola larga) "
        "indica que el promedio de likes está inflado por pocos tweets muy virales. "
        "Una distribución más uniforme indica un engagement más consistente. "
        "Ambos patrones son válidos pero implican estrategias distintas."
    )

    st.divider()

    # Hashtags
    seccion(
        "Hashtags más usados",
        "Los hashtags propios de este candidato. Revelan sus ejes de campaña, "
        "las temáticas que buscó posicionar y los eventos en que participó activamente."
    )
    n_ht = st.slider("N° de hashtags", 5, 20, 10, key="ht_cand")
    ht_c = top_hashtags(data_c["hashtags"], n=n_ht)
    if not ht_c.empty:
        fig_ht_c = px.bar(
            ht_c.sort_values("frecuencia"),
            x="frecuencia", y="hashtag", orientation="h",
            text="frecuencia",
            title=f"Top {n_ht} hashtags de {cand}",
            labels={"frecuencia":"Frecuencia","hashtag":""},
            color_discrete_sequence=[color_c],
        )
        fig_ht_c.update_traces(textposition="outside")
        fig_ht_c.update_layout(height=max(300, n_ht * 32))
        st.plotly_chart(fig_ht_c, use_container_width=True)

    st.divider()

    # Top tweets
    seccion(
        "🔥 Tweets más virales",
        "Los tweets con mayor impacto del período. Son la evidencia más directa "
        "de qué tipo de mensajes conectaron con la audiencia: tono, tema, formato."
    )
    metrica_tv = st.radio(
        "Ordenar por:",
        ["like_count","view_count","retweet_count"],
        format_func=lambda x: {"like_count":"Likes",
                               "view_count":"Views",
                               "retweet_count":"Retweets"}[x],
        horizontal=True, key="met_top_cand",
    )
    n_top = st.slider("N° de tweets a mostrar", 3, 10, 5, key="n_top_cand")
    top_c = (data_c.sort_values(metrica_tv, ascending=False)
             .head(n_top).reset_index(drop=True))

    for _, row in top_c.iterrows():
        with st.container(border=True):
            ca, cb, cc, cd = st.columns([2, 1, 1, 1])
            ca.markdown(f"🗓️ **{str(row['created_at'])[:10]}**")
            cb.metric("❤️ Likes",  fmt(row["like_count"]))
            cc.metric("🔁 RTs",    fmt(row["retweet_count"]))
            cd.metric("👁️ Views",  fmt(row["view_count"]))
            st.markdown(
                f"> {str(row['text'])[:220]}"
                f"{'…' if len(str(row['text'])) > 220 else ''}"
            )
            st.markdown(f"[Ver tweet en X ↗]({row['tweet_url']})")
