# app.py
# Ejecutar con: streamlit run app.py
#
# Archivos de datos requeridos en /data:
#   tweets_consolidado_20260328_204225.csv      ← tweets propios de candidatos
#   opinion_consolidado_20260329_172545.csv     ← tweets ciudadanos sobre candidatos

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

**Sobre los datos de opinión ciudadana:** Los tweets de ciudadanos se capturaron con 
4 estrategias de búsqueda (menciones directas, hashtags de campaña, búsqueda por nombre 
y opinión contextualizada), excluyendo siempre los tweets propios del candidato. 
El volumen es una muestra representativa, no exhaustiva.
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

@st.cache_data
def load_opinion():
    try:
        df_op = pd.read_csv("data/opinion_consolidado_20260329_172545.csv")
        df_op["created_at"] = pd.to_datetime(df_op["created_at"], utc=True, errors="coerce")
        df_op["semana"] = (df_op["created_at"]
                           .dt.to_period("W")
                           .dt.start_time
                           .dt.tz_localize(None))
        return df_op
    except FileNotFoundError:
        return pd.DataFrame()

df      = load_data()
df_prop = df[~df["is_retweet"]].copy()
df_op   = load_opinion()
HAY_OPINION = not df_op.empty

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
    st.subheader(titulo)
    st.markdown(f"<p style='color:#666;font-size:0.95em;margin-top:-8px'>{descripcion}</p>",
                unsafe_allow_html=True)

def nota_lectura(texto):
    st.markdown(
        f"<div style='background:#f0f2f6;border-left:3px solid #4a90d9;"
        f"padding:8px 12px;border-radius:4px;font-size:0.88em;color:#444;"
        f"margin-bottom:12px'>💡 <b>Cómo leer este gráfico:</b> {texto}</div>",
        unsafe_allow_html=True,
    )

# ── SCORECARD DE OPINIÓN CIUDADANA ────────────────────────────────────────────
def render_scorecard_opinion(candidato: str, color: str):
    """
    Bloque de storytelling: voz del candidato → eco ciudadano.
    Incluye scorecard comparativo + gráficos espejo de hashtags + evolución cruzada.
    """
    if not HAY_OPINION:
        st.warning("Archivo de opinión ciudadana no encontrado en /data/")
        return

    # ── Datos del candidato (voz propia) ─────────────────────────────────
    d_cand = df_prop[df_prop["candidato_nombre"] == candidato].copy()
    # ── Datos ciudadanos (eco) ────────────────────────────────────────────
    # La columna puede llamarse candidato_nombre o candidato
    col_cand = "candidato_nombre" if "candidato_nombre" in df_op.columns else "candidato"
    d_op = df_op[df_op[col_cand] == candidato].copy()

    if d_cand.empty:
        st.warning(f"Sin datos propios para {candidato}")
        return

    # ── Métricas base ─────────────────────────────────────────────────────
    tweets_propios   = len(d_cand)
    likes_propios    = int(d_cand["like_count"].sum())
    views_propios    = int(d_cand["view_count"].sum())
    rts_propios      = int(d_cand["retweet_count"].sum())

    tweets_ciudadanos = len(d_op)
    likes_ciudadanos  = int(d_op["like_count"].sum()) if not d_op.empty else 0
    views_ciudadanos  = int(d_op["view_count"].sum()) if not d_op.empty else 0
    autores_unicos    = int(d_op["autor_username"].nunique()) if not d_op.empty and "autor_username" in d_op.columns else 0

    # Ratio de resonancia (eco / voz propia)
    ratio_tweets = round(tweets_ciudadanos / max(tweets_propios, 1), 1)
    ratio_likes  = round(likes_ciudadanos  / max(likes_propios,  1), 1)

    # ── SEPARADOR Y TÍTULO DE SECCIÓN ─────────────────────────────────────
    st.divider()
    st.markdown(
        """
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>
          <span style='font-size:1.4em'>🗣️</span>
          <h3 style='margin:0'>Y esto es lo que dice la gente</h3>
        </div>
        <p style='color:#666;font-size:0.95em;margin-top:0;margin-bottom:20px'>
        Análisis de los tweets de ciudadanos que mencionan, etiquetan o comentan
        sobre este candidato durante el mismo período electoral. Contrasta con su
        comunicación propia para revelar brechas entre el mensaje que emite y
        la conversación que genera.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── SCORECARD COMPARATIVO ─────────────────────────────────────────────
    # Estilo de tarjetas con indicador de dirección
    def delta_label(ratio):
        if ratio >= 2.0:  return "🔥 Alto eco"
        if ratio >= 1.0:  return "📢 Eco moderado"
        if ratio >= 0.5:  return "📉 Eco bajo"
        return "🔇 Eco muy bajo"

    def delta_color(ratio):
        if ratio >= 2.0:  return "#2CA02C"
        if ratio >= 1.0:  return "#FF7F0E"
        if ratio >= 0.5:  return "#B22222"
        return "#888888"

    st.markdown("#### 📊 Scorecard: voz propia vs eco ciudadano")

    # Fila 1: métricas de volumen
    cols = st.columns(4)
    with cols[0]:
        st.metric(
            label="Tweets publicados (candidato)",
            value=fmt(tweets_propios),
            help="Tweets originales del candidato durante el período"
        )
    with cols[1]:
        st.metric(
            label="Tweets sobre él (ciudadanos)",
            value=fmt(tweets_ciudadanos),
            delta=f"ratio {ratio_tweets}x",
            help="Tweets de otros usuarios que mencionan, etiquetan o hablan del candidato"
        )
    with cols[2]:
        st.metric(
            label="Autores únicos",
            value=fmt(autores_unicos),
            help="Cantidad de personas distintas que tuitearon sobre el candidato"
        )
    with cols[3]:
        # Tarjeta de resonancia con color semántico
        resonancia = delta_label(ratio_tweets)
        st.markdown(
            f"""
            <div style='background:#f8f9fa;border-radius:8px;padding:14px 16px;
                        border-left:4px solid {delta_color(ratio_tweets)}'>
                <div style='font-size:0.78em;color:#666;margin-bottom:4px'>Resonancia de volumen</div>
                <div style='font-size:1.3em;font-weight:600;color:{delta_color(ratio_tweets)}'>{resonancia}</div>
                <div style='font-size:0.8em;color:#888;margin-top:2px'>
                    Por cada tweet propio, {ratio_tweets} tweets ciudadanos
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

    # Fila 2: métricas de engagement
    cols2 = st.columns(4)
    with cols2[0]:
        st.metric(
            label="Likes generados (propios)",
            value=fmt(likes_propios),
            help="Suma de likes en tweets del candidato"
        )
    with cols2[1]:
        st.metric(
            label="Likes en tweets ciudadanos",
            value=fmt(likes_ciudadanos),
            delta=f"ratio {ratio_likes}x",
            help="Suma de likes en tweets de ciudadanos sobre el candidato"
        )
    with cols2[2]:
        st.metric(
            label="Views propios",
            value=fmt(views_propios),
        )
    with cols2[3]:
        st.markdown(
            f"""
            <div style='background:#f8f9fa;border-radius:8px;padding:14px 16px;
                        border-left:4px solid {delta_color(ratio_likes)}'>
                <div style='font-size:0.78em;color:#666;margin-bottom:4px'>Resonancia de engagement</div>
                <div style='font-size:1.3em;font-weight:600;color:{delta_color(ratio_likes)}'>{delta_label(ratio_likes)}</div>
                <div style='font-size:0.8em;color:#888;margin-top:2px'>
                    Por cada like propio, {ratio_likes} likes ciudadanos
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    nota_lectura(
        "Un ratio de resonancia alto (🔥) indica que la conversación ciudadana supera "
        "en volumen o engagement al mensaje del propio candidato: su figura genera "
        "debate independientemente de lo que publique. Un ratio bajo (🔇) puede indicar "
        "menor presencia en la conversación pública o una base de seguidores muy fiel "
        "pero poco viral."
    )

    # ── GRÁFICO ESPEJO: HASHTAGS PROPIOS VS CIUDADANOS ────────────────────
    if not d_op.empty:
        st.markdown("#### #️⃣ Hashtags: la agenda del candidato vs la agenda ciudadana")
        st.markdown(
            "<p style='color:#666;font-size:0.9em;margin-top:-6px;margin-bottom:16px'>"
            "¿Los hashtags que usa el candidato son los mismos que usa la gente cuando habla de él? "
            "Las diferencias revelan brechas entre el mensaje que quiere posicionar y los temas "
            "que la ciudadanía asocia con su figura."
            "</p>",
            unsafe_allow_html=True,
        )

        col_ht1, col_ht2 = st.columns(2)

        with col_ht1:
            ht_cand = top_hashtags(d_cand["hashtags"], n=8)
            if not ht_cand.empty:
                fig_ht_c = px.bar(
                    ht_cand.sort_values("frecuencia"),
                    x="frecuencia", y="hashtag", orientation="h",
                    text="frecuencia",
                    title=f"Hashtags del candidato",
                    labels={"frecuencia": "Usos", "hashtag": ""},
                    color_discrete_sequence=[color],
                )
                fig_ht_c.update_traces(textposition="outside")
                fig_ht_c.update_layout(
                    height=320,
                    showlegend=False,
                    title_font_size=14,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_ht_c, use_container_width=True)

        with col_ht2:
            col_ht_op = "hashtags" if "hashtags" in d_op.columns else None
            if col_ht_op:
                ht_op = top_hashtags(d_op[col_ht_op], n=8)
                if not ht_op.empty:
                    # Color más suave (ciudadanía)
                    color_ciudadanos = "#7B8FA1"
                    fig_ht_op = px.bar(
                        ht_op.sort_values("frecuencia"),
                        x="frecuencia", y="hashtag", orientation="h",
                        text="frecuencia",
                        title=f"Hashtags que usa la ciudadanía",
                        labels={"frecuencia": "Usos", "hashtag": ""},
                        color_discrete_sequence=[color_ciudadanos],
                    )
                    fig_ht_op.update_traces(textposition="outside")
                    fig_ht_op.update_layout(
                        height=320,
                        showlegend=False,
                        title_font_size=14,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_ht_op, use_container_width=True)

        # ── Hashtags compartidos (intersección) ───────────────────────────
        if not ht_cand.empty and col_ht_op and not ht_op.empty:
            tags_cand = set(ht_cand["hashtag"].str.lower())
            tags_op   = set(ht_op["hashtag"].str.lower())
            comunes   = tags_cand & tags_op
            solo_cand = tags_cand - tags_op
            solo_op   = tags_op - tags_cand

            if comunes:
                st.success(
                    f"**Agenda compartida** (aparecen en ambos): "
                    f"{', '.join(f'#{t}' for t in sorted(comunes))}"
                )
            if solo_cand:
                st.info(
                    f"**Solo en tweets del candidato** (agenda propia): "
                    f"{', '.join(f'#{t}' for t in sorted(solo_cand))}"
                )
            if solo_op:
                st.warning(
                    f"**Solo en tweets ciudadanos** (temas que la gente añade): "
                    f"{', '.join(f'#{t}' for t in sorted(solo_op))}"
                )

    # ── EVOLUCIÓN TEMPORAL CRUZADA ────────────────────────────────────────
    if not d_op.empty and "semana" in d_op.columns:
        st.markdown("#### 📈 Actividad en el tiempo: candidato vs ciudadanía")
        st.markdown(
            "<p style='color:#666;font-size:0.9em;margin-top:-6px;margin-bottom:16px'>"
            "¿La ciudadanía habla más del candidato cuando él publica más, o la conversación "
            "ciudadana tiene vida propia? Picos simultáneos sugieren reacción directa; "
            "picos desfasados sugieren que los eventos externos mueven la conversación."
            "</p>",
            unsafe_allow_html=True,
        )

        semanal_cand = (d_cand.groupby("semana")
                        .agg(tweets=("tweet_id", "count"),
                             likes=("like_count", "sum"))
                        .reset_index()
                        .assign(fuente="Candidato"))

        semanal_op = (d_op.groupby("semana")
                      .agg(tweets=("tweet_id", "count"),
                           likes=("like_count", "sum"))
                      .reset_index()
                      .assign(fuente="Ciudadanía"))

        semanal_merged = pd.concat([semanal_cand, semanal_op], ignore_index=True)

        metrica_cruce = st.radio(
            "Métrica temporal",
            ["tweets", "likes"],
            format_func=lambda x: {"tweets": "Volumen (tweets)", "likes": "Engagement (likes)"}[x],
            horizontal=True,
            key=f"radio_cruce_{candidato}",
        )

        color_map_cruce = {"Candidato": color, "Ciudadanía": "#7B8FA1"}

        fig_cruce = px.line(
            semanal_merged,
            x="semana", y=metrica_cruce,
            color="fuente",
            color_discrete_map=color_map_cruce,
            markers=True,
            title=f"Actividad semanal: tweets propios vs conversación ciudadana",
            labels={
                "semana": "Semana",
                metrica_cruce: metrica_cruce.capitalize(),
                "fuente": "",
            },
        )
        fig_cruce.update_layout(height=380, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_cruce, use_container_width=True)

        nota_lectura(
            "Cuando la línea ciudadana supera a la del candidato, hay más gente hablando "
            "de él que él publicando. Si las dos líneas se mueven juntas, su comunicación "
            "arrastra la conversación. Si se mueven en sentido contrario, hay eventos "
            "externos que generan conversación independientemente de su estrategia."
        )

    # ── TIPO DE INTERACCIÓN CIUDADANA ─────────────────────────────────────
    if not d_op.empty and "tipo_query" in d_op.columns:
        st.markdown("#### 🔍 ¿Cómo llega la gente a hablar del candidato?")
        st.markdown(
            "<p style='color:#666;font-size:0.9em;margin-top:-6px;margin-bottom:16px'>"
            "Distribución por tipo de búsqueda que capturó cada tweet ciudadano: "
            "si predominan las menciones directas (@), la gente le habla a él; "
            "si predominan los hashtags, usa sus etiquetas de campaña; "
            "si predomina la búsqueda por nombre, habla de él sin etiquetarlo."
            "</p>",
            unsafe_allow_html=True,
        )

        LABELS_TIPO = {
            "mencion_directa":    "Mención directa (@)",
            "mencion_directa_t1": "Mención directa (@) — 1a mitad",
            "mencion_directa_t2": "Mención directa (@) — 2a mitad",
            "hashtag_campania":   "Hashtag de campaña",
            "hashtag_t1":         "Hashtag campaña — T1",
            "hashtag_t2":         "Hashtag campaña — T2",
            "hashtag_t3":         "Hashtag campaña — T3",
            "hashtag_t4":         "Hashtag campaña — T4",
            "busqueda_nombre":    "Búsqueda por nombre",
            "opinion_contextual": "Opinión contextual",
        }

        tipo_counts = (d_op["tipo_query"]
                       .map(LABELS_TIPO)
                       .fillna(d_op["tipo_query"])
                       .value_counts()
                       .reset_index())
        tipo_counts.columns = ["tipo", "n"]

        # Agrupar categorías para simplificar el gráfico
        def agrupar_tipo(t):
            if "Mención"  in t: return "Mención directa (@)"
            if "Hashtag"  in t: return "Hashtag de campaña"
            if "nombre"   in t: return "Búsqueda por nombre"
            if "contextual" in t: return "Opinión contextual"
            return t

        tipo_counts["tipo_agrupado"] = tipo_counts["tipo"].apply(agrupar_tipo)
        tipo_agg = (tipo_counts.groupby("tipo_agrupado")["n"]
                    .sum().reset_index()
                    .sort_values("n", ascending=False))

        colores_tipo_op = {
            "Mención directa (@)":  "#4C72B0",
            "Hashtag de campaña":   "#DD8452",
            "Búsqueda por nombre":  "#55A868",
            "Opinión contextual":   "#C44E52",
        }

        col_pie1, col_pie2 = st.columns([1, 1])
        with col_pie1:
            fig_tipo_op = px.pie(
                tipo_agg,
                values="n",
                names="tipo_agrupado",
                title="¿Cómo encontramos estos tweets?",
                color="tipo_agrupado",
                color_discrete_map=colores_tipo_op,
            )
            fig_tipo_op.update_traces(textposition="inside", textinfo="percent+label")
            fig_tipo_op.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_tipo_op, use_container_width=True)

        with col_pie2:
            # Top 5 tweets ciudadanos más virales
            top_op = (d_op.sort_values("like_count", ascending=False)
                      .head(3).reset_index(drop=True))
            if not top_op.empty:
                st.markdown("**🔥 Tweets ciudadanos más virales sobre este candidato**")
                for _, row in top_op.iterrows():
                    with st.container(border=True):
                        col_a, col_b, col_c = st.columns([3, 1, 1])
                        autor = row.get("autor_username", row.get("author_username", "—"))
                        col_a.markdown(f"**@{autor}**")
                        col_b.metric("❤️", fmt(row["like_count"]))
                        col_c.metric("🔁", fmt(row["retweet_count"]))
                        st.markdown(
                            f"> {str(row['text'])[:160]}"
                            f"{'…' if len(str(row['text'])) > 160 else ''}"
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
        "Hallazgos principales calculados a partir de los datos de este país."
    )
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        insight_box(f"**Mayor eficiencia:** {top_ef['candidato_nombre']} con {fmt(top_ef['likes_x_tweet'])} likes/tweet")
    with col_i2:
        insight_box(f"**Brecha de eficiencia:** {top_ef['candidato_nombre'].split()[0]} genera {ratio:.1f}x más likes/tweet que {low_ef['candidato_nombre'].split()[0]}")
    with col_i3:
        insight_box(f"**Mayor volumen:** {top_vol['candidato_nombre'].split()[0]} con {fmt(top_vol['tweets'])} tweets capturados")

    st.divider()

    # ── Engagement total ───────────────────────────────────────────────────
    seccion(
        "Engagement total",
        "Suma acumulada de likes y views generados por cada candidato durante todo el período analizado."
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
        "Los likes indican aprobación activa. Los views miden alcance pasivo. "
        "Una brecha grande entre views y likes puede indicar contenido que se consume pero no convence."
    )

    st.divider()

    # ── Eficiencia ─────────────────────────────────────────────────────────
    seccion(
        "Eficiencia por tweet",
        "Promedio de likes y views por cada tweet publicado. Elimina el efecto del volumen."
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
        "Cuando el ranking de eficiencia no coincide con el de engagement total, "
        "el candidato más activo no es el que mejor conecta por publicación."
    )

    st.divider()

    # ── Evolución temporal ─────────────────────────────────────────────────
    seccion(
        "Evolución temporal",
        "Actividad y respuesta semana a semana. Permite identificar picos vinculados a debates o eventos."
    )
    semanal = (data.groupby(["semana","candidato_nombre"])
               .agg(tweets=("tweet_id","count"),
                    likes=("like_count","sum"),
                    views=("view_count","sum"))
               .reset_index())

    metrica_temp = st.radio(
        "Métrica a visualizar", ["likes","tweets","views"],
        horizontal=True, key=f"radio_temp_{pais}",
    )
    fig_temp = px.line(
        semanal, x="semana", y=metrica_temp,
        color="candidato_nombre", color_discrete_map=COLORES_CANDIDATO,
        markers=True, title=f"{metrica_temp.capitalize()} por semana — {pais}",
        labels={"semana":"Semana", metrica_temp:metrica_temp.capitalize(), "candidato_nombre":"Candidato"},
    )
    fig_temp.update_layout(legend_title="Candidato", height=400)
    st.plotly_chart(fig_temp, use_container_width=True)
    nota_lectura("Busca picos simultáneos — suelen coincidir con debates electorales.")

    st.divider()

    # ── Tipo de tweet ──────────────────────────────────────────────────────
    seccion(
        "Composición de tweets",
        "Desglose por tipo: originales, replies, retweets y quotes. Revela la estrategia conversacional."
    )
    tipos_cand = (data_all.groupby(["candidato_nombre","tipo"]).size().reset_index(name="n"))
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

    nota_lectura("Alto porcentaje de replies = candidato dialógico. Alto de originales = broadcast.")

    st.divider()

    # ── Hashtags ───────────────────────────────────────────────────────────
    seccion(
        "Hashtags más usados",
        "Etiquetas temáticas que posicionan el mensaje del candidato en conversaciones más amplias."
    )
    col7, col8 = st.columns([1, 2])
    with col7:
        candidato_sel = st.selectbox("Ver hashtags de:", ["Todos"] + candidatos, key=f"ht_sel_{pais}")
    with col8:
        n_tags = st.slider("N° de hashtags", 5, 20, 10, key=f"ht_n_{pais}")

    ht_data = data if candidato_sel == "Todos" else data[data["candidato_nombre"] == candidato_sel]
    ht_df = top_hashtags(ht_data["hashtags"], n=n_tags)
    if not ht_df.empty:
        color_ht = COLORES_PAIS[pais] if candidato_sel == "Todos" else COLORES_CANDIDATO.get(candidato_sel, "#1F77B4")
        fig_ht = px.bar(
            ht_df.sort_values("frecuencia"),
            x="frecuencia", y="hashtag", orientation="h", text="frecuencia",
            title=f"Top {n_tags} hashtags — {candidato_sel}",
            labels={"frecuencia":"Frecuencia","hashtag":""},
            color_discrete_sequence=[color_ht],
        )
        fig_ht.update_traces(textposition="outside")
        fig_ht.update_layout(height=max(300, n_tags * 32))
        st.plotly_chart(fig_ht, use_container_width=True)

    st.divider()

    # ── Top tweets ─────────────────────────────────────────────────────────
    seccion(
        "Tweets más virales",
        "Los tweets con mayor impacto del período. Muestran qué contenido conectó con la audiencia."
    )
    col9, col10 = st.columns([1, 2])
    with col9:
        cand_top = st.selectbox("Filtrar por candidato:", ["Todos"] + candidatos, key=f"top_sel_{pais}")
    with col10:
        metrica_top = st.radio(
            "Ordenar por:",
            ["like_count","view_count","retweet_count"],
            format_func=lambda x: {"like_count":"Likes","view_count":"Views","retweet_count":"Retweets"}[x],
            horizontal=True, key=f"top_met_{pais}",
        )

    top_data = data if cand_top == "Todos" else data[data["candidato_nombre"] == cand_top]
    top5 = top_data.sort_values(metrica_top, ascending=False).head(5).reset_index(drop=True)
    for _, row in top5.iterrows():
        with st.container(border=True):
            ca, cb, cc, cd = st.columns([2, 1, 1, 1])
            ca.markdown(f"**{row['candidato_nombre']}**  \n🗓️ {str(row['created_at'])[:10]}")
            cb.metric("❤️ Likes",  fmt(row["like_count"]))
            cc.metric("🔁 RTs",    fmt(row["retweet_count"]))
            cd.metric("👁️ Views",  fmt(row["view_count"]))
            st.markdown(f"> {str(row['text'])[:220]}{'…' if len(str(row['text'])) > 220 else ''}")
            st.markdown(f"[Ver tweet en X ↗]({row['tweet_url']})")

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN DE OPINIÓN CIUDADANA POR CANDIDATO
    # ════════════════════════════════════════════════════════════════════════
    if HAY_OPINION:
        st.divider()
        st.markdown(
            f"""
            <div style='background:linear-gradient(90deg,#f0f4ff 0%,#f8f8f8 100%);
                        border-radius:10px;padding:18px 22px;margin-bottom:6px;
                        border:1px solid #e0e6f0'>
                <h3 style='margin:0 0 4px 0;font-size:1.2em'>
                    🗣️ Selecciona un candidato para ver qué dice la gente sobre él
                </h3>
                <p style='margin:0;color:#666;font-size:0.9em'>
                    Cada análisis muestra el contraste entre la comunicación del candidato 
                    y la conversación que genera en la ciudadanía.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        candidato_op = st.selectbox(
            "Ver opinión ciudadana de:",
            candidatos,
            key=f"op_sel_{pais}",
        )
        color_sel = COLORES_CANDIDATO.get(candidato_op, "#1F77B4")
        render_scorecard_opinion(candidato_op, color_sel)


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
    Incluye dos capas de análisis: la **comunicación propia** de cada candidato y la 
    **opinión ciudadana** que generan en la plataforma.
    """)
    st.markdown(NOTA_METODOLOGICA)
    st.divider()

    # ── KPIs combinados ────────────────────────────────────────────────────
    seccion(
        "Totales capturados",
        "Resumen de ambas capas de datos: tweets propios de candidatos y tweets ciudadanos."
    )

    if HAY_OPINION:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Tweets propios",       fmt(len(df_prop)))
        c2.metric("Tweets ciudadanos",    fmt(len(df_op)))
        c3.metric("Likes (propios)",      fmt(df_prop["like_count"].sum()))
        c4.metric("Likes (ciudadanos)",   fmt(df_op["like_count"].sum()))
        c5.metric("Autores únicos",       fmt(df_op["autor_username"].nunique() if "autor_username" in df_op.columns else 0))
        c6.metric("Candidatos",           df["candidato_nombre"].nunique())
    else:
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

    seccion("📌 Claves del período", "Tres hallazgos destacados del dataset.")
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        insight_box(f"**Mayor eficiencia:** {top_ef['candidato_nombre']} ({fmt(top_ef['likes_x_tweet'])} likes/tweet)")
    with col_i2:
        insight_box(f"**Mayor volumen:** {top_vol['candidato_nombre']} ({fmt(top_vol['tweets'])} tweets)")
    with col_i3:
        insight_box(f"**Mayor alcance:** {top_vw['candidato_nombre']} ({fmt(top_vw['views'])} views totales)")

    # ── Insight de resonancia global ──────────────────────────────────────
    if HAY_OPINION:
        st.divider()
        seccion(
            "🗣️ Resonancia ciudadana global",
            "Comparación entre la actividad propia de cada candidato y el volumen "
            "de conversación que genera entre la ciudadanía."
        )

        col_cand_op = "candidato_nombre" if "candidato_nombre" in df_op.columns else "candidato"
        res_op = (df_op.groupby(col_cand_op)
                  .agg(tweets_ciudadanos=("tweet_id","count"),
                       likes_ciudadanos=("like_count","sum"))
                  .reset_index()
                  .rename(columns={col_cand_op: "candidato_nombre"}))

        res_prop = (df_prop.groupby("candidato_nombre")
                    .agg(tweets_propios=("tweet_id","count"),
                         likes_propios=("like_count","sum"),
                         pais=("candidato_pais","first"))
                    .reset_index())

        res_merged = res_prop.merge(res_op, on="candidato_nombre", how="left").fillna(0)
        res_merged["ratio_resonancia"] = (
            res_merged["tweets_ciudadanos"] / res_merged["tweets_propios"].clip(lower=1)
        ).round(1)
        res_merged = res_merged.sort_values("ratio_resonancia", ascending=False)

        fig_res = px.bar(
            res_merged,
            x="candidato_nombre", y="ratio_resonancia",
            color="pais", color_discrete_map=COLORES_PAIS,
            text="ratio_resonancia",
            title="Ratio de resonancia: tweets ciudadanos / tweets propios",
            labels={"candidato_nombre":"","ratio_resonancia":"Tweets ciudadanos por tweet propio","pais":"País"},
        )
        fig_res.update_traces(texttemplate="%{text:.1f}x", textposition="outside")
        fig_res.update_layout(xaxis_tickangle=-20, height=420)
        fig_res.add_hline(y=1, line_dash="dot", line_color="gray",
                          annotation_text="Paridad (1 tweet ciudadano por tweet propio)")
        st.plotly_chart(fig_res, use_container_width=True)
        nota_lectura(
            "Por encima de 1x: la ciudadanía genera más conversación de la que el "
            "candidato publica. Por debajo de 1x: el candidato publica más de lo que "
            "la gente habla de él. La línea punteada marca la paridad."
        )

    st.divider()

    # ── Wordcloud ─────────────────────────────────────────────────────────
    seccion(
        "¿De qué habla cada candidato?",
        "Términos más frecuentes en los tweets propios de cada candidato (stopwords eliminadas)."
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

    candidatos_lista = sorted(df_prop["candidato_nombre"].unique().tolist())
    cand_wc = st.selectbox("Selecciona un candidato", candidatos_lista, key="wc_candidato")

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
            width=900, height=420, background_color="white",
            color_func=lambda *args, **kwargs: color,
            stopwords=STOPWORDS_ES, min_word_length=4,
            max_words=80, collocations=False, prefer_horizontal=0.85,
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
    nota_lectura("Compara las nubes de distintos candidatos para identificar diferencias temáticas.")

    st.divider()

    # ── Tabla de contexto electoral ────────────────────────────────────────
    seccion("Contexto electoral y cobertura", "Condiciones de cada candidato en el análisis.")
    contexto = (df.groupby(["candidato_pais","candidato_nombre","candidato_vuelta","periodo_busqueda"])
                .agg(
                    tweets_capturados = ("tweet_id","count"),
                    likes_totales     = ("like_count","sum"),
                    views_totales     = ("view_count","sum"),
                    seguidores_hoy    = ("author_followers","first"),
                )
                .reset_index()
                .sort_values(["candidato_pais","candidato_nombre"])
                .rename(columns={
                    "candidato_pais":"País","candidato_nombre":"Candidato",
                    "candidato_vuelta":"Vuelta","periodo_busqueda":"Período",
                    "tweets_capturados":"Tweets","likes_totales":"Likes",
                    "views_totales":"Views","seguidores_hoy":"Seguidores (hoy)",
                }))
    for col in ["Likes","Views","Seguidores (hoy)"]:
        contexto[col] = contexto[col].apply(lambda x: f"{int(x):,}")
    st.dataframe(contexto, use_container_width=True, hide_index=True)

    st.divider()

    # ── Cobertura por país ─────────────────────────────────────────────────
    seccion("Cobertura por país", "Volumen total de tweets propios y views por país.")
    tweets_pais = (df_prop.groupby("candidato_pais")
                   .agg(tweets=("tweet_id","count"), likes=("like_count","sum"), views=("view_count","sum"))
                   .reset_index())
    col_a, col_b = st.columns(2)
    with col_a:
        fig_tp = px.bar(
            tweets_pais, x="candidato_pais", y="tweets",
            color="candidato_pais", color_discrete_map=COLORES_PAIS,
            text="tweets", title="Tweets propios por país",
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

    nota_lectura("Un número alto de tweets no implica mayor impacto.")

    st.divider()

    # ── Volumen vs Eficiencia ──────────────────────────────────────────────
    seccion(
        "Volumen vs Eficiencia · comparación multi-país",
        "El único gráfico que permite comparar candidatos entre países, normalizando por likes/tweet."
    )
    fig_burbuja = px.scatter(
        met_global,
        x="tweets", y="likes_x_tweet",
        size="views", color="candidato_nombre",
        color_discrete_map=COLORES_CANDIDATO,
        symbol="candidato_pais",
        hover_name="candidato_nombre",
        hover_data={"candidato_pais":True,"tweets":True,"likes_x_tweet":True,"views_x_tweet":True,"views":False},
        title="Volumen (tweets) vs Eficiencia (likes/tweet)",
        labels={"tweets":"N° tweets","likes_x_tweet":"Likes promedio por tweet","candidato_pais":"País"},
        size_max=60,
    )
    fig_burbuja.update_layout(legend_title="Candidato", height=480)
    st.plotly_chart(fig_burbuja, use_container_width=True)
    nota_lectura(
        "El eje X muestra cuánto publica; el eje Y, qué tan bien responde la audiencia. "
        "Tamaño de burbuja = alcance total en views."
    )


# ── TABS DE PAÍS ──────────────────────────────────────────────────────────────
with tab_chile:
    st.title("🇨🇱 Chile — Elecciones presidenciales 2025")
    st.caption("Período analizado: 17 sept – 14 dic 2025 · 1ra y 2da vuelta")
    st.markdown("""
    Chile celebró elecciones presidenciales en dos vueltas. El análisis cubre desde
    el inicio de la campaña oficial hasta el balotaje. Se incluyen cinco candidatos.
    Al final de cada sección encontrarás el análisis de **opinión ciudadana** para
    ver qué dice la gente sobre cada candidato.
    """)
    render_pais("Chile")

with tab_bolivia:
    st.title("🇧🇴 Bolivia — Elecciones presidenciales 2025")
    st.caption("Período analizado: 13 jul – 19 oct 2025 · 2da vuelta")
    st.markdown("""
    El análisis de Bolivia se concentra en la segunda vuelta electoral,
    comparando los dos candidatos finalistas.
    """)
    render_pais("Bolivia")

with tab_ecuador:
    st.title("🇪🇨 Ecuador — Elecciones presidenciales 2025")
    st.caption("Período analizado: 5 ene – 13 abr 2025 · 2da vuelta")
    st.markdown("""
    Ecuador fue el primer país en celebrar elecciones en 2025. Segunda vuelta entre 
    Daniel Noboa (incumbente) y Luisa González. El contexto de incumbencia es relevante 
    para interpretar las diferencias de alcance.
    """)
    render_pais("Ecuador")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 · PERFIL DE CANDIDATO
# ════════════════════════════════════════════════════════════════════════════
with tab_candidato:
    st.title("🧠 Perfil de candidato")
    st.markdown("""
    Análisis individual y detallado. Incluye clasificación de estrategia comunicacional,
    evolución semanal, distribución de engagement, hashtags y tweets más virales.
    Al final, el análisis completo de **opinión ciudadana** sobre ese candidato.
    """)

    cand = st.selectbox("Selecciona un candidato", df["candidato_nombre"].unique())

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
            "**Estrategia dominante:** publica con alta frecuencia y cada publicación genera respuesta considerable.",
        ("alto volumen",  "engagement moderado"):
            "**Estrategia de presencia:** alta visibilidad pero impacto promedio inferior a la media. Cantidad sobre calidad.",
        ("bajo volumen",  "alto engagement"):
            "**Estrategia selectiva:** publica con moderación pero cada tweet resuena más que el promedio.",
        ("bajo volumen",  "engagement moderado"):
            "**Estrategia limitada:** baja frecuencia y respuesta inferior al promedio del país.",
    }

    seccion("📌 Perfil comunicacional", "Clasificación automática basada en volumen y engagement.")
    insight_box(
        f"**{cand}** · {pais_c} · {data_c['candidato_vuelta'].iloc[0]}  \n"
        f"**{tweets} tweets propios** · **{int(lpt):,} likes/tweet** · **{int(vpt):,} views/tweet**  \n"
        f"Clasificación: **{vol_label}** · **{eng_label}**  \n\n"
        f"{estrategia.get((vol_label, eng_label), '')}"
    )

    st.divider()

    # Evolución temporal
    seccion("Evolución semanal", "Actividad semana a semana del candidato.")
    semanal_c = (data_c.groupby("semana")
                 .agg(tweets=("tweet_id","count"), likes=("like_count","sum"), views=("view_count","sum"))
                 .reset_index())
    metrica_c = st.radio("Métrica", ["likes","tweets","views"], horizontal=True, key="radio_cand")
    color_c = COLORES_CANDIDATO.get(cand, "#1F77B4")
    fig_ev = px.area(
        semanal_c, x="semana", y=metrica_c,
        title=f"Evolución semanal — {cand}",
        labels={"semana":"Semana", metrica_c:metrica_c.capitalize()},
        color_discrete_sequence=[color_c],
    )
    st.plotly_chart(fig_ev, use_container_width=True)
    nota_lectura("Los picos pueden corresponder a debates o momentos de controversia.")

    st.divider()

    # Composición y distribución
    seccion("Composición y distribución de engagement", "Tipo de publicación e histograma de likes.")
    col_a, col_b = st.columns(2)
    with col_a:
        tipos_c = data_c_all["tipo"].value_counts().reset_index()
        tipos_c.columns = ["tipo","n"]
        fig_pie_c = px.pie(tipos_c, values="n", names="tipo", title="Tipos de publicación",
                           color_discrete_map=COLORES_TIPO)
        st.plotly_chart(fig_pie_c, use_container_width=True)
    with col_b:
        fig_hist = px.histogram(data_c, x="like_count", nbins=20,
                                title="Distribución de likes por tweet",
                                labels={"like_count":"Likes por tweet","count":"N° tweets"},
                                color_discrete_sequence=[color_c])
        st.plotly_chart(fig_hist, use_container_width=True)
    nota_lectura("Cola larga en el histograma = promedio inflado por pocos tweets virales.")

    st.divider()

    # Hashtags
    seccion("Hashtags más usados", "Ejes de campaña y temáticas posicionadas por este candidato.")
    n_ht = st.slider("N° de hashtags", 5, 20, 10, key="ht_cand")
    ht_c = top_hashtags(data_c["hashtags"], n=n_ht)
    if not ht_c.empty:
        fig_ht_c = px.bar(
            ht_c.sort_values("frecuencia"),
            x="frecuencia", y="hashtag", orientation="h", text="frecuencia",
            title=f"Top {n_ht} hashtags de {cand}",
            labels={"frecuencia":"Frecuencia","hashtag":""},
            color_discrete_sequence=[color_c],
        )
        fig_ht_c.update_traces(textposition="outside")
        fig_ht_c.update_layout(height=max(300, n_ht * 32))
        st.plotly_chart(fig_ht_c, use_container_width=True)

    st.divider()

    # Top tweets propios
    seccion("🔥 Tweets más virales", "Los tweets con mayor impacto del período.")
    metrica_tv = st.radio(
        "Ordenar por:",
        ["like_count","view_count","retweet_count"],
        format_func=lambda x: {"like_count":"Likes","view_count":"Views","retweet_count":"Retweets"}[x],
        horizontal=True, key="met_top_cand",
    )
    n_top = st.slider("N° de tweets", 3, 10, 5, key="n_top_cand")
    top_c = data_c.sort_values(metrica_tv, ascending=False).head(n_top).reset_index(drop=True)
    for _, row in top_c.iterrows():
        with st.container(border=True):
            ca, cb, cc, cd = st.columns([2, 1, 1, 1])
            ca.markdown(f"🗓️ **{str(row['created_at'])[:10]}**")
            cb.metric("❤️ Likes",  fmt(row["like_count"]))
            cc.metric("🔁 RTs",    fmt(row["retweet_count"]))
            cd.metric("👁️ Views",  fmt(row["view_count"]))
            st.markdown(f"> {str(row['text'])[:220]}{'…' if len(str(row['text'])) > 220 else ''}")
            st.markdown(f"[Ver tweet en X ↗]({row['tweet_url']})")

    # ── OPINIÓN CIUDADANA EN PERFIL ────────────────────────────────────────
    render_scorecard_opinion(cand, color_c)
