# app.py
# Ejecutar con: streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import re
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import glob

# ── CONFIG ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Observatorio LATAM 2025",
    page_icon="🗳️",
    layout="wide"
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

COLORES_PAIS = {
    "Chile":   "#D62728",
    "Bolivia": "#2CA02C",
    "Ecuador": "#1F77B4",
}

# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/tweets_consolidado_20260328_204225.csv")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["semana"] = (
        df["created_at"]
        .dt.to_period("W")
        .dt.start_time
        .dt.tz_localize(None)
    )
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
        df_op["semana"] = (
            df_op["created_at"]
            .dt.to_period("W")
            .dt.start_time
            .dt.tz_localize(None)
        )
        return df_op
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_perfiles():
    # Busca cualquier archivo perfiles_20260328_204225.csv en /data
    archivos = glob.glob("data/perfiles_*.csv")
    if not archivos:
        return pd.DataFrame()
    return pd.read_csv(sorted(archivos)[-1])

df       = load_data()
df_prop  = df[~df["is_retweet"]].copy()
df_op    = load_opinion()
df_perf  = load_perfiles()
HAY_OPINION  = not df_op.empty
HAY_PERFILES = not df_perf.empty

# ── HELPERS ────────────────────────────────────────────────────────────────────
def fmt(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.0f}k"
    return str(int(n))

def top_hashtags(series, n=10):
    tags = []
    for s in series.dropna():
        tags += [x.strip().lower() for x in str(s).split("|") if x.strip()]
    return pd.DataFrame(Counter(tags).most_common(n), columns=["hashtag", "freq"])

def metricas(data):
    return (
        data.groupby("candidato_nombre")
        .agg(
            tweets = ("tweet_id",   "count"),
            likes  = ("like_count", "sum"),
            views  = ("view_count", "sum"),
        )
        .reset_index()
        .assign(likes_x_tweet=lambda d: (d.likes / d.tweets).round(0).astype(int))
    )

def nota(texto):
    st.markdown(
        f"<div style='background:#f0f2f6;border-left:3px solid #4a90d9;"
        f"padding:8px 12px;border-radius:4px;font-size:0.87em;color:#444;"
        f"margin-bottom:10px'>💡 {texto}</div>",
        unsafe_allow_html=True,
    )

STOPWORDS_ES = {
    "de","la","el","en","y","a","los","del","se","las","por","un","para",
    "con","una","su","al","lo","como","más","pero","sus","le","ya","o",
    "este","sí","porque","esta","entre","cuando","muy","sin","sobre",
    "también","me","hasta","hay","donde","han","yo","él","ella","nos",
    "todo","estos","estas","fue","son","ser","tiene","tenemos","que","es",
    "no","si","te","mi","tu","http","https","t","co","amp","rt","via",
    "hoy","ayer","así","bien","gran","cada","hacer","puede","nuestro",
    "nuestra","nuestros","nuestras","está","están","tener","solo","todos",
    "todas","otro","otra","años","Chile","Bolivia","Ecuador","país",
    "gobierno","presidente","presidenta","hemos","sido","ser","esto",
}

@st.cache_data
def generar_wordcloud(candidato: str, color: str) -> BytesIO:
    textos = df_prop[df_prop["candidato_nombre"] == candidato]["text"].dropna()
    def limpiar(t):
        t = re.sub(r"http\S+", "", t)
        t = re.sub(r"@\w+", "", t)
        t = re.sub(r"#(\w+)", r"\1", t)
        t = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", t)
        return t.lower()
    corpus = " ".join(limpiar(t) for t in textos)
    wc = WordCloud(
        width=700, height=320, background_color="white",
        color_func=lambda *a, **k: color,
        stopwords=STOPWORDS_ES, min_word_length=4,
        max_words=60, collocations=False, prefer_horizontal=0.85,
    )
    wc.generate(corpus)
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN: ¿QUÉ DICE LA GENTE?
# ══════════════════════════════════════════════════════════════════════════════
def render_opinion(pais: str):
    """
    Sección independiente — no compara con datos propios del candidato.
    Son muestras, útiles para patrones temáticos y textos, no para volúmenes absolutos.
    """
    if not HAY_OPINION:
        return

    col_nombre = "candidato_nombre" if "candidato_nombre" in df_op.columns else "candidato"
    data_op = df_op[df_op["candidato_pais"] == pais].copy()
    if data_op.empty:
        return

    st.divider()
    st.subheader("🗣️ ¿Y qué dice la gente?")
    st.caption(
        "Muestra de tweets de ciudadanos que mencionaron o hablaron de los candidatos "
        "durante el período. No es exhaustiva — sirve para identificar temas y conversaciones, "
        "no para estimar volúmenes totales de menciones."
    )

    candidatos_op = sorted(data_op[col_nombre].unique().tolist())
    candidato_sel = st.selectbox(
        "Ver conversación sobre:", candidatos_op, key=f"op_{pais}"
    )
    d     = data_op[data_op[col_nombre] == candidato_sel]
    color = COLORES_CANDIDATO.get(candidato_sel, COLORES_PAIS[pais])

    # ── KPIs de la muestra ─────────────────────────────────────────────────
    col_autor = "autor_username" if "autor_username" in d.columns else "author_username"
    c1, c2, c3 = st.columns(3)
    c1.metric("Tweets en la muestra",  fmt(len(d)))
    c2.metric("Personas distintas",
              fmt(d[col_autor].nunique()) if col_autor in d.columns else "—")
    c3.metric("Likes acumulados",      fmt(d["like_count"].sum()))

    nota("Estos números reflejan la muestra capturada, no el universo total de menciones en X.")

    # ── Evolución semanal de la conversación ──────────────────────────────
    st.markdown("**¿Cuándo habló más la gente de este candidato?**")
    semanal_op = (
        d.groupby("semana")
        .agg(tweets=("tweet_id", "count"))
        .reset_index()
    )
    fig_ev = px.area(
        semanal_op, x="semana", y="tweets",
        labels={"semana": "Semana", "tweets": "Tweets capturados"},
        color_discrete_sequence=[color],
    )
    fig_ev.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10))
    st.plotly_chart(fig_ev, use_container_width=True)
    nota(
        "Los picos indican semanas con mayor conversación ciudadana. "
        "Pueden coincidir con debates, polémicas o hitos electorales."
    )

    # ── Hashtags ciudadanos ────────────────────────────────────────────────
    if "hashtags" in d.columns:
        st.markdown("**¿Con qué hashtags habla la gente de este candidato?**")
        ht_op = top_hashtags(d["hashtags"], n=8)
        if not ht_op.empty:
            fig_ht = px.bar(
                ht_op.sort_values("freq"),
                x="freq", y="hashtag", orientation="h",
                text="freq",
                labels={"freq": "Usos", "hashtag": ""},
                color_discrete_sequence=[color],
            )
            fig_ht.update_traces(textposition="outside")
            fig_ht.update_layout(showlegend=False, height=280, margin=dict(t=10))
            st.plotly_chart(fig_ht, use_container_width=True)

    # ── Tweets ciudadanos más virales ──────────────────────────────────────
    st.markdown("**Tweets ciudadanos con más likes sobre este candidato**")
    col_rt  = "retweet_count" if "retweet_count" in d.columns else None
    top_op  = d.sort_values("like_count", ascending=False).head(3).reset_index(drop=True)

    for _, row in top_op.iterrows():
        with st.container(border=True):
            autor = row.get("autor_username", row.get("author_username", "—"))
            ca, cb, cc = st.columns([4, 1, 1])
            ca.markdown(f"**@{autor}** · {str(row['created_at'])[:10]}")
            cb.metric("❤️", fmt(row["like_count"]))
            cc.metric("🔁", fmt(row[col_rt]) if col_rt else "—")
            st.markdown(
                f"> {str(row['text'])[:200]}{'…' if len(str(row['text'])) > 200 else ''}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN REUTILIZABLE POR PAÍS
# ══════════════════════════════════════════════════════════════════════════════
def render_pais(pais: str):
    data = df_prop[df_prop["candidato_pais"] == pais].copy()
    met  = metricas(data)
    candidatos = met["candidato_nombre"].tolist()

    # ── KPIs ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Tweets propios", fmt(len(data)))
    c2.metric("Likes totales",  fmt(data["like_count"].sum()))
    c3.metric("Views totales",  fmt(data["view_count"].sum()))

    st.divider()

    # ── Engagement y eficiencia ────────────────────────────────────────────
    st.subheader("Engagement y eficiencia")
    nota(
        "Izquierda: impacto bruto acumulado. Derecha: rendimiento promedio por tweet. "
        "Cuando el ranking difiere, el candidato más activo no es el que mejor conecta."
    )
    col1, col2 = st.columns(2)

    with col1:
        fig_likes = px.bar(
            met.sort_values("likes"),
            x="likes", y="candidato_nombre", orientation="h",
            color="candidato_nombre", color_discrete_map=COLORES_CANDIDATO,
            text="likes", title="Likes totales",
            labels={"likes": "Likes", "candidato_nombre": ""},
        )
        fig_likes.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_likes.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_likes, use_container_width=True)

    with col2:
        fig_lpt = px.bar(
            met.sort_values("likes_x_tweet"),
            x="likes_x_tweet", y="candidato_nombre", orientation="h",
            color="candidato_nombre", color_discrete_map=COLORES_CANDIDATO,
            text="likes_x_tweet", title="Likes por tweet",
            labels={"likes_x_tweet": "Likes/tweet", "candidato_nombre": ""},
        )
        fig_lpt.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_lpt.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_lpt, use_container_width=True)

    st.divider()

    # ── Evolución temporal ─────────────────────────────────────────────────
    st.subheader("Evolución semanal")
    nota("Busca picos simultáneos — suelen coincidir con debates o eventos clave del período.")

    semanal = (
        data.groupby(["semana", "candidato_nombre"])
        .agg(likes=("like_count","sum"), tweets=("tweet_id","count"))
        .reset_index()
    )
    metrica_temp = st.radio(
        "Ver por:",
        ["likes", "tweets"],
        format_func=lambda x: {"likes": "Likes", "tweets": "Volumen de tweets"}[x],
        horizontal=True,
        key=f"temp_{pais}",
    )
    fig_temp = px.line(
        semanal, x="semana", y=metrica_temp,
        color="candidato_nombre", color_discrete_map=COLORES_CANDIDATO,
        markers=True,
        labels={
            "semana": "Semana",
            metrica_temp: metrica_temp.capitalize(),
            "candidato_nombre": "",
        },
    )
    fig_temp.update_layout(legend_title="", height=360)
    st.plotly_chart(fig_temp, use_container_width=True)

    st.divider()

    # ── Hashtags ───────────────────────────────────────────────────────────
    st.subheader("Hashtags más usados")
    nota("Revelan los ejes temáticos de campaña y los eventos en que participó activamente.")

    candidato_ht = st.selectbox("Ver hashtags de:", candidatos, key=f"ht_{pais}")
    ht = top_hashtags(
        data[data["candidato_nombre"] == candidato_ht]["hashtags"], n=10
    )
    if not ht.empty:
        color_ht = COLORES_CANDIDATO.get(candidato_ht, COLORES_PAIS[pais])
        fig_ht = px.bar(
            ht.sort_values("freq"),
            x="freq", y="hashtag", orientation="h",
            text="freq",
            labels={"freq": "Usos", "hashtag": ""},
            color_discrete_sequence=[color_ht],
        )
        fig_ht.update_traces(textposition="outside")
        fig_ht.update_layout(showlegend=False, height=max(280, len(ht) * 32))
        st.plotly_chart(fig_ht, use_container_width=True)

    st.divider()

    # ── Top tweets ─────────────────────────────────────────────────────────
    st.subheader("Tweets más virales")
    nota("El contenido con mayor respuesta — en las propias palabras del candidato.")

    candidato_top = st.selectbox(
        "Ver tweets de:", ["Todos"] + candidatos, key=f"top_{pais}"
    )
    top_data = (
        data if candidato_top == "Todos"
        else data[data["candidato_nombre"] == candidato_top]
    )
    top5 = top_data.sort_values("like_count", ascending=False).head(5).reset_index(drop=True)

    for _, row in top5.iterrows():
        with st.container(border=True):
            ca, cb, cc, cd = st.columns([3, 1, 1, 1])
            ca.markdown(f"**{row['candidato_nombre']}** · {str(row['created_at'])[:10]}")
            cb.metric("❤️", fmt(row["like_count"]))
            cc.metric("🔁", fmt(row["retweet_count"]))
            cd.metric("👁️", fmt(row["view_count"]))
            st.markdown(
                f"> {str(row['text'])[:220]}{'…' if len(str(row['text'])) > 220 else ''}"
            )
            if pd.notna(row.get("tweet_url", "")):
                st.markdown(f"[Ver en X ↗]({row['tweet_url']})")

    # ── OPINIÓN CIUDADANA ─────────────────────────────────────────────────
    render_opinion(pais)


# ══════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════
st.title("🗳️ Observatorio LATAM 2025")

tab_general, tab_chile, tab_bolivia, tab_ecuador = st.tabs([
    "🌎 General",
    "🇨🇱 Chile",
    "🇧🇴 Bolivia",
    "🇪🇨 Ecuador",
])

# ── TAB GENERAL ────────────────────────────────────────────────────────────────
with tab_general:
    st.subheader("Resumen general")
    st.markdown(
        "Actividad en Twitter/X de los principales candidatos presidenciales de "
        "Chile, Bolivia y Ecuador durante sus respectivos períodos electorales de 2025."
    )

    # ── KPIs globales ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Países",          3)
    c2.metric("Candidatos",      df["candidato_nombre"].nunique())
    c3.metric("Tweets propios",  fmt(len(df_prop)))
    c4.metric("Likes totales",   fmt(df_prop["like_count"].sum()))
    c5.metric("Views totales",   fmt(df_prop["view_count"].sum()))
    c6.metric("Retweets",        fmt(df_prop["retweet_count"].sum()))

    st.divider()

    # ── Tarjetas de candidatos con foto ───────────────────────────────────
    st.subheader("Los candidatos")
    nota(
        "Foto de perfil oficial de Twitter/X · "
        "Datos capturados durante el período electoral de cada candidato."
    )

    # Preparar métricas por candidato
    met_cands = metricas(df_prop).merge(
        df_prop[["candidato_nombre","candidato_pais","candidato_vuelta","periodo_busqueda"]]
        .drop_duplicates("candidato_nombre"),
        on="candidato_nombre", how="left"
    )

    # Foto de perfil desde perfiles CSV (columna foto_perfil_url)
    foto_col = None
    if HAY_PERFILES:
        for c in ["foto_perfil_url","profilePicture","profile_image_url"]:
            if c in df_perf.columns:
                foto_col = c
                break

    def get_foto(nombre):
        if not HAY_PERFILES or foto_col is None:
            return None
        row = df_perf[df_perf["nombre_real"] == nombre] if "nombre_real" in df_perf.columns \
              else df_perf[df_perf["username"] == nombre]
        if row.empty:
            return None
        url = row.iloc[0][foto_col]
        return url if pd.notna(url) else None

    # Agrupar por país
    for pais in ["Chile", "Bolivia", "Ecuador"]:
        cands_pais = met_cands[met_cands["candidato_pais"] == pais].sort_values(
            "likes", ascending=False
        )
        st.markdown(
            f"<p style='font-weight:600;font-size:1em;color:{COLORES_PAIS[pais]};"
            f"margin-bottom:6px;margin-top:8px'>{pais}</p>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(cands_pais))
        for col, (_, row) in zip(cols, cands_pais.iterrows()):
            with col:
                foto = get_foto(row["candidato_nombre"])
                if foto:
                    st.image(foto, width=72)
                st.markdown(
                    f"**{row['candidato_nombre']}**  \n"
                    f"<span style='font-size:0.8em;color:#666'>"
                    f"{row['candidato_vuelta']}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"❤️ **{fmt(row['likes'])}** likes  \n"
                    f"📄 {fmt(row['tweets'])} tweets  \n"
                    f"⚡ {fmt(row['likes_x_tweet'])} likes/tweet"
                )

    st.divider()

    # ── Volumen vs eficiencia ──────────────────────────────────────────────
    st.subheader("Volumen vs eficiencia")
    nota(
        "Única vista multi-país: normaliza por likes/tweet para comparar "
        "candidatos con contextos electorales distintos. "
        "Eje X = cuánto publica · Eje Y = cuánto resuena · Tamaño = views totales."
    )
    fig_burbuja = px.scatter(
        met_cands,
        x="tweets", y="likes_x_tweet",
        size="views", color="candidato_pais",
        color_discrete_map=COLORES_PAIS,
        symbol="candidato_pais",
        hover_name="candidato_nombre",
        hover_data={
            "candidato_pais": True, "tweets": True,
            "likes_x_tweet": True, "views": False,
        },
        labels={
            "tweets": "N° tweets",
            "likes_x_tweet": "Likes promedio por tweet",
            "candidato_pais": "País",
        },
        size_max=55,
    )
    fig_burbuja.update_layout(height=420, legend_title="País")
    st.plotly_chart(fig_burbuja, use_container_width=True)

    st.divider()

    # ── Tabla resumen ──────────────────────────────────────────────────────
    st.subheader("Tabla de contexto")
    nota("Condiciones de cada candidato: período analizado, vuelta electoral y métricas clave.")

    tabla = met_cands[["candidato_pais","candidato_nombre","candidato_vuelta",
                        "periodo_busqueda","tweets","likes","views","likes_x_tweet"]].copy()
    tabla = tabla.sort_values(["candidato_pais","likes"], ascending=[True, False])
    tabla.columns = ["País","Candidato","Vuelta","Período","Tweets","Likes","Views","Likes/tweet"]
    for col in ["Likes","Views"]:
        tabla[col] = tabla[col].apply(lambda x: f"{int(x):,}")
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    st.divider()

    # ── Tweet más viral global ─────────────────────────────────────────────
    st.subheader("🔥 El tweet más viral del período")
    viral = df_prop.sort_values("like_count", ascending=False).iloc[0]
    with st.container(border=True):
        col_v1, col_v2 = st.columns([3, 1])
        with col_v1:
            st.markdown(
                f"**{viral['candidato_nombre']}** · {viral['candidato_pais']} · "
                f"{str(viral['created_at'])[:10]}"
            )
            st.markdown(f"> {str(viral['text'])[:300]}")
            if pd.notna(viral.get("tweet_url", "")):
                st.markdown(f"[Ver en X ↗]({viral['tweet_url']})")
        with col_v2:
            st.metric("❤️ Likes",  fmt(viral["like_count"]))
            st.metric("🔁 RTs",    fmt(viral["retweet_count"]))
            st.metric("👁️ Views",  fmt(viral["view_count"]))

    st.divider()

    # ── Wordcloud ──────────────────────────────────────────────────────────
    st.subheader("¿De qué habla cada candidato?")
    nota(
        "Términos más frecuentes en los tweets propios. "
        "Tamaño proporcional a la frecuencia. Stopwords, URLs y menciones eliminados."
    )

    candidatos_lista = sorted(df_prop["candidato_nombre"].unique().tolist())
    wc_cand = st.selectbox("Selecciona un candidato:", candidatos_lista, key="wc_home")
    color_wc = COLORES_CANDIDATO.get(wc_cand, "#1F77B4")
    pais_wc  = df_prop[df_prop["candidato_nombre"] == wc_cand]["candidato_pais"].iloc[0]
    n_wc     = len(df_prop[df_prop["candidato_nombre"] == wc_cand])
    st.caption(f"{pais_wc} · {n_wc} tweets analizados")
    buf = generar_wordcloud(wc_cand, color_wc)
    st.image(buf, use_container_width=True)

# ── TABS PAÍS ──────────────────────────────────────────────────────────────────
with tab_chile:
    st.subheader("🇨🇱 Chile — Elecciones presidenciales 2025")
    st.caption("17 sept – 14 dic 2025 · 1ra y 2da vuelta · 5 candidatos")
    render_pais("Chile")

with tab_bolivia:
    st.subheader("🇧🇴 Bolivia — Elecciones presidenciales 2025")
    st.caption("13 jul – 19 oct 2025 · 2da vuelta · 2 candidatos")
    render_pais("Bolivia")

with tab_ecuador:
    st.subheader("🇪🇨 Ecuador — Elecciones presidenciales 2025")
    st.caption("5 ene – 13 abr 2025 · 2da vuelta · 2 candidatos")
    render_pais("Ecuador")
