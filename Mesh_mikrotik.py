import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os


# =========================
# Параметры задачи
# =========================
H = 12.0  # км
HPBW_el = 90.0  # град (-3 dB ширина по месту)
HPBW_az = 90.0  # град (-3 dB ширина по азимуту)
f_GHz = 2.4  # ГГц (FSPL)

# Область/сетка
L = 18.0  # км (полуразмер области отрисовки)
N = 801  # плотность сетки (можно 401 для ускорения)

# Уровни, которые хотим показать
LEVELS_WANTED = np.array([-10.0, -6.0, -3.0])


# =========================
# Вспомогательные функции
# =========================
def wrap_deg(a: np.ndarray) -> np.ndarray:
    """Привести угол (град) к диапазону [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def safe_contours(ax, X, Y, Z, levels, **kwargs):
    """
    Рисует контуры, но:
    - уровни сортирует,
    - выкидывает уровни вне диапазона данных,
    - не падает, если уровни не пересекаются.
    Возвращает QuadContourSet или None.
    """
    zmin = np.nanmin(Z)
    zmax = np.nanmax(Z)

    levels = np.array(levels, dtype=float)
    levels = np.sort(levels)
    levels = levels[(levels >= zmin) & (levels <= zmax)]

    if levels.size == 0:
        print(
            f"[WARN] Нет контуров: уровни вне диапазона данных. Диапазон: [{zmin:.2f}, {zmax:.2f}] dB"
        )
        return None

    # matplotlib иногда ругается, если уровни не строго возрастают — гарантируем
    levels_unique = np.unique(levels)
    if levels_unique.size < 1:
        print("[WARN] Нет уникальных уровней для контуров.")
        return None

    cs = ax.contour(X, Y, Z, levels=levels_unique, **kwargs)
    return cs


# =========================
# Подготовка сетки
# =========================
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)

# Лимиты по месту и квадрат 24×24 км
theta_el_lim = np.deg2rad(HPBW_el / 2.0)  # 45°
theta_az_lim = np.deg2rad(HPBW_az / 2.0)  # 45°
half = H * np.tan(theta_el_lim)  # 12 км

# Углы в двух ортогональных плоскостях (для "прямоугольной" ДН по месту)
theta_x = np.arctan2(X, H)  # рад
theta_y = np.arctan2(Y, H)  # рад

# Азимут точки на земле
phi = (np.degrees(np.arctan2(Y, X)) + 360.0) % 360.0  # 0..360

# =========================
# Модель ДН: cos^m, чтобы на 45° было -3 dB
# cos(45)^m = 0.5 => m=2
# =========================
m_el = 2.0
m_az = 2.0
eps = 1e-12

G_el = (np.cos(theta_x) ** m_el) * (np.cos(theta_y) ** m_el)
G_el = np.where(
    (np.abs(theta_x) <= np.pi / 2) & (np.abs(theta_y) <= np.pi / 2), G_el, 0.0
)

mask_el = (np.abs(theta_x) <= theta_el_lim) & (np.abs(theta_y) <= theta_el_lim)

# =========================
# FSPL по наклонной дальности
# FSPL(dB) = 92.45 + 20log10(f_GHz) + 20log10(R_km)
# =========================
R_km = np.sqrt(H**2 + X**2 + Y**2)
FSPL_dB = 92.45 + 20.0 * np.log10(f_GHz) + 20.0 * np.log10(R_km)

# =========================
# 4 панели по азимуту: 0/90/180/270
# =========================
boresights = np.array([0.0, 90.0, 180.0, 270.0])

P_dB = np.full((boresights.size, N, N), -1e9, dtype=float)  # -inf условно

for i, b in enumerate(boresights):
    dphi_deg = wrap_deg(phi - b)
    dphi = np.deg2rad(dphi_deg)

    mask_az = np.abs(dphi) <= theta_az_lim

    G_az = np.cos(dphi) ** m_az
    G_az = np.where(np.abs(dphi) <= np.pi / 2, G_az, 0.0)

    G = np.where(mask_el & mask_az, G_el * G_az, 0.0)
    G_dB = 10.0 * np.log10(G + eps)

    # Уровень (без абсолютной EIRP-константы): pattern_dB - FSPL_dB
    P_dB[i] = G_dB - FSPL_dB

# =========================
# P_total = max
# =========================
P_total_dB = P_dB.max(axis=0)

# Нормировка к максимуму (удобно для -3/-6/-10 dB)
P_total_rel_dB = P_total_dB - np.nanmax(P_total_dB)

print(
    f"[INFO] Диапазон P_total_rel_dB: {np.nanmin(P_total_rel_dB):.2f} .. {np.nanmax(P_total_rel_dB):.2f} dB"
)


# ---------- строгая обрезка полигона полуплоскостью (Sutherland–Hodgman) ----------
def clip_polygon_halfplane(poly, f, intersect):
    """
    poly: (M,2) вершины (замкнутый не нужен)
    f(p): True если точка внутри
    intersect(p1,p2): точка пересечения отрезка с границей
    """
    if poly is None or len(poly) == 0:
        return np.empty((0, 2), dtype=float)

    out = []
    n = len(poly)
    for i in range(n):
        A = poly[i]
        B = poly[(i + 1) % n]
        Ain = f(A)
        Bin = f(B)

        if Ain and Bin:
            out.append(B)
        elif Ain and not Bin:
            out.append(intersect(A, B))
        elif (not Ain) and Bin:
            out.append(intersect(A, B))
            out.append(B)
        # else: оба снаружи -> ничего

    return np.array(out, dtype=float)


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def make_wedge_polygon_on_square(half_side, boresight_deg, half_bw_deg=45.0):
    """
    Строгий полигон основания: квадрат |x|<=half_side,|y|<=half_side
    пересечён с азимутальным сектором [b-45, b+45] (ширина 90°).
    """
    # стартовый квадрат (CCW)
    poly = np.array(
        [
            [-half_side, -half_side],
            [half_side, -half_side],
            [half_side, half_side],
            [-half_side, half_side],
        ],
        dtype=float,
    )

    a1 = np.deg2rad(boresight_deg - half_bw_deg)
    a2 = np.deg2rad(boresight_deg + half_bw_deg)
    d1 = np.array([np.cos(a1), np.sin(a1)])  # луч границы 1
    d2 = np.array([np.cos(a2), np.sin(a2)])  # луч границы 2

    # Внутри сектора (ширина <180°) можно задать двумя полуплоскостями:
    # 1) cross(d1, p) >= 0  (p слева от луча d1)
    # 2) cross(d2, p) <= 0  (p справа от луча d2)

    def f1(p):  # cross(d1,p) >= 0
        return cross2(d1, p) >= -1e-12

    def inter1(A, B):
        cA = cross2(d1, A)
        cB = cross2(d1, B)
        denom = cB - cA
        if abs(denom) < 1e-15:
            return B
        t = -cA / denom
        t = np.clip(t, 0.0, 1.0)
        return A + t * (B - A)

    poly = clip_polygon_halfplane(poly, f1, inter1)

    def f2(p):  # cross(d2,p) <= 0
        return cross2(d2, p) <= 1e-12

    def inter2(A, B):
        cA = cross2(d2, A)
        cB = cross2(d2, B)
        denom = cB - cA
        if abs(denom) < 1e-15:
            return B
        t = -cA / denom
        t = np.clip(t, 0.0, 1.0)
        return A + t * (B - A)

    poly = clip_polygon_halfplane(poly, f2, inter2)

    return poly


fig = go.Figure()

# =========================
# PLOTLY PRO 3D (RF-simulator look)
# Требует: X, Y, P_total_rel_dB, half, H (км), make_wedge_polygon_on_square(...)
# =========================

DECIM2 = 6
Xg = X[::DECIM2, ::DECIM2]
Yg = Y[::DECIM2, ::DECIM2]
Zg = np.zeros_like(Xg)

dB = P_total_rel_dB[::DECIM2, ::DECIM2]
dB_clip = np.clip(dB, -20, 0)

fig.add_trace(
    go.Surface(
        x=Xg,
        y=Yg,
        z=Zg,  # ← покрытие НЕ поднимается
        surfacecolor=dB_clip,  # ← цвет = уровень
        colorscale="Viridis",
        cmin=-20,
        cmax=0,
        opacity=0.98,
        colorbar=dict(title="dB (rel)"),
        name="Coverage",
        hoverinfo="skip",
    )
)

# Контуры -3/-6/-10 dB "на земле" (z=0)
# (в Plotly это делается встроенными контурами Surface)
fig.update_traces(
    contours_z=dict(
        show=True, usecolormap=False, highlightcolor="white", project_z=True
    ),
    selector=dict(name="Coverage"),
)

# Объект в воздухе
fig.add_trace(
    go.Scatter3d(
        x=[0],
        y=[0],
        z=[H],
        mode="markers+text",
        marker=dict(size=6),
        text=[f"Объект (H={H} км)"],
        textposition="top center",
        name="Object",
    )
)

# Квадрат 24×24 км (по месту) на земле
hs = float(half)
sqx = [-hs, hs, hs, -hs, -hs]
sqy = [-hs, -hs, hs, hs, -hs]
sqz = [0, 0, 0, 0, 0]
fig.add_trace(
    go.Scatter3d(
        x=sqx, y=sqy, z=sqz, mode="lines", line=dict(width=6), name="Footprint"
    )
)

Z_BASE = 0.001


# --- 4 пирамиды ДН (строго: сектор ∩ квадрат)
def add_pyramid(fig, base_xy, apex=(0.0, 0.0, H), name="beam", opacity=0.18):
    """
    base_xy: (M,2) полигон основания на земле (z=0), CCW/любой порядок ок
    apex: (x,y,z)
    """
    M = base_xy.shape[0]
    if M < 3:
        return

    # вершины: сначала основание (z=0), потом apex
    vx = list(base_xy[:, 0]) + [apex[0]]
    vy = list(base_xy[:, 1]) + [apex[1]]
    vz = [0.0] * M + [apex[2]]
    apex_idx = M

    # треугольники боковых граней: (apex, i, i+1)
    I, J, K = [], [], []
    for i in range(M):
        j = (i + 1) % M
        I.append(apex_idx)
        J.append(i)
        K.append(j)

    fig.add_trace(
        go.Mesh3d(
            x=vx,
            y=vy,
            z=vz,
            i=I,
            j=J,
            k=K,
            opacity=opacity,
            name=name,
            flatshading=True,
        )
    )

    # контур основания
    base_closed = np.vstack([base_xy, base_xy[0]])
    fig.add_trace(
        go.Scatter3d(
            x=base_closed[:, 0],
            y=base_closed[:, 1],
            z=np.zeros(base_closed.shape[0]),
            mode="lines",
            line=dict(width=4),
            name=f"{name} base",
        )
    )


bores = [0.0, 90.0, 180.0, 270.0]
for b in bores:
    base = make_wedge_polygon_on_square(half_side=hs, boresight_deg=b, half_bw_deg=45.0)
    add_pyramid(fig, base, name=f"Beam {int(b)}°", opacity=0.20)

# =============================
# 3D "объём" поля: 3 изосёрфейса (-3/-6/-10 dB)
# ВСТАВИТЬ: после пирамид, ДО fig.update_layout(...)
# =============================
NX, NY, NZ = 18, 18, 12  # плотность 3D-сетки (если тормозит: 25,25,16)
Z_MIN, Z_MAX = 0.2, H  # от земли до объекта

# 3D сетка (км)
xv = np.linspace(-half, half, NX)
yv = np.linspace(-half, half, NY)
zv = np.linspace(Z_MIN, Z_MAX, NZ)
XX, YY, ZZ = np.meshgrid(xv, yv, zv, indexing="xy")

# Относительно объекта (0,0,H)
dx = XX
dy = YY
dz = H - ZZ
R = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9

# Углы
theta_x3 = np.arctan2(dx, dz)
theta_y3 = np.arctan2(dy, dz)
phi3 = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

# ДН: cos^2 и маски
m_el = 2.0
m_az = 2.0
theta_el_lim = np.deg2rad(HPBW_el / 2)
theta_az_lim = np.deg2rad(HPBW_az / 2)
eps = 1e-12

G_el3 = (np.cos(theta_x3) ** m_el) * (np.cos(theta_y3) ** m_el)
G_el3 = np.where(
    (np.abs(theta_x3) <= np.pi / 2) & (np.abs(theta_y3) <= np.pi / 2), G_el3, 0.0
)
mask_el3 = (np.abs(theta_x3) <= theta_el_lim) & (np.abs(theta_y3) <= theta_el_lim)

FSPL3 = 92.45 + 20 * np.log10(f_GHz) + 20 * np.log10(R)

boresights = [0.0, 90.0, 180.0, 270.0]
P3 = np.full((len(boresights),) + R.shape, -1e9, dtype=float)

for i, b in enumerate(boresights):
    dphi_deg = wrap_deg(phi3 - b)  # wrap_deg у тебя уже есть
    dphi = np.deg2rad(dphi_deg)

    mask_az3 = np.abs(dphi) <= theta_az_lim

    G_az3 = np.cos(dphi) ** m_az
    G_az3 = np.where(np.abs(dphi) <= np.pi / 2, G_az3, 0.0)

    G3 = np.where(mask_el3 & mask_az3, G_el3 * G_az3, 0.0)
    G3_dB = 10 * np.log10(G3 + eps)

    P3[i] = G3_dB - FSPL3

P3_total = np.max(P3, axis=0)
P3_rel = P3_total - np.nanmax(P3_total)  # dB rel

# =============================
# Радиальные расстояния до уровней -3/-6/-10 dB (8 направлений)
# Требует: X, Y, P_total_rel_dB, half
# =============================
levels_r = [-3.0, -6.0, -10.0]
angles_deg = [0, 90, 180, 270]

# 1D оси сетки
x1d = X[0, :]
y1d = Y[:, 0]
ix0 = int(np.argmin(np.abs(x1d)))
iy0 = int(np.argmin(np.abs(y1d)))


def bilinear(Z, x, y):
    """Билинейная интерполяция Z на регулярной сетке X,Y (из meshgrid)."""
    # x в пределах x1d, y в пределах y1d
    ix = np.searchsorted(x1d, x) - 1
    iy = np.searchsorted(y1d, y) - 1
    ix = int(np.clip(ix, 0, len(x1d) - 2))
    iy = int(np.clip(iy, 0, len(y1d) - 2))

    x0, x1 = x1d[ix], x1d[ix + 1]
    y0, y1 = y1d[iy], y1d[iy + 1]

    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    z00 = Z[iy, ix]
    z10 = Z[iy, ix + 1]
    z01 = Z[iy + 1, ix]
    z11 = Z[iy + 1, ix + 1]

    return (
        (1 - tx) * (1 - ty) * z00
        + tx * (1 - ty) * z10
        + (1 - tx) * ty * z01
        + tx * ty * z11
    )


# =============================
# Размерная линия "24 км" по верхней стороне квадрата
# Требует: half (=12 км)
# =============================
hs = float(half)

# линия-стрелка (просто толстая линия) над верхней стороной
y_dim = hs + 0.9  # чуть выше квадрата
fig.add_trace(
    go.Scatter3d(
        x=[-hs, hs],
        y=[y_dim, y_dim],
        z=[0.06, 0.06],
        mode="lines+text",
        line=dict(width=3, color="rgba(255,255,255,0.75)"),
        text=["", "24 км"],
        textposition="middle right",
        textfont=dict(size=14, color="white"),
        showlegend=False,
    )
)

# выносные линии к концам стороны
for sx in (-hs, hs):
    fig.add_trace(
        go.Scatter3d(
            x=[sx, sx],
            y=[hs, y_dim],
            z=[0.06, 0.06],
            mode="lines",
            line=dict(width=3, color="rgba(255,255,255,0.5)"),
            showlegend=False,
        )
    )

# =============================
# ULTRA-LIGHT CINEMATIC: "дымка" вокруг объекта (очень лёгкая)
# =============================

# параметры дымки
R0 = 1.2  # радиус сферы, км
Nphi = 40
Ntheta = 24
fog_opacity = 0.18

phi = np.linspace(0, 2 * np.pi, Nphi)
theta = np.linspace(0, np.pi, Ntheta)
PHI, TH = np.meshgrid(phi, theta)

# сфера вокруг объекта (центр в (0,0,H))
Xf = R0 * np.cos(PHI) * np.sin(TH)
Yf = R0 * np.sin(PHI) * np.sin(TH)
Zf = H + R0 * np.cos(TH)

# градиент "яркости" (сильнее вниз/к центру)
brightness = 1 - (TH / np.pi)  # 1..0

fig.add_trace(
    go.Surface(
        x=Xf,
        y=Yf,
        z=Zf,
        surfacecolor=brightness,
        colorscale=[[0, "rgba(120,180,255,0.0)"], [1, "rgba(120,180,255,1.0)"]],
        showscale=False,
        opacity=fog_opacity,
        name="Emitter fog",
    )
)

# =============================
# FINAL TOUCH — RF emitter core
# =============================

# маленькая светящаяся сфера под объектом
R_core = 0.35  # радиус ядра, км
Nphi = 26
Ntheta = 18

phi = np.linspace(0, 2 * np.pi, Nphi)
theta = np.linspace(0, np.pi, Ntheta)
PHI, TH = np.meshgrid(phi, theta)

Xc = R_core * np.cos(PHI) * np.sin(TH)
Yc = R_core * np.sin(PHI) * np.sin(TH)
Zc = H + R_core * np.cos(TH)

# яркость центра
core_brightness = 1 - (TH / np.pi)

fig.add_trace(
    go.Surface(
        x=Xc,
        y=Yc,
        z=Zc,
        surfacecolor=core_brightness,
        colorscale=[[0, "rgba(255,255,200,0.0)"], [1, "rgba(255,255,120,1.0)"]],
        showscale=False,
        opacity=0.55,
        name="RF core",
    )
)

# =============================
# ULTRA-CLEAN GRID (лёгкая RF сетка земли)
# =============================

grid_step = 3.0  # шаг сетки (км) — можно 2.0 если хочешь плотнее
grid_color = "rgba(80,120,255,0.03)"

# линии X
for gx in np.arange(-L, L + grid_step, grid_step):
    fig.add_trace(
        go.Scatter3d(
            x=[gx, gx],
            y=[-L, L],
            z=[0, 0],
            mode="lines",
            name="Grid",
            line=dict(width=2, color=grid_color),
            showlegend=False,
        )
    )

# линии Y
for gy in np.arange(-L, L + grid_step, grid_step):
    fig.add_trace(
        go.Scatter3d(
            x=[-L, L],
            y=[gy, gy],
            z=[0, 0],
            mode="lines",
            name="Grid",
            line=dict(width=2, color=grid_color),
            showlegend=False,
        )
    )

# =========================
# Подписи контуров -3/-6/-10 dB (Plotly) через matplotlib paths
# =========================
levels_lbl = [-10.0, -6.0, -3.0]  # по возрастанию
labels_per_level = 4

fig_tmp, ax_tmp = plt.subplots(figsize=(4, 4))
cs = ax_tmp.contour(Xg, Yg, dB_clip, levels=levels_lbl)
plt.close(fig_tmp)

# =============================
# УДАЛЯЕМ старые подписи (если код перезапускается)
# =============================
fig.data = tuple(
    tr
    for tr in fig.data
    if not (
        hasattr(tr, "name") and isinstance(tr.name, str) and tr.name.startswith("LBL_")
    )
)

# =============================
# ПРОСТЫЕ ПОДПИСИ НА ОСНОВАНИИ ПИРАМИДЫ
# =============================

Z_BASE = 0.001

# сначала удалим старые подписи (чтобы не было каши)
fig.data = tuple(
    tr
    for tr in fig.data
    if not (
        hasattr(tr, "name") and isinstance(tr.name, str) and tr.name.startswith("LBL_")
    )
)

# цвета уровней
lvl_color = {
    -3.0: "rgba(255,220,120,1.0)",
    -6.0: "rgba(160,255,160,1.0)",
    -10.0: "rgba(160,200,255,1.0)",
}


# --- внешний вид DEMO STYLE (быстро и красиво)
fig.update_layout(
    template="plotly_dark",
    title="RF Coverage PRO 3D (interactive): pyramids + ground + level-as-height",
    paper_bgcolor="rgb(5,6,10)",
    plot_bgcolor="rgb(5,6,10)",
    scene=dict(
        bgcolor="rgb(5,6,10)",
        xaxis=dict(
            title="X, км",
            range=[-L, L],
            showbackground=True,
            backgroundcolor="rgb(10,12,18)",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="Y, км",
            range=[-L, L],
            showbackground=True,
            backgroundcolor="rgb(10,12,18)",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        zaxis=dict(
            title="Height, км",
            range=[0, H * 1.15],
            showbackground=True,
            backgroundcolor="rgb(10,12,18)",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.85),
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.1), up=dict(x=0, y=0, z=1)),
    ),
    margin=dict(l=0, r=0, t=50, b=0),
)

# =============================
# RF planner toggle: TOP (читабельный) / 3D
# вставить перед fig.write_html(...)
# =============================

cam_3d = dict(
    eye=dict(x=1.6, y=1.6, z=1.1),
    projection=dict(type="perspective"),
    up=dict(x=0, y=0, z=1),
)

cam_top = dict(
    eye=dict(x=0.0, y=0.0, z=2.6),
    projection=dict(type="orthographic"),
    up=dict(x=0, y=1, z=0),
)

# видимость трасс
vis_3d = []
vis_top = []

for tr in fig.data:
    nm = (getattr(tr, "name", "") or "").lower()

    is_cov = "coverage" in nm
    is_sq = ("footprint" in nm) or ("24×24" in nm) or ("24" in nm and "km" in nm)
    is_meas = "measure" in nm
    is_grid = "grid" in nm

    is_beam = "beam" in nm
    is_obj = "object" in nm

    # 3D: всё показываем
    vis_3d.append(True)

    # TOP: только “аналитика”
    vis_top.append(is_cov or is_sq or is_meas)


# =============================
# RF MODE: stable 3D + readable TOP (planner)
# вставить перед fig.write_html(...)
# Требует: ann_top (список annotations для TOP), fig уже собран
# =============================

# --- камеры (стабильные)
cam_3d = dict(
    eye=dict(x=1.55, y=1.55, z=1.05),
    up=dict(x=0, y=0, z=1),
    projection=dict(type="perspective"),
    center=dict(x=0, y=0, z=0),
)

cam_top = dict(
    # "чуть выше карты", чтобы занимала почти весь экран
    eye=dict(x=0.0, y=0.0, z=0.85),
    up=dict(x=0, y=1, z=0),
    projection=dict(type="orthographic"),
    center=dict(x=0, y=0, z=0),
)

# --- видимость трасс по именам
vis_3d = []
vis_top = []

for tr in fig.data:
    nm = (getattr(tr, "name", "") or "").lower()

    is_cov = "coverage" in nm
    is_fp = ("footprint" in nm) or ("24×24" in nm)
    is_meas = "measure" in nm  # если ты оставила measure traces; если нет — ок
    is_beam = "beam" in nm
    is_obj = "object" in nm
    is_grid = "grid" in nm

    # 3D: всё, кроме лишней "карточной" сетки по желанию
    vis_3d.append(True)

    # TOP: только карта + квадрат (+measure если они трассами)
    # пирамиды/объект/сетка скрываем — чтобы сверху было читабельно
    vis_top.append(is_cov or is_fp or is_meas)


# =============================
# REAL ISO-LINES (-3/-6/-10 dB) from field, as Plotly lines
# Требует: X, Y, P_total_rel_dB, fig
# Нужны импорты: import matplotlib.pyplot as plt
# =============================

iso_levels = [-10.0, -6.0, -3.0]  # обязательно по возрастанию
iso_style = {
    -3.0: dict(color="rgba(255,220,120,0.95)", width=4),
    -6.0: dict(color="rgba(140,255,160,0.90)", width=4),
    -10.0: dict(color="rgba(120,180,255,0.90)", width=4),
}

# Чтобы не грузить: берём прореженную сетку для изолиний (качество почти не падает)
STEP_ISO = 3
Xs = X[::STEP_ISO, ::STEP_ISO]
Ys = Y[::STEP_ISO, ::STEP_ISO]
Zs = P_total_rel_dB[::STEP_ISO, ::STEP_ISO]

# Строим изолинии matplotlib (только чтобы получить пути)
fig_tmp, ax_tmp = plt.subplots(figsize=(4, 4))
cs = ax_tmp.contour(Xs, Ys, Zs, levels=iso_levels)
plt.close(fig_tmp)


# cs.allsegs: список по уровням; каждый элемент — список сегментов (Nx2)
for lev, segs in zip(cs.levels, cs.allsegs):
    lev = float(lev)
    st = iso_style.get(lev, dict(color="rgba(255,255,255,0.8)", width=5))

    if not segs:
        continue

    # каждый сегмент рисуем отдельной линией (Plotly так проще)
    for seg in segs:
        if seg is None or len(seg) < 6:
            continue

        # seg: array(K,2) где [:,0]=x, [:,1]=y
        xx = seg[:, 0]
        yy = seg[:, 1]
        zz = np.full_like(xx, 0.02)  # чуть над землёй, чтобы не мерцало

        fig.add_trace(
            go.Scatter3d(
                x=xx,
                y=yy,
                z=zz,
                mode="lines",
                line=dict(width=st["width"], color=st["color"]),
                showlegend=False,
                name=f"Iso {int(lev)} dB",
            )
        )


# =============================
# Labels near isolines WITHOUT find_r_for_level
# Требует: cs (matplotlib.contour), fig
# =============================
Z_BASE = 0.02  # чтобы не z=0 в ноль с поверхностью

lvl_style = {
    -3.0: dict(color="rgba(255,220,120,1.0)"),
    -6.0: dict(color="rgba(140,255,160,1.0)"),
    -10.0: dict(color="rgba(120,180,255,1.0)"),
}


def pick_label_point(seg: np.ndarray) -> tuple[float, float]:
    k = int(np.argmax(seg[:, 1]))  # верхняя точка изолинии
    return float(seg[k, 0]), float(seg[k, 1])


for lev, segs in zip(cs.levels, cs.allsegs):
    lev = float(lev)
    if not segs:
        continue

    # берём самый длинный сегмент уровня
    best = None
    best_len = 0.0
    for seg in segs:
        if seg is None or len(seg) < 20:
            continue
        Lseg = float(np.sum(np.sqrt(np.sum(np.diff(seg, axis=0) ** 2, axis=1))))
        if Lseg > best_len:
            best_len = Lseg
            best = seg
    if best is None:
        continue

    x0, y0 = pick_label_point(best)
    r0 = (x0 * x0 + y0 * y0) ** 0.5  # расстояние от центра

    col = lvl_style.get(lev, dict(color="rgba(255,255,255,1.0)"))["color"]

    # небольшой сдвиг наружу от центра, чтобы текст стоял рядом с линией
    scale = 1.06
    x1, y1 = x0 * scale, y0 * scale

    # 1) halo (кольцо) — ВНИМАНИЕ: тут x0/y0, а не px/py
    fig.add_trace(
        go.Scatter3d(
            x=[x0],
            y=[y0],
            z=[Z_BASE],
            mode="markers",
            marker=dict(
                size=14,
                symbol="circle-open",
                color="rgba(255,255,255,0.85)",
                line=dict(width=2, color="rgba(255,255,255,0.85)"),
            ),
            showlegend=False,
            name=f"ISO_LABEL_{int(lev)}",
            hoverinfo="skip",
        )
    )

    # 2) core (точка)
    fig.add_trace(
        go.Scatter3d(
            x=[x0],
            y=[y0],
            z=[Z_BASE],
            mode="markers",
            marker=dict(size=6, symbol="circle", color=col),
            showlegend=False,
            name=f"ISO_LABEL_{int(lev)}",
            hovertemplate=f"{int(lev)} dB · {r0:.1f} км<extra></extra>",
        )
    )

    # 3) “ножка” к подписи (в плоскости основания)
    fig.add_trace(
        go.Scatter3d(
            x=[x0, x1],
            y=[y0, y1],
            z=[Z_BASE, Z_BASE],
            mode="lines",
            line=dict(width=4, color=col),
            showlegend=False,
            name=f"ISO_LABEL_{int(lev)}",
            hoverinfo="skip",
        )
    )

    # 4) подпись (тоже на основании)
    fig.add_trace(
        go.Scatter3d(
            x=[x1],
            y=[y1],
            z=[Z_BASE],
            mode="text",
            text=[f"{int(lev)} dB · {r0:.1f} км"],
            textfont=dict(size=18, color="white", family="Arial Black"),
            showlegend=False,
            name=f"ISO_LABEL_{int(lev)}",
            hoverinfo="skip",
        )
    )

# --- нормальный RF 3D старт (без TOP)
fig.update_layout(
    scene=dict(
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.9),  # z↑ = выше “высота” (попробуй 0.9)
    )
)

print("[INFO] RF MODE: normal 3D view")

# --- сохранить HTML рядом со скриптом
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, "coverage_PRO_3D.html")
fig.write_html(out_path)
print("PRO HTML сохранён сюда:", out_path)
