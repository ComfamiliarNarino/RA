"""Construye experiencia.glb (web y Android) y experiencia.usdz (iPhone) a partir de config.json y la carpeta imagenes/.
Secuencia: la foto del edificio se despliega (3 s) -> se ve completa (3 s) -> rayo y destello -> aparece la valla ->
las imágenes rotan en ciclo con la duración definida en el panel. Lo ejecuta GitHub Actions cada vez que se publica."""
import json, os, sys, time, shutil, tempfile
import numpy as np, trimesh, pygltflib
from PIL import Image
from shapely.geometry import box
from trimesh.visual.material import PBRMaterial
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdUtils, Vt

RAIZ = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
cfg = json.load(open(os.path.join(RAIZ, "config.json"), encoding="utf-8"))
MAX_IMGS = 10                                            # máximo de imágenes por valla (peso y fluidez en celulares)
_validas = [i for i in cfg["imagenes"] if os.path.exists(os.path.join(RAIZ, i["archivo"]))][:MAX_IMGS]
IMGS = [(i["archivo"], max(1.0, float(i["segundos"]))) for i in _validas]
TIPOS = [i.get("transicion", "giro") if i.get("transicion", "giro") in ("giro", "persiana", "zoom", "corte") else "giro" for i in _validas]
if not IMGS: sys.exit("config.json no tiene imágenes válidas")

# ---------------- parámetros ----------------
V_ANCHO = float(cfg.get("valla_ancho", 1.70))          # ancho de la valla (m)
BASE = float(cfg.get("altura_base", 1.50))             # toda la animación surge a esta altura del piso (m)
FOTO_W = float(cfg.get("edificio_ancho", 1.50))        # ancho de la foto del edificio (m)
k = V_ANCHO / 1.30
VIS_W = 1.16 * k; VIS_H = VIS_W * 1.5                   # área visible 2:3 (imágenes 1080 x 1620 o 1024 x 1536)
V_ALTO = 2.00 * k                                       # misma proporción de la valla anterior (1,30 x 2,00)
V_Z = -0.10
T_DESPLIEGUE, T_PAUSA, T_TRANSICION = 3.0, 3.0, 1.5
T_INTRO = T_DESPLIEGUE + T_PAUSA + T_TRANSICION
CICLO = sum(d for _, d in IMGS)
N_CICLOS = max(3, int(np.ceil(180 / CICLO)))            # versión lineal (iPhone) de unos 3 minutos
T_FIN = T_INTRO + N_CICLOS * CICLO
FPS, EPS, N_FRANJAS, T_FRANJA = 24, 1e-4, 24, 0.55
TMP = tempfile.mkdtemp()

def srgb2lin(h):
    c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return [float(v) for v in np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)]
def rrect(w, h, r): return box(-w / 2, -h / 2, w / 2, h / 2).buffer(-r).buffer(r, resolution=20)
def extruir(poly, z0, z1):
    m = trimesh.creation.extrude_polygon(poly, height=z1 - z0); m.apply_translation([0, 0, z0]); return m
suave = lambda u: (lambda v: v * v * (3 - 2 * v))(np.clip(u, 0, 1))
def tramo(x, pts): xs, ys = zip(*pts); return float(np.interp(x, xs, ys))

# ---------------- imágenes: ajustadas a 2:3 (768 x 1152) ----------------
texturas = []
for n, (ruta, _) in enumerate(IMGS):
    im = Image.open(os.path.join(RAIZ, ruta)).convert("RGB")
    lienzo = Image.new("RGB", (768, 1152), "white")
    esc = min(768 / im.width, 1152 / im.height); im = im.resize((round(im.width * esc), round(im.height * esc)), Image.LANCZOS)
    lienzo.paste(im, ((768 - im.width) // 2, (1152 - im.height) // 2))       # si no es 2:3, se centra con fondo blanco
    destino = os.path.join(TMP, f"img{n:02d}.jpg"); lienzo.save(destino, quality=85); texturas.append(destino)

# ---------------- foto del edificio en franjas ----------------
foto = Image.open(os.path.join(RAIZ, "recursos", "edificio.png")).convert("RGBA")
foto = foto.crop(foto.getchannel("A").point(lambda a: 255 if a > 20 else 0).getbbox())
foto = foto.resize((1024, round(1024 * foto.size[1] / foto.size[0])), Image.LANCZOS)
ruta_foto = os.path.join(TMP, "edificio_foto.png"); foto.save(ruta_foto)
FOTO_H = FOTO_W * foto.size[1] / foto.size[0]; h_fr = FOTO_H / N_FRANJAS
def franja(kk):
    v = np.array([[-FOTO_W / 2, 0, 0], [FOTO_W / 2, 0, 0], [FOTO_W / 2, h_fr, 0], [-FOTO_W / 2, h_fr, 0]])
    uv = np.array([[0, kk / N_FRANJAS], [1, kk / N_FRANJAS], [1, (kk + 1) / N_FRANJAS], [0, (kk + 1) / N_FRANJAS]], float)
    return v, np.array([[0, 1, 2], [0, 2, 3]]), uv
def inicio_franja(kk): return kk * (T_DESPLIEGUE - T_FRANJA) / (N_FRANJAS - 1)

# ---------------- valla ----------------
FONDO, REL, R = 0.075, 0.022, 0.06 * k
ext = rrect(V_ANCHO, V_ALTO, R); abert = rrect(VIS_W + 0.024, VIS_H + 0.024, 0.03); labio = rrect(VIS_W, VIS_H, 0.02)
PANEL_Z = 0.055
valla = {
    "v_caja":  (trimesh.util.concatenate([extruir(ext.difference(abert), 0, FONDO), extruir(abert, 0, 0.004)]), ("#474F57", 0.35, 0.50)),
    "v_marco": (trimesh.util.concatenate([extruir(ext.buffer(-e, join_style=1).difference(abert.buffer(e, join_style=1)), FONDO + REL * a, FONDO + REL * b)
                                          for a, b, e in [(0, .7, 0), (.7, .88, .004), (.88, 1, .009)]]), ("#4E565F", 0.40, 0.42)),
    "v_labio": (extruir(abert.difference(labio), PANEL_Z, FONDO + 0.006), ("#A9AFB5", 0.80, 0.30)),
    "v_panel": (extruir(labio.buffer(0.002), PANEL_Z - 0.003, PANEL_Z), ("#E3E5E6", 0.00, 0.85)),
}
for m, _ in valla.values(): m.apply_translation([0, 0, -FONDO / 2])
Z_VIS, Z_OCU = PANEL_Z + 0.0015 - FONDO / 2, 0.02 - FONDO / 2
PW, PH = VIS_W + 0.006, VIS_H + 0.006
P_V = np.array([[-PW / 2, -PH / 2, 0], [PW / 2, -PH / 2, 0], [PW / 2, PH / 2, 0], [-PW / 2, PH / 2, 0]])
P_F = np.array([[0, 1, 2], [0, 2, 3]]); P_UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
V_CENTRO = (0.0, BASE + V_ALTO / 2, V_Z)

# ---------------- transiciones: cada imagen se arma con 8 franjas verticales ----------------
N_TIRAS, H_TR, ESC_T = 8, 0.4, 0.002        # franjas, medio tiempo de transición (s), salto instantáneo (s)
W_T = PW / N_TIRAS
def tira(kk):   # franja kk con pivote en su centro y su porción de la imagen
    v = np.array([[-W_T / 2, -PH / 2, 0], [W_T / 2, -PH / 2, 0], [W_T / 2, PH / 2, 0], [-W_T / 2, PH / 2, 0]])
    uv = np.array([[kk / N_TIRAS, 0], [(kk + 1) / N_TIRAS, 0], [(kk + 1) / N_TIRAS, 1], [kk / N_TIRAS, 1]], float)
    return v, np.array([[0, 1, 2], [0, 2, 3]]), uv, -PW / 2 + (kk + 0.5) * W_T
INICIOS = np.cumsum([0] + [d for _, d in IMGS[:-1]])
def estado_ciclo(i, tc):
    """(z, escala_imagen, [(escala_x, angulo_y) por franja]) de la imagen i en el instante tc del ciclo."""
    s_i = INICIOS[i]; e_i = s_i + IMGS[i][1]
    t_in, t_out = TIPOS[i], TIPOS[(i + 1) % len(IMGS)]
    oculto = (Z_OCU, 1.0, [(1.0, 0.0)] * N_TIRAS)
    if not (s_i - 1e-9 <= tc < e_i - 1e-9): return oculto
    if tc < s_i + H_TR and t_in != "corte": tipo, p, signo = t_in, (tc - s_i) / H_TR, -1       # entrando
    elif tc >= e_i - H_TR and t_out != "corte": tipo, p, signo = t_out, (e_i - tc) / H_TR, 1   # saliendo
    else: return (Z_VIS, 1.0, [(1.0, 0.0)] * N_TIRAS)
    p = float(np.clip(p, 0, 1))
    if tipo == "zoom": return (Z_VIS, max(EPS, suave(p)), [(1.0, 0.0)] * N_TIRAS)
    tiras = []
    for kk in range(N_TIRAS):
        pk = suave(np.clip((p - 0.3 * kk / (N_TIRAS - 1)) / 0.7, 0, 1))
        tiras.append((max(EPS, pk), 0.0) if tipo == "persiana" else (1.0, signo * 90.0 * (1 - pk)))
    return (Z_VIS, 1.0, tiras)
def estado(i, t, base):
    if t < base - 1e-9: return estado_ciclo(i, 0.0)               # durante la intro: como al inicio del ciclo (imagen 1 aún sin entrar)
    tc = (t - base) % CICLO
    if abs(tc - CICLO) < 1e-6: tc = 0.0
    return estado_ciclo(i, round(tc, 6))
def tiempos_img(i, base, n_ciclos, con_intro):
    ts = {0.0} if con_intro else set()
    ts |= {base, base + n_ciclos * CICLO}
    paso = 1 / FPS
    for c in range(n_ciclos + 1):
        s = base + c * CICLO + INICIOS[i]; e = s + IMGS[i][1]
        for b, anim_tr in ((s, TIPOS[i] != "corte"), (e, TIPOS[(i + 1) % len(IMGS)] != "corte")):
            ts |= {b - ESC_T, b}
            if anim_tr:
                ventana = np.arange(b, b + H_TR + 1e-9, paso) if b == s else np.arange(b - H_TR, b + 1e-9, paso)
                ts |= set(np.round(ventana, 5)) | {round(b + H_TR if b == s else b - H_TR, 5)}
    fin = base + n_ciclos * CICLO
    return np.array(sorted(t for t in ts if (0.0 if con_intro else base) - 1e-9 <= t <= fin + 1e-9))
USA_TIRAS = [TIPOS[i] in ("giro", "persiana") or TIPOS[(i + 1) % len(IMGS)] in ("giro", "persiana") for i in range(len(IMGS))]
def quat_y(grados): a = np.radians(grados) / 2; return [0.0, float(np.sin(a)), 0.0, float(np.cos(a))]

# ---------------- luz ----------------
H_RAYO = V_ALTO + 0.3
rayo = trimesh.creation.cylinder(radius=0.03, segment=[[0, 0, 0], [0, H_RAYO, 0]], sections=16)
halo = trimesh.creation.cylinder(radius=0.09, segment=[[0, 0, 0], [0, H_RAYO, 0]], sections=16)
DW, DH = V_ANCHO + 0.25, V_ALTO + 0.25
destello = trimesh.Trimesh([[-DW / 2, -DH / 2, 0], [DW / 2, -DH / 2, 0], [DW / 2, DH / 2, 0], [-DW / 2, DH / 2, 0]], [[0, 1, 2], [0, 2, 3]], process=False)
ancla = trimesh.Trimesh([[-0.01, 0, 0.02], [0.01, 0, 0.02], [0.01, 0, 0.04], [-0.01, 0, 0.04]], [[0, 2, 1], [0, 3, 2]], process=False)

# ---------------- curvas de la intro ----------------
T0 = T_DESPLIEGUE + T_PAUSA
ti = np.round(np.linspace(0, T_INTRO, int(round(T_INTRO * FPS)) + 1), 5)
edif_s = []
for kk in range(N_FRANJAS):
    t_off = T0 + 0.1 + (N_FRANJAS - 1 - kk) * 0.5 / (N_FRANJAS - 1)
    edif_s.append(np.array([[1, max(EPS, suave((x - inicio_franja(kk)) / T_FRANJA) * (1 - suave((x - t_off) / 0.2))), 1] for x in ti]))
ky = [max(EPS, tramo(x, [(0, 0), (T0, 0), (T0 + 0.3, 1), (T0 + 0.8, 1), (T0 + 1.0, 0), (T_INTRO, 0)])) for x in ti]
rayo_s = np.array([[1 if v > 0.01 else EPS, v, 1 if v > 0.01 else EPS] for v in ky])
dk = [max(EPS, tramo(x, [(0, 0), (T0 + 0.2, 0), (T0 + 0.6, 1), (T0 + 1.0, 1), (T0 + 1.4, 0), (T_INTRO, 0)])) for x in ti]
dest_s = np.array([[v, v, v] for v in dk])
vk = [max(EPS, suave((x - (T0 + 0.6)) / 0.5)) for x in ti]
valla_s = np.array([[v, v, v] for v in vk])
def visible_en(tc):
    acc = 0
    for i, (_, d) in enumerate(IMGS):
        if tc < acc + d: return i
        acc += d
    return len(IMGS) - 1

# ---------------- GLB ----------------
esc = trimesh.Scene(); base_fr = esc.graph.base_frame
T = np.eye(4); T[1, 3] = BASE; esc.graph.update(frame_to="Edificio", frame_from=base_fr, matrix=T)
MAT_FOTO = PBRMaterial(name="edificio_foto", baseColorTexture=Image.open(ruta_foto), metallicFactor=0.0, roughnessFactor=0.8, alphaMode="MASK", alphaCutoff=0.5)
for kk in range(N_FRANJAS):
    v, f, uv = franja(kk); fr = trimesh.Trimesh(v, f, vertex_normals=np.tile([0, 0, 1.0], (4, 1)), process=False)
    fr.visual = trimesh.visual.TextureVisuals(uv=uv, material=MAT_FOTO)
    Tk = np.eye(4); Tk[1, 3] = kk * h_fr; esc.add_geometry(fr, node_name=f"Franja_{kk:02d}", geom_name=f"Franja_{kk:02d}", parent_node_name="Edificio", transform=Tk)
T = np.eye(4); T[:3, 3] = V_CENTRO; esc.graph.update(frame_to="Valla", frame_from=base_fr, matrix=T)
for n, (m, (col, met, rug)) in valla.items():
    m = m.copy(); m.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(name=n, baseColorFactor=srgb2lin(col) + [1.0], metallicFactor=met, roughnessFactor=rug))
    esc.add_geometry(m, node_name=n, geom_name=n, parent_node_name="Valla")
for i, tex in enumerate(texturas):
    Ti = np.eye(4); Ti[2, 3] = Z_VIS if i == 0 else Z_OCU
    esc.graph.update(frame_to=f"Img_{i:02d}", frame_from="Valla", matrix=Ti)
    mat_i = PBRMaterial(name=f"img{i:02d}", baseColorTexture=Image.open(tex), metallicFactor=0.0, roughnessFactor=0.6)
    for kk in range(N_TIRAS):
        v, f, uv, cx = tira(kk); tm = trimesh.Trimesh(v, f, vertex_normals=np.tile([0, 0, 1.0], (4, 1)), process=False)
        tm.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat_i)
        Tk = np.eye(4); Tk[0, 3] = cx
        esc.add_geometry(tm, node_name=f"Tira_{i:02d}_{kk}", geom_name=f"Tira_{i:02d}_{kk}", parent_node_name=f"Img_{i:02d}", transform=Tk)
LUZ = PBRMaterial(name="luz", baseColorFactor=[1, 1, 1, 1], emissiveFactor=[1, 1, 1], metallicFactor=0, roughnessFactor=1)
LUZ_AZ = PBRMaterial(name="luz_azul", baseColorFactor=srgb2lin("#9FD8FF") + [1], emissiveFactor=srgb2lin("#9FD8FF"), metallicFactor=0, roughnessFactor=1)
T = np.eye(4); T[:3, 3] = [0, BASE, V_Z + 0.05]; esc.graph.update(frame_to="Rayo", frame_from=base_fr, matrix=T)
r1 = rayo.copy(); r1.visual = trimesh.visual.TextureVisuals(material=LUZ); esc.add_geometry(r1, node_name="rayo", geom_name="rayo", parent_node_name="Rayo")
h1 = halo.copy(); h1.visual = trimesh.visual.TextureVisuals(material=LUZ_AZ); esc.add_geometry(h1, node_name="halo", geom_name="halo", parent_node_name="Rayo")
T = np.eye(4); T[:3, 3] = [0, V_CENTRO[1], V_Z + 0.09]; esc.graph.update(frame_to="Destello", frame_from=base_fr, matrix=T)
d1 = destello.copy(); d1.visual = trimesh.visual.TextureVisuals(material=LUZ); esc.add_geometry(d1, node_name="destello", geom_name="destello", parent_node_name="Destello")
an = ancla.copy(); an.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(name="ancla", baseColorFactor=[1, 1, 1, 0.0], alphaMode="BLEND"))
esc.add_geometry(an, node_name="ancla_piso", geom_name="ancla_piso")
glb_base = os.path.join(TMP, "base.glb"); esc.export(glb_base)

gl = pygltflib.GLTF2().load(glb_base); blob = gl.binary_blob(); idx = {x.name: i for i, x in enumerate(gl.nodes)}
def agregar(d, tipo, minmax=False):
    global blob
    while len(blob) % 4: blob += b"\x00"
    off = len(blob); raw = np.asarray(d, np.float32).tobytes(); blob += raw
    gl.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=off, byteLength=len(raw)))
    a = pygltflib.Accessor(bufferView=len(gl.bufferViews) - 1, componentType=pygltflib.FLOAT, count=len(d), type=tipo)
    if minmax: a.min, a.max = [float(np.min(d))], [float(np.max(d))]
    gl.accessors.append(a); return len(gl.accessors) - 1
def fijar(nombre, t, s):
    x = gl.nodes[idx[nombre]]; x.matrix = None; x.translation = [float(v) for v in t]; x.scale = [float(v) for v in s]; x.rotation = [0, 0, 0, 1]
fijar("Edificio", [0, BASE, 0], [1, 1, 1]); fijar("Valla", V_CENTRO, [1, 1, 1])
for kk in range(N_FRANJAS): fijar(f"Franja_{kk:02d}", [0, kk * h_fr, 0], [1, EPS, 1])       # al final de la intro el edificio queda oculto
fijar("Rayo", [0, BASE, V_Z + 0.05], [EPS] * 3); fijar("Destello", [0, V_CENTRO[1], V_Z + 0.09], [EPS] * 3)
for i in range(len(texturas)):
    fijar(f"Img_{i:02d}", [0, 0, Z_VIS if i == 0 else Z_OCU], [1, 1, 1])
    for kk in range(N_TIRAS): fijar(f"Tira_{i:02d}_{kk}", [tira(kk)[3], 0, 0], [1, 1, 1])
def anim(nombre, pistas):
    s_, c_ = [], []
    for nodo, path, t, v, ip in pistas:
        tipo_v = pygltflib.VEC4 if path == "rotation" else pygltflib.VEC3
        s_.append(pygltflib.AnimationSampler(input=agregar(t, pygltflib.SCALAR, True), output=agregar(v, tipo_v), interpolation=ip))
        c_.append(pygltflib.AnimationChannel(sampler=len(s_) - 1, target=pygltflib.AnimationChannelTarget(node=idx[nodo], path=path)))
    return pygltflib.Animation(name=nombre, samplers=s_, channels=c_)
def pistas_imgs(base, n_ciclos, con_intro, t_ini=None, t_fin=None):
    out = []
    for i in range(len(IMGS)):
        ts = tiempos_img(i, base, n_ciclos, con_intro) if t_ini is None else np.array([t_ini, t_fin])
        est = [estado(i, t, base) for t in ts]
        out.append((f"Img_{i:02d}", "translation", ts, np.array([[0, 0, e[0]] for e in est]), "LINEAR"))
        out.append((f"Img_{i:02d}", "scale", ts, np.array([[e[1]] * 3 for e in est]), "LINEAR"))
        if USA_TIRAS[i]:
            for kk in range(N_TIRAS):
                out.append((f"Tira_{i:02d}_{kk}", "scale", ts, np.array([[e[2][kk][0], 1, 1] for e in est]), "LINEAR"))
                out.append((f"Tira_{i:02d}_{kk}", "rotation", ts, np.array([quat_y(e[2][kk][1]) for e in est]), "LINEAR"))
    return out
intro = [(f"Franja_{kk:02d}", "scale", ti, edif_s[kk], "LINEAR") for kk in range(N_FRANJAS)]
intro += [("Rayo", "scale", ti, rayo_s, "LINEAR"), ("Destello", "scale", ti, dest_s, "LINEAR"), ("Valla", "scale", ti, valla_s, "LINEAR")]
N_ESC = len(intro)
intro += pistas_imgs(T_INTRO, 1, True, 0.0, T_INTRO)                    # imágenes quietas (y la primera aún sin entrar) durante la intro
completa = [(n, p, np.append(t, T_FIN), np.vstack([v, v[-1:]]), ip) for n, p, t, v, ip in intro[:N_ESC]] + pistas_imgs(T_INTRO, N_CICLOS, True)
gl.animations = [anim("Completa", completa), anim("Intro", intro), anim("Ciclo", pistas_imgs(0.0, 1, False))]
gl.buffers[0].byteLength = len(blob); gl.set_binary_blob(blob)
gl.save(os.path.join(RAIZ, "experiencia.glb"))

# ---------------- USDZ ----------------
usdc = os.path.join(TMP, "experiencia.usdc")
st = Usd.Stage.CreateNew(usdc)
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(st, 1.0)
st.SetTimeCodesPerSecond(FPS); st.SetStartTimeCode(0); st.SetEndTimeCode(T_FIN * FPS)
root = UsdGeom.Xform.Define(st, "/Experiencia"); st.SetDefaultPrim(root.GetPrim()); UsdGeom.Scope.Define(st, "/Experiencia/Materiales")
def mat_usd(n, col=None, met=0.0, rug=0.5, tex=None, emis=None, opac=None, alfa=False):
    mt = UsdShade.Material.Define(st, f"/Experiencia/Materiales/{n}")
    sh = UsdShade.Shader.Define(st, f"/Experiencia/Materiales/{n}/Superficie"); sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(met)); sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rug))
    if emis: sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emis))
    if opac is not None: sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opac))
    if tex:
        lec = UsdShade.Shader.Define(st, f"/Experiencia/Materiales/{n}/LectorST"); lec.CreateIdAttr("UsdPrimvarReader_float2")
        lec.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        tx = UsdShade.Shader.Define(st, f"/Experiencia/Materiales/{n}/Textura"); tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex)
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(lec.ConnectableAPI(), "result")
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB"); tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
        if alfa:
            tx.CreateOutput("a", Sdf.ValueTypeNames.Float); sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(tx.ConnectableAPI(), "a")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col))
    mt.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface"); return mt
def malla(ruta, v, f, n, mt, uv=None):
    me = UsdGeom.Mesh.Define(st, ruta)
    me.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(v, np.float32)))
    me.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    me.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(f, np.int32).flatten()))
    me.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(n, np.float32)))
    me.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying if len(n) == 3 * len(f) else UsdGeom.Tokens.vertex)
    me.CreateSubdivisionSchemeAttr("none"); me.CreateExtentAttr([Gf.Vec3f(*np.min(v, 0).astype(float)), Gf.Vec3f(*np.max(v, 0).astype(float))])
    if uv is not None:
        UsdGeom.PrimvarsAPI(me).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
    UsdShade.MaterialBindingAPI.Apply(me.GetPrim()).Bind(mt)
def xform_anim(ruta, trasl, t_claves, escalas):
    x = UsdGeom.Xform.Define(st, ruta); ot = x.AddTranslateOp(); os_ = x.AddScaleOp(); ot.Set(Gf.Vec3d(*trasl))
    for t, s in zip(t_claves, escalas): os_.Set(Gf.Vec3f(*map(float, s)), float(t * FPS))
tk = np.append(ti, T_FIN)
UsdGeom.Xform.Define(st, "/Experiencia/Edificio").AddTranslateOp().Set(Gf.Vec3d(0, BASE, 0))
m_foto = mat_usd("edificio_foto", met=0, rug=0.8, tex="edificio_foto.png", alfa=True)
for kk in range(N_FRANJAS):
    xform_anim(f"/Experiencia/Edificio/Franja_{kk:02d}", [0, kk * h_fr, 0], tk, np.vstack([edif_s[kk], edif_s[kk][-1:]]))
    v, f, uv = franja(kk); malla(f"/Experiencia/Edificio/Franja_{kk:02d}/Imagen", v, f, np.tile([0, 0, 1.0], (4, 1)), m_foto, uv)
xform_anim("/Experiencia/Valla", V_CENTRO, tk, np.vstack([valla_s, valla_s[-1:]]))
for n, (m, (col, met, rug)) in valla.items(): malla(f"/Experiencia/Valla/{n}", m.vertices, m.faces, np.repeat(m.face_normals, 3, 0), mat_usd(n, srgb2lin(col), met, rug))
for i, tex in enumerate(texturas):
    x = UsdGeom.Xform.Define(st, f"/Experiencia/Valla/Img_{i:02d}"); ot = x.AddTranslateOp(); oe = x.AddScaleOp()
    ts = tiempos_img(i, T_INTRO, N_CICLOS, True); est = [estado(i, t, T_INTRO) for t in ts]
    for t, e in zip(ts, est): ot.Set(Gf.Vec3d(0, 0, e[0]), float(t * FPS)); oe.Set(Gf.Vec3f(e[1], e[1], e[1]), float(t * FPS))
    mt = mat_usd(f"img{i:02d}", met=0, rug=0.6, tex=os.path.basename(tex))
    for kk in range(N_TIRAS):
        v, f, uv, cx = tira(kk)
        xt = UsdGeom.Xform.Define(st, f"/Experiencia/Valla/Img_{i:02d}/Tira_{kk}")
        xt.AddTranslateOp().Set(Gf.Vec3d(cx, 0, 0)); orot = xt.AddRotateYOp(); osc = xt.AddScaleOp()
        if USA_TIRAS[i]:
            for t, e in zip(ts, est): orot.Set(float(e[2][kk][1]), float(t * FPS)); osc.Set(Gf.Vec3f(e[2][kk][0], 1, 1), float(t * FPS))
        else: orot.Set(0.0); osc.Set(Gf.Vec3f(1, 1, 1))
        malla(f"/Experiencia/Valla/Img_{i:02d}/Tira_{kk}/Imagen", v, f, np.tile([0, 0, 1.0], (4, 1)), mt, uv)
xform_anim("/Experiencia/Rayo", [0, BASE, V_Z + 0.05], tk, np.vstack([rayo_s, rayo_s[-1:]]))
malla("/Experiencia/Rayo/rayo", rayo.vertices, rayo.faces, np.repeat(rayo.face_normals, 3, 0), mat_usd("luz", [1, 1, 1], 0, 1, emis=[1, 1, 1]))
malla("/Experiencia/Rayo/halo", halo.vertices, halo.faces, np.repeat(halo.face_normals, 3, 0), mat_usd("luz_azul", srgb2lin("#9FD8FF"), 0, 1, emis=srgb2lin("#9FD8FF")))
xform_anim("/Experiencia/Destello", [0, V_CENTRO[1], V_Z + 0.09], tk, np.vstack([dest_s, dest_s[-1:]]))
malla("/Experiencia/Destello/destello", destello.vertices, destello.faces, np.tile([0, 0, 1.0], (4, 1)), mat_usd("luz_destello", [1, 1, 1], 0, 1, emis=[1, 1, 1]))
malla("/Experiencia/ancla_piso", ancla.vertices, ancla.faces, np.tile([0, 1.0, 0], (4, 1)), mat_usd("ancla", [1, 1, 1], 0, 1, opac=0.0))
st.GetRootLayer().Save()
UsdUtils.CreateNewUsdzPackage(usdc, os.path.join(RAIZ, "experiencia.usdz"))

version = time.strftime("%Y%m%d-%H%M%S")
json.dump({"version": version, "imagenes": len(IMGS), "ciclo_segundos": CICLO}, open(os.path.join(RAIZ, "version.json"), "w"), ensure_ascii=False)
shutil.rmtree(TMP, ignore_errors=True)
print(f"OK · {len(IMGS)} imágenes (transiciones: {', '.join(TIPOS)}) · ciclo {CICLO:g} s · valla {V_ANCHO:.2f} x {V_ALTO:.2f} m desde {BASE:.2f} m · versión {version}")
