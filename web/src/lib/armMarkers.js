export const ARM_MARKER_COLORS = {
  p_coarse: 0xfa8c16,
  p_safe: 0x52c41a,
  c_star: 0x13c2c2,
  p_pregrasp: 0x722ed1,
  s0_pregrasp: 0xeb2f96,
  hole_grasp: 0xfa541c,
  hole_center: 0xf5222d,
  hole_normal: 0xfadb14,
  viewpoint: 0x2f54eb,
  safe_point: 0x1677ff,
  tcp: 0xf0b429,
  default: 0x3dd6c6,
};

/** Ghost-arm target from Grasp / other tabs → ArmPanel Three.js preview. */
export const armGhostTarget = {
  current: null,
  listeners: new Set(),
  _emit() {
    this.listeners.forEach((fn) => {
      try { fn(this.current); } catch (_) {}
    });
  },
  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  },
  set(payload) {
    if (!payload || !Array.isArray(payload.joints_rad) || payload.joints_rad.length < 6) {
      return false;
    }
    this.current = {
      joints_rad: payload.joints_rad.slice(0, 6).map(Number),
      xyzrpy: Array.isArray(payload.xyzrpy) ? payload.xyzrpy.slice(0, 6).map(Number) : null,
      label: payload.label || 'ghost',
      source: payload.source || '',
      at: Date.now(),
    };
    this._emit();
    return true;
  },
  clear() {
    if (!this.current) return;
    this.current = null;
    this._emit();
  },
};

export const armSceneMarkers = {
  items: [],
  version: 0,
  listeners: new Set(),
  _emit() {
    this.version += 1;
    this.listeners.forEach((fn) => {
      try { fn(this.items.slice()); } catch (_) {}
    });
  },
  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  },
  upsert(point) {
    const id = String(point.id || point.label || "pt");
    const xyz = point.xyz;
    if (!xyz || xyz.length < 3 || ![0, 1, 2].every((i) => Number.isFinite(Number(xyz[i])))) {
      return false;
    }
    const next = this.items.filter((p) => p.id !== id);
    const entry = {
      id,
      label: point.label || id,
      xyz: [Number(xyz[0]), Number(xyz[1]), Number(xyz[2])],
      color: point.color != null ? point.color : ARM_MARKER_COLORS.default,
    };
    // Optional unit direction in robot/base frame → Three.js arrow.
    const dir = point.dir || point.normal;
    if (Array.isArray(dir) && dir.length >= 3 && [0, 1, 2].every((i) => Number.isFinite(Number(dir[i])))) {
      entry.dir = [Number(dir[0]), Number(dir[1]), Number(dir[2])];
      const len = Number(point.dir_length ?? point.normal_length_m ?? 0.1);
      entry.dir_length = Number.isFinite(len) && len > 0 ? len : 0.1;
    }
    next.push(entry);
    this.items = next;
    this._emit();
    return true;
  },
  upsertMany(points) {
    let ok = 0;
    (points || []).forEach((p) => { if (this.upsert(p)) ok += 1; });
    return ok;
  },
  remove(id) {
    const before = this.items.length;
    this.items = this.items.filter((p) => p.id !== id);
    if (this.items.length !== before) this._emit();
  },
  clear() {
    if (!this.items.length) return;
    this.items = [];
    this._emit();
  },
};

