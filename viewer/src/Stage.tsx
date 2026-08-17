// White-room stage: many clips on one grid floor, orthographic 3/4 camera, Canvas2D.
// Drag = orbit, wheel = zoom, click a figure = focus (App shows its prompt).
import { useEffect, useRef } from 'react'
import { type Clip, type Index, J, PALETTE, INK } from './data'

export type Placed = { clip: Clip; x: number; z: number; yaw: number }
export type CamState = { yaw: number; pitch: number; ppm: number }

type Props = {
  index: Index; placed: Placed[]; t: number; cam: CamState; onCam: (c: CamState) => void
  focus: string | null; onPick: (id: string | null) => void; showTrails: boolean
}

export default function Stage({ index, placed, t, cam, onCam, focus, onPick, showTrails }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null)
  const parents = index.parents.map((p) => (p ? index.joints.indexOf(p) : -1))
  const names = index.joints
  const HEAD = names.indexOf('Head'), HIPS = names.indexOf('Hips')
  const skip = new Set([names.indexOf('LeftHandThumb1'), names.indexOf('RightHandThumb1')])

  useEffect(() => {
    const cv = ref.current!
    const dpr = window.devicePixelRatio || 1
    const W = cv.clientWidth, H = cv.clientHeight
    cv.width = W * dpr; cv.height = H * dpr
    const g = cv.getContext('2d')!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.fillStyle = '#fff'; g.fillRect(0, 0, W, H)
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch)
    const cx0 = W / 2, cy0 = H * 0.6, k = cam.ppm
    const proj = (x: number, y: number, z: number) => {
      const sx = z * sy + x * cy
      const d = z * cy - x * sy
      const y2 = y * cp + d * sp
      return [cx0 + sx * k, cy0 - y2 * k, d * cp - y * sp] as const
    }
    // floor grid
    const ext = 12
    g.strokeStyle = '#e4e4e4'; g.lineWidth = 1
    for (let i = -ext; i <= ext; i++) {
      let a = proj(i, 0, -ext), b = proj(i, 0, ext)
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke()
      a = proj(-ext, 0, i); b = proj(ext, 0, i)
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke()
    }
    // figures, far first
    const items = placed.map((p) => {
      const T = p.clip.frames, f = Math.min(T - 1, t)
      const c = Math.cos(p.yaw), s = Math.sin(p.yaw)
      const pts: [number, number, number][] = []
      for (let j = 0; j < J; j++) {
        const o = (f * J + j) * 3
        const x = p.clip.data[o], y = p.clip.data[o + 1], z = p.clip.data[o + 2]
        pts.push([c * x + s * z + p.x, y, -s * x + c * z + p.z])
      }
      const hips = proj(...pts[HIPS])
      return { p, pts, depth: hips[2] }
    })
    items.sort((a, b) => a.depth - b.depth)
    for (const it of items) {
      const dim = focus && it.p.clip.id !== focus
      g.globalAlpha = dim ? 0.18 : 1
      // trail (root path so far)
      if (showTrails && !dim) {
        const T = it.p.clip.frames, c = Math.cos(it.p.yaw), s = Math.sin(it.p.yaw)
        g.strokeStyle = '#bbb'; g.lineWidth = 1.5; g.beginPath()
        for (let f = 0; f <= Math.min(T - 1, t); f += 2) {
          const o = (f * J + HIPS) * 3
          const x = it.p.clip.data[o], z = it.p.clip.data[o + 2]
          const q = proj(c * x + s * z + it.p.x, 0, -s * x + c * z + it.p.z)
          f === 0 ? g.moveTo(q[0], q[1]) : g.lineTo(q[0], q[1])
        }
        g.stroke()
      }
      // shadow
      const hp = it.pts[HIPS]
      const sh = proj(hp[0], 0, hp[2])
      g.fillStyle = '#d6d6d6'; g.beginPath(); g.ellipse(sh[0], sh[1], 0.28 * k, 0.28 * k * sp, 0, 0, Math.PI * 2); g.fill()
      // bones, sorted by depth within figure
      const bones: { d: number; a: readonly [number, number, number]; b: readonly [number, number, number]; col: string }[] = []
      for (let j = 0; j < J; j++) {
        const pj = parents[j]
        if (pj < 0 || skip.has(j)) continue
        const a = proj(...it.pts[pj]), b = proj(...it.pts[j])
        bones.push({ d: (a[2] + b[2]) / 2, a, b, col: PALETTE[names[j]] ?? INK })
      }
      bones.sort((x, y) => x.d - y.d)
      g.lineCap = 'round'; g.lineWidth = Math.max(2, 0.055 * k)
      for (const bn of bones) {
        g.strokeStyle = bn.col; g.beginPath(); g.moveTo(bn.a[0], bn.a[1]); g.lineTo(bn.b[0], bn.b[1]); g.stroke()
      }
      const hd = proj(...it.pts[HEAD])
      g.fillStyle = INK; g.beginPath(); g.arc(hd[0], hd[1], 0.125 * k, 0, Math.PI * 2); g.fill()
      if (it.p.clip.id === focus) {
        g.strokeStyle = '#ff3b30'; g.lineWidth = 2; g.beginPath(); g.ellipse(sh[0], sh[1], 0.5 * k, 0.5 * k * sp, 0, 0, Math.PI * 2); g.stroke()
      }
      // remember screen pos for picking
      ;(it.p as any)._screen = [proj(...hp)[0], proj(...hp)[1]]
    }
    g.globalAlpha = 1
  })

  const onDown = (e: React.MouseEvent) => { drag.current = { x: e.clientX, y: e.clientY, moved: false } }
  const onMove = (e: React.MouseEvent) => {
    if (!drag.current) return
    const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true
    drag.current.x = e.clientX; drag.current.y = e.clientY
    onCam({ ...cam, yaw: cam.yaw + dx * 0.006, pitch: Math.max(0.02, Math.min(1.3, cam.pitch + dy * 0.004)) })
  }
  const onUp = (e: React.MouseEvent) => {
    if (drag.current && !drag.current.moved) {
      // pick nearest figure by screen distance
      const r = ref.current!.getBoundingClientRect()
      const mx = e.clientX - r.left, my = e.clientY - r.top
      let best: string | null = null, bd = 40
      for (const p of placed) {
        const s = (p as any)._screen as [number, number] | undefined
        if (!s) continue
        const d = Math.hypot(s[0] - mx, s[1] - my)
        if (d < bd) { bd = d; best = p.clip.id }
      }
      onPick(best)
    }
    drag.current = null
  }
  const onWheel = (e: React.WheelEvent) => onCam({ ...cam, ppm: Math.max(15, Math.min(200, cam.ppm * Math.exp(-e.deltaY * 0.001))) })

  return <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block', cursor: 'grab' }}
    onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={() => (drag.current = null)} onWheel={onWheel} />
}
