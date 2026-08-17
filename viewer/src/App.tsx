// stickdance viewer — white room with every ARDY clip at once.
// Left: stage. Right: group filter, clip list (click to focus), transport.
import { useEffect, useMemo, useRef, useState } from 'react'
import Stage, { type Placed, type CamState } from './Stage'
import { loadIndex, loadClip, type Index, type Clip, GROUP_COLORS } from './data'

export default function App() {
  const [index, setIndex] = useState<Index | null>(null)
  const [clips, setClips] = useState<Record<string, Clip>>({})
  const [groups, setGroups] = useState<Set<string>>(new Set())
  const [seed, setSeed] = useState<number | 'all'>(0)
  const [maxN, setMaxN] = useState(64)
  const [t, setT] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [focus, setFocus] = useState<string | null>(null)
  const [trails, setTrails] = useState(true)
  const [cam, setCam] = useState<CamState>({ yaw: 0.6, pitch: 0.4, ppm: 55 })
  const [q, setQ] = useState('')
  const raf = useRef(0)

  useEffect(() => { loadIndex().then((ix) => { setIndex(ix); setGroups(new Set(ix.clips.map((c) => c.group))) }) }, [])

  const visibleMeta = useMemo(() => {
    if (!index) return []
    return index.clips.filter((c) => groups.has(c.group) && (seed === 'all' || c.seed === seed) &&
      (!q || c.text.toLowerCase().includes(q.toLowerCase()))).slice(0, maxN)
  }, [index, groups, seed, q, maxN])

  // load what's visible
  useEffect(() => {
    let alive = true
    ;(async () => {
      for (const m of visibleMeta) {
        if (clips[m.id]) continue
        const c = await loadClip(m)
        if (!alive) return
        setClips((prev) => (prev[m.id] ? prev : { ...prev, [m.id]: c }))
      }
    })()
    return () => { alive = false }
  }, [visibleMeta])

  // place on a grid, deterministic yaw per clip
  const placed: Placed[] = useMemo(() => {
    const ready = visibleMeta.map((m) => clips[m.id]).filter(Boolean) as Clip[]
    const cols = Math.ceil(Math.sqrt(ready.length))
    return ready.map((clip, i) => {
      let h = 0; for (const ch of clip.id) h = (h * 31 + ch.charCodeAt(0)) | 0
      return { clip, x: (i % cols - (cols - 1) / 2) * 2.4, z: (Math.floor(i / cols) - (cols - 1) / 2) * 2.4, yaw: ((h % 360) * Math.PI) / 180 }
    })
  }, [visibleMeta, clips])

  const T = useMemo(() => Math.max(1, ...placed.map((p) => p.clip.frames)), [placed])
  useEffect(() => {
    if (!playing) return
    let last = performance.now(), acc = 0
    const step = (now: number) => {
      acc += ((now - last) / 1000) * 20 * speed; last = now
      if (acc >= 1) { const n = Math.floor(acc); acc -= n; setT((v) => (v + n) % T) }
      raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [playing, speed, T])

  if (!index) return <div style={{ padding: 24 }}>loading…</div>
  const allGroups = Array.from(new Set(index.clips.map((c) => c.group)))
  const focusMeta = focus ? index.clips.find((c) => c.id === focus) : null

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', height: '100%' }}>
      <div style={{ position: 'relative' }}>
        <Stage index={index} placed={placed} t={t} cam={cam} onCam={setCam} focus={focus} onPick={setFocus} showTrails={trails} />
        <div style={{ position: 'absolute', left: 12, top: 10, fontSize: 12, color: '#666' }}>
          stickdance · white room — {placed.length} figures · frame {t}/{T} · drag orbit · wheel zoom · click figure
        </div>
        {focusMeta && (
          <div style={{ position: 'absolute', left: 12, bottom: 12, background: '#fff', border: '1px solid #ddd', borderRadius: 10, padding: '8px 12px', fontSize: 13, maxWidth: 480 }}>
            <span style={{ color: GROUP_COLORS[focusMeta.group], fontWeight: 600 }}>{focusMeta.group}</span> · seed {focusMeta.seed}
            <div style={{ marginTop: 2 }}>{focusMeta.text}</div>
            <div style={{ color: '#999', fontSize: 11, marginTop: 2 }}>{focusMeta.id}</div>
          </div>
        )}
      </div>
      <div style={{ borderLeft: '1px solid #eee', padding: 12, overflow: 'auto', fontSize: 13, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button onClick={() => setPlaying((p) => !p)} style={btn}>{playing ? '⏸' : '▶'}</button>
          <input type="range" min={0} max={T - 1} value={t} onChange={(e) => { setPlaying(false); setT(+e.target.value) }} style={{ flex: 1 }} />
          <select value={speed} onChange={(e) => setSpeed(+e.target.value)}>{[0.25, 0.5, 1, 2].map((s) => <option key={s} value={s}>{s}×</option>)}</select>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {allGroups.map((g) => (
            <button key={g} onClick={() => setGroups((s) => { const n = new Set(s); n.has(g) ? n.delete(g) : n.add(g); return n })}
              style={{ ...btn, background: groups.has(g) ? GROUP_COLORS[g] ?? '#333' : '#f2f2f2', color: groups.has(g) ? '#fff' : '#666' }}>{g}</button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          seed <select value={seed} onChange={(e) => setSeed(e.target.value === 'all' ? 'all' : +e.target.value)}>
            <option value="all">all</option><option value={0}>0</option><option value={1}>1</option><option value={2}>2</option></select>
          max <select value={maxN} onChange={(e) => setMaxN(+e.target.value)}>{[16, 36, 64, 100, 200, 500].map((n) => <option key={n} value={n}>{n}</option>)}</select>
          <label><input type="checkbox" checked={trails} onChange={(e) => setTrails(e.target.checked)} /> trails</label>
        </div>
        <input placeholder="filter prompt…" value={q} onChange={(e) => setQ(e.target.value)} style={{ padding: 6, border: '1px solid #ddd', borderRadius: 6 }} />
        <div style={{ color: '#888', fontSize: 11 }}>{visibleMeta.length} clips shown / {index.clips.length} total</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {visibleMeta.map((m) => (
            <div key={m.id} onClick={() => setFocus(focus === m.id ? null : m.id)}
              style={{ padding: '4px 6px', borderRadius: 6, cursor: 'pointer', background: focus === m.id ? '#ffecec' : 'transparent', borderLeft: `3px solid ${GROUP_COLORS[m.group] ?? '#999'}` }}>
              <span style={{ color: '#999', fontSize: 11 }}>s{m.seed} </span>{m.text}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
const btn: React.CSSProperties = { border: 0, borderRadius: 6, padding: '4px 8px', cursor: 'pointer', background: '#f2f2f2', fontSize: 12 }
