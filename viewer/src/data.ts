// Clip loading: index.json + float32 [T,27,3] .bin per clip (x left, y up, z fwd; metres).
export type ClipMeta = { id: string; group: string; text: string; seed: number; frames: number; fps: number }
export type Index = { joints: string[]; parents: (string | null)[]; clips: ClipMeta[] }
export type Clip = ClipMeta & { data: Float32Array } // data[(t*27+j)*3 + k]

export const J = 27

export async function loadIndex(): Promise<Index> {
  const r = await fetch('/clips/index.json')
  return r.json()
}

export async function loadClip(m: ClipMeta): Promise<Clip> {
  const r = await fetch(`/clips/${m.id}.bin`)
  const buf = await r.arrayBuffer()
  return { ...m, data: new Float32Array(buf) }
}

// bone colours by child joint (same palette as generator/render.py)
export const INK = '#282828'
export const PALETTE: Record<string, string> = {
  LeftForeArm: '#e84030', LeftHand: '#ff9628', LeftHandEnd: '#ff9628',
  RightForeArm: '#286ee6', RightHand: '#50c8f0', RightHandEnd: '#50c8f0',
  LeftLeg: '#c832a0', LeftFoot: '#ff78c8', LeftToeBase: '#ff78c8',
  RightLeg: '#1e965a', RightFoot: '#78dc5a', RightToeBase: '#78dc5a',
}
export const GROUP_COLORS: Record<string, string> = {
  idle: '#999', locomotion: '#286ee6', gesture: '#ff9628', dance: '#e84030', sport: '#1e965a', transitions: '#c832a0',
  acrobatic: '#7a4bd8', moonwalk: '#111',
}
