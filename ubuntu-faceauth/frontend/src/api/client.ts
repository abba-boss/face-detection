/**
 * FaceAuth API client.
 * All requests go through the Vite dev-proxy to http://127.0.0.1:8765.
 * No auth, no duplicated business logic — purely a thin fetch wrapper.
 */

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} → ${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${path} → ${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string
  version: string
}

export interface StatusResponse {
  version: string
  model: string
  camera: string
  threshold: number
  liveness_timeout: number
  storage: string
  enrolled: number
  users: string[]
}

export interface UserInfo {
  username: string
  enrolled_at: string
}

export interface UsersResponse {
  count: number
  users: UserInfo[]
}

export interface LogsResponse {
  limit: number
  count: number
  events: string[]
}

export interface DoctorCheck {
  name: string
  ok: boolean
  note: string
}

export interface DoctorResponse {
  ok: boolean
  failures: number
  checks: DoctorCheck[]
}

export interface AuthRequest {
  user: string
}

export interface AuthResponse {
  success: boolean
  user: string
  similarity: number
  message: string
  outcome: string
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const api = {
  health:       ()                       => get<HealthResponse>('/health'),
  status:       ()                       => get<StatusResponse>('/status'),
  users:        ()                       => get<UsersResponse>('/users'),
  logs:         (limit = 20)             => get<LogsResponse>(`/logs?limit=${limit}`),
  doctor:       ()                       => get<DoctorResponse>('/doctor'),
  authenticate: (body: AuthRequest)      => post<AuthResponse>('/authenticate', body),
}
