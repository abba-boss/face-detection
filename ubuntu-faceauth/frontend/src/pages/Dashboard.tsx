import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { ErrorBox } from '../components/ErrorBox'
import { PageHeader } from '../components/PageHeader'
import { StatusBadge } from '../components/StatusBadge'
import type { ReactNode } from 'react'

function StatRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '8px 0', borderBottom: '1px solid var(--color-border)',
    }}>
      <span style={{ color: 'var(--color-muted)', fontSize: 13 }}>{label}</span>
      <span style={{ color: 'var(--color-text)', fontSize: 13, fontWeight: 500 }}>{value}</span>
    </div>
  )
}

export function Dashboard() {
  const status = useApi(() => api.status())
  const health = useApi(() => api.health())
  const logs   = useApi(() => api.logs(5))

  const isOnline = health.data?.status === 'ok'

  return (
    <div style={{ padding: '28px 32px', maxWidth: 1100 }}>
      <PageHeader
        title="Dashboard"
        subtitle="Ubuntu FaceAuth system overview"
        action={
          <StatusBadge
            ok={!health.loading && isOnline}
            labelOk="API ONLINE"
            labelFail={health.loading ? 'CONNECTING…' : 'API OFFLINE'}
          />
        }
      />

      {/* Top metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 24 }}>
        {[
          { label: 'VERSION',    value: status.data?.version        ?? '—' },
          { label: 'MODEL',      value: status.data?.model          ?? '—' },
          { label: 'ENROLLED',   value: status.data?.enrolled ?? '—' },
          { label: 'THRESHOLD',  value: status.data?.threshold      ?? '—' },
          { label: 'LIVENESS',   value: status.data ? `${status.data.liveness_timeout}s` : '—' },
          { label: 'STORAGE',    value: status.data?.storage        ?? '—' },
        ].map(m => (
          <div key={m.label} style={{
            background: 'var(--color-surface)', border: '1px solid var(--color-border)',
            borderRadius: 8, padding: '16px 18px',
          }}>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', color: 'var(--color-muted)', marginBottom: 6 }}>
              {m.label}
            </div>
            <div style={{ fontSize: 18, fontWeight: 600, color: 'var(--color-text)', fontFamily: 'inherit' }}>
              {status.loading ? <Spinner /> : String(m.value)}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* System status */}
        <Card title="System Status">
          {status.loading && <Spinner />}
          {status.error   && <ErrorBox message={status.error} onRetry={status.refetch} />}
          {status.data    && (
            <>
              <StatRow label="Camera device"       value={status.data.camera} />
              <StatRow label="Recognition threshold" value={status.data.threshold} />
              <StatRow label="Liveness timeout"    value={`${status.data.liveness_timeout}s`} />
              <StatRow label="Storage backend"     value={status.data.storage} />
              <StatRow label="InsightFace model"   value={status.data.model} />
              <div style={{ paddingTop: 8, borderBottom: 'none' }}>
                <StatRow label="Enrolled users" value={
                  <span style={{ color: 'var(--color-blue)', fontWeight: 600 }}>
                    {status.data.enrolled}
                  </span>
                } />
              </div>
              {status.data.users.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  {status.data.users.map(u => (
                    <span key={u} style={{
                      display: 'inline-block', marginRight: 6, marginBottom: 4,
                      padding: '2px 8px', background: 'rgba(88,166,255,0.1)',
                      border: '1px solid rgba(88,166,255,0.25)', borderRadius: 4,
                      fontSize: 12, color: 'var(--color-blue)',
                    }}>{u}</span>
                  ))}
                </div>
              )}
            </>
          )}
        </Card>

        {/* Recent events */}
        <Card title="Recent Authentication Events">
          {logs.loading && <Spinner />}
          {logs.error   && <ErrorBox message={logs.error} onRetry={logs.refetch} />}
          {logs.data?.events.length === 0 && (
            <p style={{ color: 'var(--color-muted)', fontSize: 13, margin: 0 }}>No events yet.</p>
          )}
          {logs.data?.events.map((e, i) => (
            <div key={i} style={{
              fontSize: 11, padding: '5px 0',
              borderBottom: i < (logs.data!.events.length - 1) ? '1px solid var(--color-border)' : 'none',
              color: e.includes('SUCCESS') ? 'var(--color-green)'
                   : e.includes('FAIL') || e.includes('DENY') ? 'var(--color-red)'
                   : 'var(--color-muted)',
              fontFamily: 'inherit',
              wordBreak: 'break-all',
            }}>{e}</div>
          ))}
        </Card>
      </div>
    </div>
  )
}
