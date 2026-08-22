import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { ErrorBox } from '../components/ErrorBox'
import { PageHeader } from '../components/PageHeader'

interface SettingRowProps { label: string; value: string | number; description?: string }

function SettingRow({ label, value, description }: SettingRowProps) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
      padding: '11px 0', borderBottom: '1px solid var(--color-border)',
    }}>
      <div>
        <div style={{ fontSize: 13, color: 'var(--color-text)', fontWeight: 500 }}>{label}</div>
        {description && (
          <div style={{ fontSize: 11, color: 'var(--color-muted)', marginTop: 2 }}>{description}</div>
        )}
      </div>
      <code style={{
        background: 'rgba(88,166,255,0.08)', border: '1px solid rgba(88,166,255,0.2)',
        color: 'var(--color-blue)', borderRadius: 4, padding: '2px 10px',
        fontSize: 12, fontFamily: 'inherit', flexShrink: 0, marginLeft: 16,
      }}>
        {String(value)}
      </code>
    </div>
  )
}

export function Settings() {
  const { data, loading, error, refetch } = useApi(() => api.status())

  return (
    <div style={{ padding: '28px 32px', maxWidth: 800 }}>
      <PageHeader
        title="Settings"
        subtitle="Current FaceAuth configuration — read only"
        action={
          <div style={{
            padding: '4px 12px', borderRadius: 4, fontSize: 11, fontWeight: 600,
            background: 'rgba(210,153,34,0.12)', border: '1px solid rgba(210,153,34,0.3)',
            color: 'var(--color-yellow)',
          }}>
            READ ONLY
          </div>
        }
      />

      <p style={{ color: 'var(--color-muted)', fontSize: 13, marginBottom: 24, marginTop: 0 }}>
        To change settings, edit <code style={{ color: 'var(--color-blue)' }}>app/config/settings.py</code> and restart the API server.
      </p>

      {loading && <Spinner />}
      {error   && <ErrorBox message={error} onRetry={refetch} />}

      {data && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <Card title="FaceAuth">
            <SettingRow label="Version"  value={data.version} />
          </Card>

          <Card title="Recognition">
            <SettingRow
              label="InsightFace model"
              value={data.model}
              description="buffalo_sc = fast CPU · buffalo_l = higher accuracy"
            />
            <SettingRow
              label="Recognition threshold"
              value={data.threshold}
              description="Cosine similarity above this → AUTHORIZED"
            />
          </Card>

          <Card title="Liveness">
            <SettingRow
              label="Liveness timeout"
              value={`${data.liveness_timeout}s`}
              description="Seconds to complete the head-turn challenge"
            />
          </Card>

          <Card title="Hardware">
            <SettingRow
              label="Camera device"
              value={data.camera}
              description="V4L2 camera device used for capture"
            />
          </Card>

          <Card title="Storage">
            <SettingRow
              label="Backend"
              value={data.storage}
              description="sqlite = default · npz = legacy"
            />
            <SettingRow
              label="Enrolled users"
              value={data.enrolled}
            />
          </Card>
        </div>
      )}
    </div>
  )
}
