import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { ErrorBox } from '../components/ErrorBox'
import { PageHeader } from '../components/PageHeader'
import { StatusBadge } from '../components/StatusBadge'
import type { DoctorCheck } from '../api/client'

function CheckRow({ check }: { check: DoctorCheck }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: 12,
      padding: '9px 0', borderBottom: '1px solid var(--color-border)',
    }}>
      <span style={{
        fontSize: 14, marginTop: 1,
        color: check.ok ? 'var(--color-green)' : 'var(--color-red)',
        flexShrink: 0,
      }}>
        {check.ok ? '✓' : '✗'}
      </span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 500, fontSize: 13, color: 'var(--color-text)' }}>
          {check.name}
        </div>
        {check.note && (
          <div style={{
            fontSize: 11, color: 'var(--color-muted)', marginTop: 2,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            maxWidth: 560,
          }} title={check.note}>
            {check.note}
          </div>
        )}
      </div>
      <StatusBadge ok={check.ok} labelOk="PASS" labelFail="FAIL" />
    </div>
  )
}

export function Doctor() {
  const { data, loading, error, refetch } = useApi(() => api.doctor())

  return (
    <div style={{ padding: '28px 32px', maxWidth: 900 }}>
      <PageHeader
        title="System Doctor"
        subtitle="Automated health checks for all FaceAuth components"
        action={
          <button onClick={refetch} style={{
            background: 'none', border: '1px solid var(--color-border)',
            color: 'var(--color-muted)', borderRadius: 6, padding: '6px 14px',
            cursor: 'pointer', fontSize: 12, fontFamily: 'inherit',
          }}>↻ Run checks</button>
        }
      />

      {loading && <Spinner />}
      {error   && <ErrorBox message={error} onRetry={refetch} />}

      {data && (
        <>
          {/* Summary banner */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 14, marginBottom: 20,
            padding: '12px 18px', borderRadius: 8,
            background: data.ok ? 'rgba(63,185,80,0.08)' : 'rgba(248,81,73,0.08)',
            border: `1px solid ${data.ok ? 'rgba(63,185,80,0.25)' : 'rgba(248,81,73,0.25)'}`,
          }}>
            <span style={{ fontSize: 20 }}>{data.ok ? '✓' : '✗'}</span>
            <div>
              <div style={{
                fontWeight: 600, fontSize: 14,
                color: data.ok ? 'var(--color-green)' : 'var(--color-red)',
              }}>
                {data.ok ? 'All checks passed' : `${data.failures} check${data.failures !== 1 ? 's' : ''} failed`}
              </div>
              <div style={{ fontSize: 12, color: 'var(--color-muted)', marginTop: 2 }}>
                {data.checks.length} checks total · {data.checks.filter(c => c.ok).length} passed · {data.failures} failed
              </div>
            </div>
          </div>

          {/* Checks */}
          <Card title="Checks">
            <div>
              {data.checks.map((c, i) => (
                <CheckRow key={i} check={c} />
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
