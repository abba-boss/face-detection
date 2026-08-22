import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { ErrorBox } from '../components/ErrorBox'
import { PageHeader } from '../components/PageHeader'

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString('en-GB', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  } catch {
    return iso
  }
}

export function Users() {
  const { data, loading, error, refetch } = useApi(() => api.users())

  return (
    <div style={{ padding: '28px 32px', maxWidth: 900 }}>
      <PageHeader
        title="Enrolled Users"
        subtitle="Biometric face enrollments stored in FaceAuth"
        action={
          <button onClick={refetch} style={{
            background: 'none', border: '1px solid var(--color-border)',
            color: 'var(--color-muted)', borderRadius: 6, padding: '6px 14px',
            cursor: 'pointer', fontSize: 12, fontFamily: 'inherit',
          }}>↻ Refresh</button>
        }
      />

      {loading && <Spinner />}
      {error   && <ErrorBox message={error} onRetry={refetch} />}

      {data && (
        <Card title={`${data.count} user${data.count !== 1 ? 's' : ''} enrolled`}>
          {data.count === 0 ? (
            <p style={{ color: 'var(--color-muted)', fontSize: 13, margin: 0 }}>
              No users enrolled yet. Run: <code style={{ color: 'var(--color-blue)' }}>python main.py enroll --user &lt;name&gt;</code>
            </p>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                  {['Username', 'Enrolled At', 'Status'].map(h => (
                    <th key={h} style={{
                      textAlign: 'left', padding: '6px 10px', fontSize: 11,
                      fontWeight: 600, letterSpacing: '0.06em', color: 'var(--color-muted)',
                      textTransform: 'uppercase',
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.users.map((u, i) => (
                  <tr key={u.username} style={{
                    borderBottom: i < data.users.length - 1 ? '1px solid var(--color-border)' : 'none',
                  }}>
                    <td style={{ padding: '10px 10px', color: 'var(--color-text)', fontWeight: 600 }}>
                      <span style={{
                        display: 'inline-flex', alignItems: 'center', gap: 8,
                      }}>
                        <span style={{
                          width: 28, height: 28, borderRadius: '50%',
                          background: 'rgba(88,166,255,0.15)', border: '1px solid rgba(88,166,255,0.25)',
                          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                          fontSize: 12, color: 'var(--color-blue)',
                        }}>
                          {u.username[0].toUpperCase()}
                        </span>
                        {u.username}
                      </span>
                    </td>
                    <td style={{ padding: '10px 10px', color: 'var(--color-muted)' }}>
                      {formatDate(u.enrolled_at)}
                    </td>
                    <td style={{ padding: '10px 10px' }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                        background: 'rgba(63,185,80,0.12)', color: 'var(--color-green)',
                        border: '1px solid rgba(63,185,80,0.25)',
                      }}>Active</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      )}
    </div>
  )
}
