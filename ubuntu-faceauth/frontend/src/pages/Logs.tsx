import { useState } from 'react'
import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { ErrorBox } from '../components/ErrorBox'
import { PageHeader } from '../components/PageHeader'

const LIMITS = [10, 20, 50, 100]

function eventColor(line: string): string {
  if (line.includes('SUCCESS')) return 'var(--color-green)'
  if (line.includes('FAIL') || line.includes('DENIED') || line.includes('ERROR')) return 'var(--color-red)'
  if (line.includes('TIMEOUT') || line.includes('WARNING')) return 'var(--color-yellow)'
  if (line.includes('START') || line.includes('command=')) return 'var(--color-blue)'
  return 'var(--color-muted)'
}

export function Logs() {
  const [limit, setLimit] = useState(20)
  const { data, loading, error, refetch } = useApi(() => api.logs(limit))

  // re-fetch when limit changes
  const changeLimit = (n: number) => { setLimit(n); setTimeout(refetch, 0) }

  return (
    <div style={{ padding: '28px 32px', maxWidth: 1100 }}>
      <PageHeader
        title="Authentication Logs"
        subtitle="Recent FaceAuth events — per-frame noise filtered"
        action={
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {LIMITS.map(l => (
              <button key={l} onClick={() => changeLimit(l)} style={{
                background: l === limit ? 'rgba(88,166,255,0.15)' : 'none',
                border: `1px solid ${l === limit ? 'rgba(88,166,255,0.4)' : 'var(--color-border)'}`,
                color: l === limit ? 'var(--color-blue)' : 'var(--color-muted)',
                borderRadius: 5, padding: '4px 12px', cursor: 'pointer',
                fontSize: 12, fontFamily: 'inherit',
              }}>{l}</button>
            ))}
            <button onClick={refetch} style={{
              background: 'none', border: '1px solid var(--color-border)',
              color: 'var(--color-muted)', borderRadius: 5, padding: '4px 12px',
              cursor: 'pointer', fontSize: 12, fontFamily: 'inherit',
            }}>↻</button>
          </div>
        }
      />

      {loading && <Spinner />}
      {error   && <ErrorBox message={error} onRetry={refetch} />}

      {data && (
        <Card title={`${data.count} event${data.count !== 1 ? 's' : ''} (limit ${data.limit})`}>
          {data.count === 0 ? (
            <p style={{ color: 'var(--color-muted)', fontSize: 13, margin: 0 }}>
              No authentication events found in the log yet.
            </p>
          ) : (
            <div style={{
              fontFamily: 'inherit', fontSize: 12, lineHeight: 1.7,
              maxHeight: 600, overflowY: 'auto',
            }}>
              {data.events.map((e, i) => (
                <div key={i} style={{
                  padding: '4px 0',
                  borderBottom: i < data.events.length - 1 ? '1px solid rgba(48,54,61,0.5)' : 'none',
                  color: eventColor(e),
                  wordBreak: 'break-all',
                }}>
                  {e}
                </div>
              ))}
            </div>
          )}
        </Card>
      )}
    </div>
  )
}
