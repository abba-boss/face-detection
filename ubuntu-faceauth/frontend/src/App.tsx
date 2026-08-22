import { Layout }    from './components/Layout'
import { Router, Switch, Route } from './components/router'
import { Dashboard } from './pages/Dashboard'
import { Users }     from './pages/Users'
import { Logs }      from './pages/Logs'
import { Doctor }    from './pages/Doctor'
import { Settings }  from './pages/Settings'

export default function App() {
  return (
    <Router>
      <Layout>
        <Switch>
          <Route path="/"        element={<Dashboard />} />
          <Route path="/users"   element={<Users />}     />
          <Route path="/logs"    element={<Logs />}      />
          <Route path="/doctor"  element={<Doctor />}    />
          <Route path="/settings" element={<Settings />} />
        </Switch>
      </Layout>
    </Router>
  )
}
