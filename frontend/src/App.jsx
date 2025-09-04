import { WorkflowTabs } from './components/WorkflowTabs'
import './App.css'

function App() {
 return (
    <div className="app">
      <header className="app-header">
        <h1>Trade Agent - ML Trading Platform</h1>
        <p className="app-subtitle">Machine Learning Workflow for Algorithmic Trading</p>
      </header>
      <main className="app-main">
        <WorkflowTabs />
        <div className="workflow-content">
          {/* Workflow content will be rendered based on current step */}
          <div className="step-content">
            <h2>Workflow Step Content</h2>
            <p>This is where the workflow step content will be displayed.</p>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
