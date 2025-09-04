import { useWorkflowStore } from '../stores/workflowStore'
import './WorkflowTabs.css'

export const WorkflowTabs = () => {
    const { currentStep, setCurrentStep } = useWorkflowStore()

    const steps = [
        { key: 'data', label: 'Data', icon: '📊' },
        { key: 'features', label: 'Features', icon: '🧩' },
        { key: 'model', label: 'Model', icon: '🤖' },
        { key: 'train', label: 'Train', icon: '🚂' },
        { key: 'backtest', label: 'Backtest', icon: '🔄' },
        { key: 'evaluate', label: 'Evaluate', icon: '📈' },
        { key: 'deploy', label: 'Deployment', icon: '🚀' },
    ]

    const handleStepClick = (step) => {
        setCurrentStep(step)
    }

    return (
        <div className="workflow-tabs">
            {steps.map((step, index) => (
                <button
                    key={step.key}
                    className={`tab ${currentStep === step.key ? 'active' : ''}`}
                    onClick={() => handleStepClick(step.key)}
                >
                    <span className="tab-icon">{step.icon}</span>
                    <span className="tab-label">{step.label}</span>
                    {index < steps.length - 1 && (
                        <div className="tab-connector"></div>
                    )}
                </button>
            ))}
        </div>
    )
}
