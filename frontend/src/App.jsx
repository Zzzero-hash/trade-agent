import { useWorkflowStore } from "./stores/workflowStore";
import { WorkflowTabs } from "./components/WorkflowTabs";
import { DataStep } from "./components/DataStep";

function App() {
  const currentStep = useWorkflowStore((state) => state.currentStep);

  const renderStepContent = () => {
    switch (currentStep) {
      case "data":
        return <DataStep />;
      case "features":
        return <div>Feature Engineering Step</div>;
      case "model":
        return <div>Model Selection Step</div>;
      case "train":
        return <div>Training Step</div>;
      case "evaluate":
        return <div>Evaluation Step</div>;
      case "deploy":
        return <div>Deployment Step</div>;
      default:
        return <div>Welcome! Please select a step to begin.</div>;
    }
  };

  return (
    <div className="app">
      <WorkflowTabs />
      <div className="workflow-content">{renderStepContent()}</div>
    </div>
  );
}

export default App;
