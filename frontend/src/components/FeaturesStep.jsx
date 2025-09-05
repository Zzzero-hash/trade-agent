import { useState } from "react";
import { useWorkflowStore } from "../stores/workflowStore";

export const FeaturesStep = () => {
  const { nextStep, previousStep, canGoNext, canGoPrevious } =
    useWorkflowStore();

  const [features, setFeatures] = useState({
    sma: { enabled: true, period: 20 },
    ema: { enabled: true, period: 12 },
    rsi: { enabled: true, period: 14 },
    bollinger: { enabled: false, period: 20, std: 2 },
    macd: { enabled: false, fast: 12, slow: 26, signal: 9 },
  });

  const toggleFeature = (featureName) => {
    setFeatures((prev) => ({
      ...prev,
      [featureName]: {
        ...prev[featureName],
        enabled: !prev[featureName].enabled,
      },
    }));
  };

  const updateFeatureParam = (featureName, paramName, value) => {
    setFeatures((prev) => ({
      ...prev,
      [featureName]: {
        ...prev[featureName],
        [paramName]: parseInt(value) || value,
      },
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    nextStep();
  };

  return (
    <div className="features-step">
      <h2>Feature Engineering</h2>
      <form onSubmit={handleSubmit}>
        {Object.entries(features).map(([featureName, config]) => (
          <div key={featureName} className="feature-config">
            <label>
              <input
                type="checkbox"
                checked={config.enabled}
                onChange={() => toggleFeature(featureName)}
              />
              {featureName.toUpperCase()}
            </label>
            {config.enabled && (
              <div className="feature-params">
                {Object.entries(config).map(([paramName, value]) => {
                  if (paramName === "enabled") return null;
                  return (
                    <div key={paramName} className="param-input">
                      <label>{paramName}:</label>
                      <input
                        type="number"
                        value={value}
                        onChange={(e) =>
                          updateFeatureParam(
                            featureName,
                            paramName,
                            e.target.value,
                          )
                        }
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ))}
        <button type="submit" disabled={!canGoNext}>
          Next
        </button>
      </form>
    </div>
  );
};
