import { useState } from "react";
import { useWorkflowStore } from "../stores/workflowStore";
import {
  CandlestickChartComponent,
  LineChartComponent,
  DataSummary,
} from "./ChartVisualization";

export const DataStep = () => {
  const { nextStep, previousStep, canGoPrevious } = useWorkflowStore();

  const [formData, setFormData] = useState({
    dataSource: "yfinance",
    filePath: "",
    symbol: "AAPL",
    timeframe: "1h",
    startDate: "",
    endDate: "",
    yfPeriod: "1y",
    yfInterval: "1d",
  });

  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dataFetched, setDataFetched] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Here you would typically save the data or make an API call
    if (dataFetched) {
      nextStep();
    }
  };

  const handleFetchData = async () => {
    if (!formData.symbol) {
      setError("Please enter a symbol");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Use the new smart fetch endpoint that checks local storage first
      const response = await fetch(`http://localhost:8000/data/fetch`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          symbol: formData.symbol,
          period: formData.yfPeriod,
          interval: formData.yfInterval,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          `Failed to fetch data: ${errorData.detail || response.statusText}`,
        );
      }

      const result = await response.json();
      setChartData(result.data);
      setDataFetched(true);

      // Add success notification with source information
      console.log(
        `Successfully fetched ${result.rows} data points for ${formData.symbol} from ${result.source}`,
      );
    } catch (err) {
      setError(err.message);
      console.error("Error fetching data:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveData = async () => {
    if (!dataFetched || chartData.length === 0) {
      setError("No data to save. Please fetch data first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Save data to backend
      const response = await fetch(`http://localhost:8000/data/write`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          symbol: formData.symbol,
          timeframe: formData.yfInterval,
          data: chartData,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to save data: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("Data saved successfully:", result);

      // Move to next step after successful save
      nextStep();
    } catch (err) {
      setError(err.message);
      console.error("Error saving data:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="step-content">
      <h2>📊 RL Data Configuration</h2>
      <p>
        Configure and visualize market data for reinforcement learning training.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="step-grid">
          {/* Data Source Card */}
          <div className="card">
            <h3>Data Source</h3>
            <div className="form-group">
              <label>Source Type</label>
              <select
                name="dataSource"
                value={formData.dataSource}
                onChange={handleInputChange}
                className="form-control"
              >
                <option value="csv">CSV File</option>
                <option value="api">API Connection</option>
                <option value="database">Database</option>
                <option value="simulator">Market Simulator</option>
                <option value="yfinance">Yahoo Finance</option>
              </select>
            </div>

            {formData.dataSource === "csv" && (
              <div className="form-group">
                <label>File Path</label>
                <input
                  type="text"
                  name="filePath"
                  value={formData.filePath}
                  onChange={handleInputChange}
                  placeholder="/path/to/data.csv"
                  className="form-control"
                />
              </div>
            )}

            {formData.dataSource === "yfinance" && (
              <div className="form-group">
                <label>Ticker Symbol</label>
                <input
                  type="text"
                  name="symbol"
                  value={formData.symbol}
                  onChange={handleInputChange}
                  placeholder="e.g., AAPL, MSFT"
                  className="form-control"
                />
              </div>
            )}

            {formData.dataSource === "api" && (
              <div className="form-group">
                <label>API Endpoint</label>
                <input
                  type="text"
                  name="apiEndpoint"
                  value={formData.apiEndpoint || ""}
                  onChange={handleInputChange}
                  placeholder="https://api.example.com/data"
                  className="form-control"
                />
              </div>
            )}
          </div>

          {/* Parameters Card */}
          <div className="card">
            <h3>RL Parameters</h3>
            <div className="form-grid">
              {formData.dataSource === "yfinance" ? (
                <>
                  <div className="form-group">
                    <label>Period</label>
                    <select
                      name="yfPeriod"
                      value={formData.yfPeriod}
                      onChange={handleInputChange}
                      className="form-control"
                    >
                      <option value="1d">1 Day</option>
                      <option value="5d">5 Days</option>
                      <option value="1mo">1 Month</option>
                      <option value="3mo">3 Months</option>
                      <option value="6mo">6 Months</option>
                      <option value="1y">1 Year</option>
                      <option value="2y">2 Years</option>
                      <option value="5y">5 Years</option>
                      <option value="10y">10 Years</option>
                      <option value="ytd">Year to Date</option>
                      <option value="max">Maximum</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Interval</label>
                    <select
                      name="yfInterval"
                      value={formData.yfInterval}
                      onChange={handleInputChange}
                      className="form-control"
                    >
                      <option value="1m">1 Minute</option>
                      <option value="2m">2 Minutes</option>
                      <option value="5m">5 Minutes</option>
                      <option value="15m">15 Minutes</option>
                      <option value="30m">30 Minutes</option>
                      <option value="60m">60 Minutes</option>
                      <option value="90m">90 Minutes</option>
                      <option value="1h">1 Hour</option>
                      <option value="1d">1 Day</option>
                      <option value="5d">5 Days</option>
                      <option value="1wk">1 Week</option>
                      <option value="1mo">1 Month</option>
                      <option value="3mo">3 Months</option>
                    </select>
                  </div>
                </>
              ) : (
                <>
                  <div className="form-group">
                    <label>Symbol</label>
                    <input
                      type="text"
                      name="symbol"
                      value={formData.symbol}
                      onChange={handleInputChange}
                      placeholder="e.g., BTCUSD, AAPL"
                      className="form-control"
                    />
                  </div>

                  <div className="form-group">
                    <label>Timeframe</label>
                    <select
                      name="timeframe"
                      value={formData.timeframe}
                      onChange={handleInputChange}
                      className="form-control"
                    >
                      <option value="1m">1 Minute</option>
                      <option value="5m">5 Minutes</option>
                      <option value="15m">15 Minutes</option>
                      <option value="1h">1 Hour</option>
                      <option value="4h">4 Hours</option>
                      <option value="1d">1 Day</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Start Date</label>
                    <input
                      type="date"
                      name="startDate"
                      value={formData.startDate}
                      onChange={handleInputChange}
                      className="form-control"
                    />
                  </div>

                  <div className="form-group">
                    <label>End Date</label>
                    <input
                      type="date"
                      name="endDate"
                      value={formData.endDate}
                      onChange={handleInputChange}
                      className="form-control"
                    />
                  </div>
                </>
              )}
            </div>

            {/* Action Buttons */}
            <div className="form-actions" style={{ marginTop: "1rem" }}>
              <button
                type="button"
                onClick={handleFetchData}
                disabled={loading || !formData.symbol}
                className="btn-primary"
                style={{ marginRight: "1rem" }}
              >
                {loading ? "Fetching..." : "Fetch Data"}
              </button>

              {dataFetched && (
                <button
                  type="button"
                  onClick={handleSaveData}
                  disabled={loading}
                  className="btn-success"
                >
                  {loading ? "Saving..." : "Save & Continue"}
                </button>
              )}
            </div>
          </div>

          {/* Data Preview Card */}
          <div className="card full-width">
            <h3>RL Data Preview & Analysis</h3>
            <div className="data-preview-container">
              {error && (
                <div
                  className="error-message"
                  style={{ color: "red", marginBottom: "1rem" }}
                >
                  ❌ {error}
                </div>
              )}

              <DataSummary data={chartData} />

              <div className="chart-section">
                <CandlestickChartComponent
                  data={chartData}
                  loading={loading}
                  error={error}
                />
              </div>

              <div className="chart-grid-container">
                <div className="chart-item">
                  <LineChartComponent
                    data={chartData}
                    dataKey="close"
                    title="Closing Price"
                    color="#82ca9d"
                  />
                </div>
                <div className="chart-item">
                  <LineChartComponent
                    data={chartData}
                    dataKey="volume"
                    title="Volume"
                    color="#8884d8"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Buttons */}
        <div
          className="form-actions"
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: "2rem",
          }}
        >
          <button
            type="button"
            onClick={previousStep}
            disabled={!canGoPrevious()}
            className="btn-secondary"
          >
            ← Previous
          </button>
          <button
            type="submit"
            disabled={!dataFetched || loading}
            className="btn-primary"
          >
            Next: Features →
          </button>
        </div>
      </form>
    </div>
  );
};
