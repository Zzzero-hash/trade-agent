import {
  LineChart,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Line,
  Bar,
  ComposedChart,
} from "recharts";

// Simple loading spinner component
const LoadingSpinner = () => (
  <div className="loading-spinner">
    <div className="spinner"></div>
    <p>Loading data...</p>
  </div>
);

// Error display component
const ErrorDisplay = ({ error }) => (
  <div className="error-display">
    <p>❌ {error}</p>
  </div>
);

// Empty state component
const EmptyState = () => (
  <div className="empty-state">
    <p>
      No data available. Configure your data source and click "Fetch Data" to
      load chart.
    </p>
  </div>
);

// Candlestick chart component for OHLCV data
export const CandlestickChartComponent = ({ data, loading, error }) => {
  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorDisplay error={error} />;
  if (!data || data.length === 0) return <EmptyState />;

  // Transform data for candlestick chart
  const chartData = data.map((item) => ({
    ...item,
    timestamp: new Date(item.timestamp).toLocaleDateString(),
    open: parseFloat(item.open),
    high: parseFloat(item.high),
    low: parseFloat(item.low),
    close: parseFloat(item.close),
    volume: parseFloat(item.volume),
  }));

  return (
    <div className="chart-container">
      <h4>Price Chart (Candlestick)</h4>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => value}
          />
          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 12 }} />
          <Tooltip
            formatter={(value) => [Number(value).toFixed(2), "Price"]}
            labelFormatter={(value) => `Date: ${value}`}
          />
          <Legend />
          <Bar dataKey="volume" fill="#8884d8" opacity={0.3} name="Volume" />
          <Line
            type="monotone"
            dataKey="close"
            stroke="#82ca9d"
            strokeWidth={2}
            dot={false}
            name="Closing Price"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

// Simple line chart for individual metrics
export const LineChartComponent = ({
  data,
  dataKey,
  title,
  color = "#8884d8",
}) => {
  if (!data || data.length === 0) return null;

  const chartData = data.map((item) => ({
    timestamp: new Date(item.timestamp).toLocaleDateString(),
    value: parseFloat(item[dataKey]),
  }));

  return (
    <div className="chart-container">
      <h4>{title}</h4>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
            tick={{ fontSize: 10 }}
            tickFormatter={(value) => value}
          />
          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10 }} />
          <Tooltip formatter={(value) => [Number(value).toFixed(2), title]} />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// Data summary component
export const DataSummary = ({ data }) => {
  if (!data || data.length === 0) return null;

  const latest = data[data.length - 1];
  const first = data[0];

  const priceChange = (
    ((latest.close - first.close) / first.close) *
    100
  ).toFixed(2);
  const maxPrice = Math.max(...data.map((d) => d.high));
  const minPrice = Math.min(...data.map((d) => d.low));
  const totalVolume = data.reduce((sum, d) => sum + d.volume, 0);

  return (
    <div className="data-summary">
      <div className="summary-grid">
        <div className="summary-item">
          <span className="summary-label">Latest Close</span>
          <span className="summary-value">${latest.close.toFixed(2)}</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Price Change</span>
          <span
            className={`summary-value ${priceChange >= 0 ? "positive" : "negative"}`}
          >
            {priceChange}%
          </span>
        </div>
        <div className="summary-item">
          <span className="summary-label">High/Low</span>
          <span className="summary-value">
            ${maxPrice.toFixed(2)} / ${minPrice.toFixed(2)}
          </span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Total Volume</span>
          <span className="summary-value">{totalVolume.toLocaleString()}</span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Data Points</span>
          <span className="summary-value">{data.length}</span>
        </div>
      </div>
    </div>
  );
};
