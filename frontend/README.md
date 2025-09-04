# Trade Agent Frontend

A React-based frontend for the Trade Agent ML Trading Platform.

## Features

- **ML-First Workflow**: Data → Features → Model → Train → Backtest → Evaluate → Deploy
- **Interactive UI**: Tab-based navigation with visual progress indicators
- **Real-time Feedback**: Training progress and backtest results visualization
- **Responsive Design**: Works on desktop and tablet devices

## Tech Stack

- **React 18** with Hooks
- **Zustand** for state management
- **Vite** for fast development and building
- **CSS Modules** for styling

## Getting Started

### Prerequisites

- Node.js 16+ 
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

This will start the development server on http://localhost:3000

### Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Project Structure

```
frontend/
├── src/
│   ├── components/     # React components
│   ├── stores/         # Zustand state stores
│   ├── App.jsx         # Main application component
│   ├── main.jsx        # React entry point
│   └── index.css       # Global styles
├── public/             # Static assets
├── index.html          # HTML entry point
├── vite.config.js      # Vite configuration
└── package.json        # Dependencies and scripts
```

## Workflow Steps

1. **Data**: Configure market data sources and preview data
2. **Features**: Select technical indicators and create custom features
3. **Model**: Choose ML model type and configure training parameters
4. **Train**: Monitor model training progress and metrics
5. **Backtest**: Configure and run backtesting with risk parameters
6. **Evaluate**: Analyze performance metrics and equity curves
7. **Deploy**: Deploy strategy to paper or live trading

## Development

### Adding New Components

1. Create new components in `src/components/`
2. Import and use them in `App.jsx`
3. Add any required state to `stores/workflowStore.js`

### Styling

- Use existing CSS classes from `App.css` and component CSS files
- Follow the established design system tokens
- Maintain responsive design principles

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT
