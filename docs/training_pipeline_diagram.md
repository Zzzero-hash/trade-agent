# Enhanced Training Pipeline Flow Chart (Post-Refactor)

## Mermaid Diagram: Unified Experimentation Framework

```mermaid
graph TD
    %% Configuration Layer
    A1[Experiment Config<br/>📁 src/experiments/config.py<br/>ExperimentConfig, ModelConfig, DataConfig] --> A2[CLI Interface<br/>📁 scripts/run_experiment.py<br/>Unified command-line interface]
    A1 --> A3[Hydra Integration<br/>📁 scripts/train_sl_hydra.py<br/>Enhanced Hydra with new framework]
    A2 --> A4[Training Orchestrator<br/>📁 src/experiments/orchestrator.py<br/>TrainingOrchestrator.run_full_pipeline]
    A3 --> A4

    %% Data Ingestion Layer
    A4 --> B1[Data Loader<br/>🔧 src/data/loaders.py:16-67<br/>load_ohlcv_data function]
    B1 --> B2[Data Validation<br/>🔧 src/data/loaders.py:70-110<br/>check_data_integrity function]
    B2 --> B3[Data Cleaning<br/>Forward fill missing values<br/>Timezone handling<br/>Duplicate removal]

    %% Feature Engineering Layer
    B3 --> C1[Feature Engineering<br/>🔧 src/features/build.py:1-400]
    C1 --> C2[Technical Indicators<br/>🔧 Lines 26-45: Log returns<br/>🔧 Lines 48-65: Rolling stats<br/>🔧 Lines 68-85: ATR computation]
    C1 --> C3[Market Features<br/>🔧 Lines 88-105: RSI<br/>🔧 Lines 108-125: Z-scores<br/>🔧 Lines 128-145: Realized volatility]
    C1 --> C4[Calendar Features<br/>🔧 Lines 148-165: Day/month flags<br/>🔧 Lines 168-185: Market open flags]

    C2 --> C5[Feature Matrix X<br/>All numeric features<br/>excluding targets]
    C3 --> C5
    C4 --> C5
    C5 --> C6[Target Construction<br/>mu_hat, sigma_hat<br/>Return/volatility predictions]

    %% Enhanced Cross-Validation Layer
    C5 --> D1[Purged Time Series CV<br/>🔧 src/optimize/unified_tuner.py:35-80<br/>PurgedTimeSeriesCV with embargo & purging]
    C6 --> D1
    D1 --> D2[Temporal Data Splits<br/>🔧 src/data/splits.py:82-130<br/>purged_walk_forward_splits function]
    D2 --> D3[X_train, y_train<br/>Training features & targets]
    D2 --> D4[X_val, y_val<br/>Validation features & targets]

    %% Unified Hyperparameter Optimization
    D3 --> E1[Unified Hyperparameter Tuner<br/>🔧 src/optimize/unified_tuner.py:85-150<br/>UnifiedHyperparameterTuner class]
    D4 --> E1
    E1 --> E2[Optuna Study Management<br/>🔧 Lines 110-125: Study creation<br/>🔧 Lines 160-200: Objective functions]
    E2 --> E3[SL Model Optimization<br/>🔧 Lines 150-220: _sl_objective method<br/>Grid search across model types]
    E2 --> E4[RL Agent Optimization<br/>🔧 Lines 250-320: _rl_objective method<br/>PPO/SAC hyperparameter tuning]

    %% Model Training Layer
    E3 --> F1[SL Model Factory<br/>🔧 src/sl/models/factory.py<br/>Creates model instances with best params]
    E4 --> F2[RL Training Environment<br/>🔧 src/envs/trading_env.py:100-130<br/>TradingEnvironment with optimized params]

    F1 --> F3[SL Model Training<br/>🔧 src/experiments/orchestrator.py:270-320<br/>_train_sl_model with unified metrics]
    F2 --> F4[RL Agent Training<br/>🔧 src/experiments/orchestrator.py:324-340<br/>_train_rl_model with environment]

    F3 --> F5[Trained SL Models<br/>Ridge, MLP, CNN-LSTM, etc.<br/>📁 experiments/{exp_id}/models/]
    F4 --> F6[Trained RL Agents<br/>PPO & SAC policies<br/>📁 experiments/{exp_id}/agents/]

    %% Enhanced Ensemble Framework
    F5 --> G1[Ensemble Configuration<br/>🔧 src/experiments/config.py:55-65<br/>EnsembleConfig class]
    F6 --> G1
    G1 --> G2[Ensemble Combination<br/>🔧 src/ensemble/combine.py:230-290<br/>ensemble_action function]
    G2 --> G3[Gating Model<br/>🔧 src/ensemble/combine.py:100-180<br/>Volatility-based regime detection]
    G2 --> G4[Risk Governor<br/>🔧 src/ensemble/combine.py:30-98<br/>Exposure limits & drawdown protection]

    G3 --> G5[Dynamic Weight Selection<br/>High vol → PPO preference<br/>Low vol → SAC preference]
    G4 --> G6[Risk-Constrained Actions<br/>Position limits enforced<br/>Overtrading prevention]

    %% Experiment Registry & Tracking
    F3 --> H1[Experiment Registry<br/>🔧 src/experiments/registry.py:20-80<br/>SQLite-based experiment tracking]
    F4 --> H1
    G5 --> H1
    G6 --> H1

    H1 --> H2[Results Logging<br/>🔧 Lines 125-150: log_results method<br/>Performance metrics & parameters]
    H1 --> H3[Artifact Management<br/>🔧 Lines 150-170: log_artifact method<br/>Model files & outputs]
    H1 --> H4[Experiment Querying<br/>🔧 Lines 200-250: get_best_config method<br/>Historical performance analysis]

    %% Evaluation & Backtesting
    G5 --> I1[Ensemble Action Selection<br/>🔧 src/ensemble/combine.py:230-270<br/>Weighted combination of policies]
    G6 --> I1

    I1 --> I2[Validation Environment<br/>🔧 src/ensemble/combine.py:320-360<br/>create_validation_environment]
    I2 --> I3[Episode Execution<br/>🔧 scripts/backtest_ensemble_comparison.py:180-250<br/>Step-by-step trading simulation]

    I3 --> I4[Performance Metrics<br/>🔧 scripts/backtest_ensemble_comparison.py:80-140<br/>Sharpe, drawdown, returns]
    I4 --> I5[Backtesting Engine<br/>🔧 src/eval/backtest.py:166-250<br/>P&L calculation with fees/slippage]

    I5 --> I6[Individual Policy Evaluation<br/>🔧 scripts/backtest_ensemble_comparison.py:150-200<br/>PPO vs SAC performance]
    I5 --> I7[Ensemble Evaluation<br/>🔧 scripts/backtest_ensemble_comparison.py:250-350<br/>Fixed weight vs dynamic gating]

    I6 --> J1[Comparative Analysis<br/>🔧 scripts/backtest_ensemble_comparison.py:500-600<br/>Performance comparison & validation]
    I7 --> J1

    J1 --> J2[Experiment Summary<br/>� src/experiments/registry.py:280-320<br/>get_experiment_summary method]
    J2 --> J3[Reproducible Results<br/>📁 experiments/{exp_id}/<br/>Config, models, metrics, artifacts]

    %% Styling
    classDef config fill:#e8f5e8
    classDef dataIngestion fill:#e1f5fe
    classDef featureEng fill:#f3e5f5
    classDef crossVal fill:#fff3e0
    classDef optimization fill:#fce4ec
    classDef training fill:#f1f8e9
    classDef ensemble fill:#e0f2f1
    classDef registry fill:#f9fbe7
    classDef evaluation fill:#e3f2fd
    classDef output fill:#fafafa

    class A1,A2,A3,A4 config
    class B1,B2,B3 dataIngestion
    class C1,C2,C3,C4,C5,C6 featureEng
    class D1,D2,D3,D4 crossVal
    class E1,E2,E3,E4 optimization
    class F1,F2,F3,F4,F5,F6 training
    class G1,G2,G3,G4,G5,G6 ensemble
    class H1,H2,H3,H4 registry
    class I1,I2,I3,I4,I5,I6,I7,J1 evaluation
    class J2,J3 output
```

## Enhanced Architecture Summary

### Key Improvements Implemented

#### 1. **Unified Configuration System**

- **Location**: `src/experiments/config.py`
- **Features**: Type-safe dataclasses for experiment, data, model, CV, and optimization configs
- **Benefits**: Single source of truth, validation, YAML serialization, Hydra integration

#### 2. **Consolidated Hyperparameter Optimization**

- **Location**: `src/optimize/unified_tuner.py`
- **Features**: Optuna-based tuner for both SL and RL models with advanced CV strategies
- **Benefits**: Consistent optimization across model types, purged time-series CV, study persistence

#### 3. **Advanced Cross-Validation**

- **Location**: `src/optimize/unified_tuner.py:35-80` + `src/data/splits.py`
- **Features**: Purged walk-forward splits with embargo periods and temporal gaps
- **Benefits**: Prevents data leakage, realistic backtesting conditions

#### 4. **Experiment Registry & Tracking**

- **Location**: `src/experiments/registry.py`
- **Features**: SQLite-based experiment tracking with metrics, artifacts, and reproducibility
- **Benefits**: Historical comparison, model lineage, automated experiment management

#### 5. **Training Orchestration**

- **Location**: `src/experiments/orchestrator.py`
- **Features**: End-to-end pipeline execution with error handling and result aggregation
- **Benefits**: Streamlined workflows, automated model comparison, ensemble evaluation

#### 6. **Enhanced CLI Interface**

- **Location**: `scripts/run_experiment.py`
- **Features**: Comprehensive command-line interface with experiment management
- **Benefits**: User-friendly access, batch processing, experiment querying

### Migration Path from Legacy System

#### Phase 1: Foundation (Completed)

✅ **Unified hyperparameter tuning** - Consolidated from fragmented implementations
✅ **Standardized cross-validation** - Implemented purged temporal CV
✅ **Enhanced Hydra integration** - Backward-compatible with new framework support
✅ **Experiment configuration system** - Type-safe, validatable configurations
✅ **Experiment registry** - Centralized tracking and reproducibility

#### Phase 2: Integration (In Progress)

🔄 **Model factory integration** - Consistent usage across training scripts
🔄 **Ensemble framework enhancement** - Advanced meta-learning strategies
🔄 **Performance monitoring** - Real-time metrics and alerting

#### Phase 3: Advanced Features (Future)

🔮 **Distributed optimization** - Multi-worker hyperparameter search
🔮 **Auto-ML capabilities** - Automated model selection and architecture search
🔮 **Production deployment** - Model serving and monitoring infrastructure

## Key Data Flow Points

1. **Data Validation**: Every step includes validation checks for NaN/Inf values and bounds
2. **Temporal Consistency**: All splits maintain chronological order to prevent data leakage
3. **Deterministic Processing**: Fixed seeds ensure reproducible results across all components
4. **Feature Alignment**: Sequence-based models handle different output lengths correctly
5. **Risk Management**: Multi-layered risk controls from individual model constraints to ensemble-level governors
6. **Performance Tracking**: Comprehensive metrics collection at every evaluation stage

## Configuration Files Referenced

- **conf/config.yaml**: Main Hydra configuration
- **conf/model/\*.yaml**: Individual model configurations (ridge, mlp, cnn_lstm, etc.)
- **configs/\*.json**: Legacy model configurations for backward compatibility

This pipeline ensures end-to-end traceability from raw market data to final ensemble trading performance with comprehensive validation and risk management at every stage.
