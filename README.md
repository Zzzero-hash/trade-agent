trade_agent layout:
trade_agent/
    __init__.py                   (existing)
    engine/                       (existing core execution area)
        __init__.py
        pipeline.py               (existing: pipeline orchestration)
        nodes/                    (existing container for node types)
            __init__.py
            data_source.py        (planned: market data fetch node)
            sma.py                (planned: SMA transform)
            crossover.py          (planned: SMA crossover signal)
            (future) features/    (more indicator/feature nodes)
        (future) execution.py     (higher-level execute helpers)
    backtest/                     (existing namespace reserved)
        __init__.py
        runner.py                 (planned: SimpleLongOnlyBacktester + BacktestResult)
        (future) metrics.py       (planned: separated performance metrics)
    telemetry/
        __init__.py
        core.py                   (planned: opt-in event recorder, env flag)
    flags/
        __init__.py
        store.py                   (existing: feature flag store)
    api/
        __init__.py
        app.py
        routes/
            __init__.py
            health.py
            pipelines.py
        persistence/
            __init__.py
            repository.py
            sqlite.py
        utils/
            __init__.py
            logging.py
    scripts/
        verify_release.py        (existing release gate script)
    tests/
        __init__.py
        test_engine_pipeline.py  (existing baseline)
        test_backtest_runner.py  (planned)
        test_api_health.py       (planned)
        test_version_bump.py     (existing)
        test_flags.py            (existing)
        core/
            __init__.py
            test_engine_pipeline.py (planned)
            test_metrics.py         (planned)
            test_telemetry_opt_in.py (planned)
