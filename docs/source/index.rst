.. trade-agent documentation master file

Welcome to trade-agent's documentation!
=======================================

Overview
--------
The trade-agent project provides a modular platform for training, evaluating, and deploying trading agents using reinforcement learning (RL) and supervised learning (SL) techniques.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Project Overview

   system_overview
   architecture_summary
   file_tree_structure
   data_pipeline_dag
   ray_parallelization_plan

.. toctree::
   :maxdepth: 2
   :caption: Agents

   agents/ppo_agent_summary
   agents/ppo_agent_detailed_plan
   agents/ppo_agent_acceptance_tests
   agents/ppo_agent_dag
   agents/ppo_agent_makefile_tasks
   agents/ppo_agent_rollback_plan
   agents/ppo_configuration
   agents/ppo_file_tree_structure
   agents/sac_agent_summary
   agents/sac_agent_detailed_plan
   agents/sac_agent_acceptance_tests
   agents/sac_agent_dag
   agents/sac_agent_implementation_overview
   agents/sac_agent_makefile_tasks
   agents/sac_agent_rollback_plan
   agents/sac_configuration
   agents/sac_file_tree_structure

.. toctree::
   :maxdepth: 2
   :caption: Data & Features

   data_handling_pipeline_plan
   interfaces
   features/index
   data/market_data
   features/feature_engineering_summary
   features/feature_engineering_dag
   features/feature_engineering_makefile_tasks
   features/feature_engineering
   features/step3_feature_engineering_detailed_plan

.. toctree::
   :maxdepth: 2
   :caption: Reinforcement Learning

   rl/index
   rl/reinforcement_learning

.. toctree::
   :maxdepth: 2
   :caption: Supervised Learning

   sl/index
   sl/supervised_learning
   sl/sl_model_dag
   sl/sl_model_acceptance_tests
   sl/sl_model_makefile_tasks
   sl/sl_model_rollback_plan
   sl/step4_sl_model_detailed_plan
   sl/training_instructions

.. toctree::
   :maxdepth: 1
   :caption: Operations

   makefile_plan
   rollback_plan
   dependency_update_plan
   acceptance_tests
   task_list
   dag_representation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Environments

   envs/rl_environment_summary
   envs/step5_rl_environment_detailed_plan
   envs/rl_environment_dag
   envs/rl_environment_makefile_tasks
   envs/rl_environment_acceptance_tests
   envs/rl_environment_rollback_plan
   envs/trading_environment

.. toctree::
   :maxdepth: 2
   :caption: Evaluation

   eval/evaluation_framework_summary
   eval/implementation_plan
   eval/evaluation_framework_design
   eval/backtesting_pipeline_design
   eval/backtesting_implementation_summary
   eval/backtesting_usage_guide
   eval/backtesting_verification_results
   eval/performance_metrics_design
   eval/risk_metrics_design
   eval/file_structure
   eval/acceptance_tests
   eval/rollback_plan
   eval/usage_guide

.. toctree::
   :maxdepth: 2
   :caption: Ensemble

   ensemble/ensemble_combiner

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
