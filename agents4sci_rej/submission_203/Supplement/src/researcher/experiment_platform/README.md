# YuLan-OneSim Experimental Platform

YuLan-OneSim实验平台提供了一套完整的受控实验工具，支持Agent profile干预、实验配置生成和系统化的实验管理。

## 概述

实验平台支持以下核心功能：
- **Profile干预**：在模拟开始前修改Agent profiles
- **实验配置生成**：为每个实验组生成专用配置文件
- **实验管理**：完整的实验生命周期管理
- **结果追踪**：详细的修改记录和对比分析

## 模块结构

```
experiment_platform/
├── __init__.py                 # 模块入口
├── profile_modifier.py         # Profile修改核心引擎
├── intervention_engine.py      # 干预规范解析和应用
├── config_generator.py         # 实验配置生成器
├── experiment_project.py       # 实验项目协调器
├── test_integration.py         # 集成测试
├── simple_test.py              # 简化测试
└── README.md                   # 本文档
```

## 核心组件

### 1. ProfileModifier

Profile修改的核心引擎，支持多种修改类型和验证机制。

#### 主要接口

```python
class ProfileModifier:
    def __init__(self, model_name: Optional[str] = None)
    
    def modify_profiles(
        self,
        profiles: List[AgentProfile],
        modifications: List[ModificationSpec],
        target_agents: Optional[List[str]] = None,
        target_percentage: Optional[float] = None,
        selection_strategy: str = "random",
        llm_enhancement: bool = False
    ) -> List[ModificationResult]
    
    def save_modified_profiles(
        self,
        results: List[ModificationResult],
        output_dir: Union[str, Path],
        create_originals: bool = True
    ) -> Dict[str, str]
    
    def create_profile_data_files(
        self,
        results: List[ModificationResult],
        agent_types: Dict[str, List[str]],
        output_dir: Union[str, Path]
    ) -> Dict[str, str]
```

#### 支持的修改类型

- `set_value`: 设置字段为指定值
- `add_value`: 在原值基础上增加数值
- `multiply_by`: 将原值乘以指定倍数
- `llm_generate`: 使用LLM生成新值
- `add_field`: 添加新字段到profile和schema
- `remove_field`: 从profile和schema中删除字段

#### 目标选择策略

- `random_sample`: 随机选择指定百分比的agents
- `all`: 选择所有agents
- `specific_agents`: 指定特定agent IDs
- `by_agent_type`: 根据agent类型批量选择
- `by_profile_criteria`: 根据profile属性筛选agents

### 2. InterventionEngine

干预规范的解析器和执行器，管理实验干预的应用。

#### 主要接口

```python
class InterventionEngine:
    def __init__(self, model_name: Optional[str] = None)
    
    def load_intervention_specifications(self, spec_file: Union[str, Path]) -> None
    
    def apply_intervention(
        self,
        intervention_id: str,
        profiles: List[AgentProfile],
        target_selection: Dict[str, Any],
        llm_enhancement: bool = False
    ) -> InterventionResult
    
    def apply_multiple_interventions(
        self,
        intervention_configs: List[Dict[str, Any]],
        profiles: List[AgentProfile],
        llm_enhancement: bool = False
    ) -> List[InterventionResult]
    
    def get_available_interventions(self) -> Dict[str, Dict[str, Any]]
```

### 3. ConfigGenerator

为不同实验组生成专用的OneSim配置文件。

#### 主要接口

```python
class ConfigGenerator:
    def __init__(self, base_config_path: Union[str, Path])
    
    def generate_experiment_configs(
        self,
        experiment_groups: List[ExperimentGroupConfig],
        output_dir: Union[str, Path],
        modified_profile_paths: Dict[str, Dict[str, str]] = None,
        environment_name: Optional[str] = None
    ) -> Dict[str, str]
    
    def validate_config(self, config_path: Union[str, Path]) -> Dict[str, Any]
    
    def generate_batch_configs(
        self,
        batch_specification: Dict[str, Any],
        output_dir: Union[str, Path]
    ) -> Dict[str, List[str]]
```

### 4. RuntimeInterventionEngine

运行时干预引擎，支持模拟运行过程中的实时干预。

#### 主要接口

```python
class RuntimeInterventionEngine:
    def __init__(self)
    
    def load_runtime_interventions(self, spec_file: Union[str, Path]) -> None
    
    def execute_scheduled_interventions(
        self,
        current_time_step: int,
        simulation_context: Dict[str, Any]
    ) -> List[RuntimeInterventionResult]
    
    def check_conditional_interventions(
        self,
        simulation_context: Dict[str, Any],
        current_time_step: int
    ) -> List[RuntimeInterventionResult]
    
    def manually_execute_intervention(
        self,
        intervention_id: str,
        simulation_context: Dict[str, Any],
        current_time_step: int
    ) -> RuntimeInterventionResult
```

#### 支持的干预类型

- **Agent干预**: 修改Agent的属性值
- **Environment干预**: 修改环境参数
- **Broadcast干预**: 向目标Agent群体广播消息

#### 干预时机

- `scheduled`: 在指定时间步执行
- `conditional`: 当满足条件时执行
- `manual`: 手动触发执行

### 5. ExperimentProject

实验项目的主协调器，整合所有组件提供完整的工作流程。

#### 主要接口

```python
class ExperimentProject:
    def __init__(self, project_config: ProjectConfig)
    
    def load_base_profiles(self, environment_path: Union[str, Path]) -> Dict[str, List[AgentProfile]]
    
    def design_experiment(self, experiment_design: Dict[str, Any]) -> List[ExperimentGroupConfig]
    
    def apply_interventions(
        self,
        experiment_groups: List[ExperimentGroupConfig],
        intervention_configs: List[Dict[str, Any]]
    ) -> Dict[str, InterventionResult]
    
    def generate_configurations(
        self,
        experiment_groups: List[ExperimentGroupConfig]
    ) -> Dict[str, str]
    
    def run_full_workflow(
        self,
        environment_path: Union[str, Path],
        experiment_design: Dict[str, Any],
        intervention_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]
```

## 实验工作流程

### 阶段1: 场景和实验设计

在进行模拟实验之前，需要在实验设计阶段生成以下关键文件：

#### 必需文件列表

##### 1. `experiment_config.json` - 实验总体配置 (纯实验设计)
```json
{
  "experiment_info": {
    "name": "experiment_name",
    "title": "实验标题",
    "description": "实验描述",
    "research_question": "研究问题",
    "created_date": "2025-07-01"
  },
  "base_scenario": {
    "environment": "environment_name",
    "source_path": "src/envs/environment_name",
    "agent_count": 50,
    "simulation_steps": 100,
    "base_profile_template": "default_agent"
  },
  "experimental_groups": {
    "control_group": {
      "name": "control_group_001",
      "description": "基线场景",    
      "parameters": {"random_seed": 12345}
    },
    "treatment_groups": [
      {
        "name": "treatment_group_001",
        "description": "干预组1描述",
        "parameters": {"random_seed": 12345}
      }
    ],
    "replicates": [
      {
        "name": "replicate_001",
        "description": "重复实验",
        "base_treatment": "treatment_group_001",
        "parameters": {"random_seed": 54321}
      }
    ]
  },
  "metrics": {
    "primary": ["metric1", "metric2"],
    "secondary": ["metric3", "metric4"]
  },
  "analysis_config": {
    "statistical_tests": ["t_test", "anova"],
    "confidence_level": 0.95,
    "comparison_groups": [
      {
        "name": "control_vs_treatment1",
        "groups": ["control_group_001", "treatment_group_001"]
      }
    ]
  }
}
```

##### 2. `intervention_specifications.json` - 完整干预规范定义
```json
{
  "treatment_groups": {
    "treatment_group_001": {
      "profile_modifications": [
        {
          "target_selection": {
            "method": "by_agent_type",
            "agent_types": ["CulturalAgent"],
            "percentage": 0.6
          },
          "modifications": [
            {
              "field": "personality_trait",
              "modification_type": "set_value",
              "value": "open",
              "validation": {"type": "str"}
            },
            {
              "field": "new_cultural_openness",
              "modification_type": "add_field",
              "value": 0.8,
              "validation": {"min": 0.0, "max": 1.0, "type": "float"},
              "field_config": {
                "type": "float",
                "default": 0.5,
                "private": false,
                "sampling": "random",
                "range": [0.0, 1.0],
                "description": "Openness to adopting new cultural traits"
              }
            }
          ],
          "llm_prompt_template": "Modify this agent to be more culturally open..."
        }
      ]
    },
    "treatment_group_003": {
      "runtime_modifications": [
        {
          "intervention_type": "broadcast",
          "target_type": "agent",
          "timing": {
            "type": "scheduled",
            "schedule": [10, 20]
          },
          "target_selection": {
            "method": "all"
          },
          "config": {
            "message": "A major cultural festival is happening!",
            "message_type": "cultural_event",
            "message_data": {
              "event_type": "cultural_festival",
              "influence_strength": 0.3
            }
          }
        },
        {
          "intervention_type": "attribute_modification",
          "target_type": "agent",
          "timing": {
            "type": "conditional",
            "condition": "step > 15"
          },
          "target_selection": {
            "method": "random_sample",
            "percentage": 0.4
          },
          "config": {
            "modifications": [
              {
                "field": "music_preference",
                "modification_type": "set_value",
                "value": "Electronic"
              }
            ]
          }
        },
        {
          "intervention_type": "attribute_modification",
          "target_type": "environment",
          "timing": {
            "type": "scheduled",
            "schedule": [25]
          },
          "target_selection": {
            "method": "environment_global"
          },
          "config": {
            "modifications": [
              {
                "field": "cultural_climate",
                "modification_type": "set_value",
                "value": "progressive"
              }
            ]
          }
        }
      ]
    }
  },
  "target_selection_strategies": {
    "random_sample": {
      "description": "随机选择agents",
      "parameters": {
        "percentage": "required",
        "seed": "optional"
      }
    },
    "by_agent_type": {
      "description": "根据agent类型选择",
      "parameters": {
        "agent_types": "required",
        "percentage": "optional"
      }
    },
    "by_profile_criteria": {
      "description": "根据profile属性筛选",
      "parameters": {
        "criteria": {
          "field_path": {
            "type": "equals|greater_than|less_than|in_range|contains|in_list|regex",
            "value": "comparison_value"
          }
        }
      }
    },
    "environment_global": {
      "description": "全局环境目标",
      "parameters": {}
    }
  }
}
```

##### 3. `scenario_config.json` - 基线场景配置
```json
{
  "scenario_info": {
    "name": "scenario_baseline",
    "description": "基线场景配置",
    "environment_source": "environment_name",
    "created_date": "2025-07-01"
  },
  "simulation_config": {
    "mode": "ROUND|TIMED",
    "max_steps": 100,
    "step_duration": 1.0,
    "termination_conditions": [
      {
        "type": "max_steps_reached",
        "value": 100
      }
    ]
  },
  "agent_config": {
    "total_count": 50,
    "types": {
      "AgentType1": {
        "count": 30,
        "profile_template": "default_template",
        "schema_path": "profile/schema/AgentType1.json"
      },
      "AgentType2": {
        "count": 20,
        "profile_template": "default_template",
        "schema_path": "profile/schema/AgentType2.json"
      }
    },
    "relationship_config": {
      "initial_connections": 5,
      "connection_probability": 0.1
    }
  },
  "environment_config": {
    "environment_specific_parameters": {
      "param1": "value1",
      "param2": "value2"
    }
  },
  "metrics_config": {
    "collection_interval": 1,
    "metrics": [
      {
        "name": "metric_name",
        "type": "metric_type",
        "description": "指标描述"
      }
    ]
  }
}
```

#### 文件传递规范

实验设计阶段生成的文件将按以下方式传递给模拟执行阶段：

##### 输入接口
```python
def execute_experiment_simulation(
    project_directory: str,              # 实验项目根目录
    experiment_config_path: str,         # experiment_config.json路径
    intervention_specs_path: str,        # intervention_specifications.json路径
    base_scenario_config_path: str,      # scenario_config.json路径
    base_environment_path: str,          # 基础环境路径
    output_directory: str,               # 输出目录
    execution_config: Dict[str, Any]     # 执行配置
) -> Dict[str, Any]:
    """
    执行实验模拟
    
    返回:
    {
        "status": "success|failed",
        "experiment_groups": ["group1", "group2"],
        "generated_configs": {"group1": "config_path1"},
        "modified_profiles": {"group1": "profile_dir1"},
        "execution_results": {"group1": "result_path1"},
        "summary": "experiment_summary.json"
    }
    """
```

##### 传递流程
1. **配置验证阶段**
   - 验证所有必需文件存在
   - 检查文件格式和内容完整性
   - 验证干预规范和实验设计的一致性

2. **Profile处理阶段**
   - 加载基础environment的profiles
   - 根据intervention_specifications.json应用干预
   - 为每个实验组生成修改后的profile文件

3. **配置生成阶段**
   - 基于scenario_config.json生成基础配置
   - 为每个实验组创建专用配置文件
   - 配置中指向对应的profile文件路径

4. **执行准备阶段**
   - 创建每个实验组的运行目录
   - 复制必要的环境文件
   - 生成执行脚本和监控配置

### 阶段2: 模拟实验执行

#### 生成的文件结构
```
project_directory/
├── experiment_config.json              # 从阶段1传入
├── intervention_specifications.json    # 从阶段1传入
├── base_scenario/
│   └── scenario_config.json           # 从阶段1传入
├── groups/                             # 实验组织结构
│   ├── control_group_001/
│   │   ├── config.json                 # 固定配置文件，每次运行都使用
│   │   ├── profile/                    # 固定profiles，每次运行都使用
│   │   │   ├── data/                   # profile数据文件
│   │   │   │   ├── AgentType1.json
│   │   │   │   └── AgentType2.json
│   │   │   └── schema/                 # schema文件
│   │   └── runs/                       # 每次运行的输出记录
│   │       ├── 20250703_143022/        # 时间戳格式: YYYYMMDD_HHMMSS
│   │       │   ├── simulation_log.json # 模拟执行日志
│   │       │   ├── results/            # 模拟结果
│   │       │   └── output/             # 其他运行输出
│   │       └── 20250703_150134/        # 另一次运行
│   │           ├── simulation_log.json
│   │           ├── results/
│   │           └── output/
│   └── treatment_group_001/
│       ├── config.json                 # 固定配置文件，每次运行都使用
│       ├── profile/                    # 固定profiles（应用干预后），每次运行都使用
│       │   ├── data/                   # profile数据文件
│       │   │   ├── AgentType1.json
│       │   │   └── AgentType2.json
│       │   └── schema/                 # schema文件
│       ├── intervention_config.json    # 运行时干预配置（如有）
│       └── runs/                       # 每次运行的输出记录
│           ├── 20250703_143022/        # 与对照组同时运行
│           │   ├── simulation_log.json
│           │   ├── results/
│           │   └── output/
│           └── 20250703_150134/        # 另一次运行
│               ├── simulation_log.json
│               ├── results/
│               └── output/
└── project_summary.json               # 执行总结
```

## 使用示例

### 基本使用流程

```python
# 1. 创建项目配置
project_config = ProjectConfig(
    project_name="culture_dissemination_experiment",
    project_description="文化传播实验",
    base_environment="culture_dissemination",
    base_config_path="config/base_config.json",
    intervention_specs_path="experiment_design/intervention_specifications.json",
    output_directory="experiments/culture_dissemination_impact"
)

# 2. 初始化实验项目
project = ExperimentProject(project_config)

# 3. 执行完整工作流程
workflow_result = project.run_full_workflow(
    environment_path="src/envs/culture_dissemination",
    experiment_design=experiment_design_dict,
    intervention_configs=intervention_configs_list
)

# 4. 检查结果
if workflow_result['status'] == 'completed':
    print(f"实验配置已生成: {workflow_result['configurations_generated']}")
    print(f"输出目录: {workflow_result['output_directory']}")
```

### 单独使用各组件

```python
# 仅使用Profile修改（包含新功能）
modifier = ProfileModifier(model_name="gpt-4o")

# 添加新字段的修改规范
add_field_spec = ModificationSpec(
    field="new_social_score",
    modification_type="add_field",
    value=0.5,
    validation={"min": 0.0, "max": 1.0, "type": "float"},
    field_config={
        "type": "float",
        "default": 0.5,
        "private": false,
        "sampling": "random",
        "range": [0.0, 1.0],
        "description": "新的社交评分字段"
    }
)

# 按agent类型选择目标
results = modifier.modify_profiles(
    profiles=agent_profiles,
    modifications=[add_field_spec],
    target_selection={
        "method": "by_agent_type",
        "agent_types": ["SocialAgent", "MediaAgent"],
        "percentage": 0.5
    }
)

# 按profile属性筛选目标
criteria_selection = {
    "method": "by_profile_criteria",
    "criteria": {
        "cultural_openness": {
            "type": "greater_than",
            "value": 0.7
        },
        "age": {
            "type": "in_range", 
            "value": [25, 65]
        }
    }
}

results2 = modifier.modify_profiles(
    profiles=agent_profiles,
    modifications=modification_specs,
    target_selection=criteria_selection
)

# 仅使用配置生成
generator = ConfigGenerator("base_config.json")
configs = generator.generate_experiment_configs(
    experiment_groups=groups,
    output_dir="output/configs"
)

# 运行时干预
runtime_engine = RuntimeInterventionEngine()
runtime_engine.load_runtime_interventions("runtime_interventions.json")

# 在模拟循环中执行干预
for step in range(simulation_steps):
    # 执行计划干预
    scheduled_results = runtime_engine.execute_scheduled_interventions(
        current_time_step=step,
        simulation_context={
            'current_step': step,
            'agents': agents,
            'environment': environment,
            'metrics': current_metrics
        }
    )
    
    # 检查条件干预
    conditional_results = runtime_engine.check_conditional_interventions(
        simulation_context=simulation_context,
        current_time_step=step
    )
```

## 配置文件模板

### 基础配置模板
完整的OneSim配置文件模板，包含所有必要的sections：

```json
{
  "simulator": {
    "environment": {
      "name": "environment_name",
      "mode": "round",
      "max_steps": 100,
      "interval": 1.0,
      "export_training_data": false,
      "export_event_data": true
    }
  },
  "agent": {
    "planning": "COTPlanning",
    "memory": {
      "strategy": "ShortLongStrategy",
      "storages": {
        "short_term_storage": {
          "class": "ListMemoryStorage",
          "capacity": 100
        }
      }
    },
    "profile_config": {
      "use_custom_profiles": true,
      "custom_profile_paths": {
        "AgentType1": "path/to/modified/AgentType1.json",
        "AgentType2": "path/to/modified/AgentType2.json"
      }
    }
  },
  "monitor": {
    "enabled": true,
    "update_interval": 30
  },
  "experiment_metadata": {
    "group_id": "treatment_group_001",
    "group_type": "treatment",
    "description": "实验组描述",
    "profile_modifications": ["intervention_id"]
  }
}
```

## 扩展和自定义

### 添加新的修改类型

```python
# 在ProfileModifier中添加新的modification_type
def _apply_modification(self, profile: AgentProfile, mod_spec: ModificationSpec) -> bool:
    # ... 现有代码 ...
    elif mod_spec.modification_type == "custom_modification":
        new_value = self._custom_modification_logic(current_value, mod_spec)
    # ...
```

### 添加新的目标选择策略

```python
# 在InterventionEngine中添加新的selection_method
def _select_target_agents(self, profiles, target_selection):
    # ... 现有代码 ...
    elif selection_method == 'custom_strategy':
        selected_profiles = self._custom_selection_logic(profiles, target_selection)
    # ...
```

## 错误处理和验证

系统提供多层验证机制：

1. **配置文件验证** - 检查JSON格式和必需字段
2. **Profile修改验证** - 验证字段类型和值范围
3. **干预规范验证** - 检查干预定义的完整性
4. **实验设计验证** - 验证实验组配置的一致性

## 性能优化

- **批量处理** - 支持批量profile修改和配置生成
- **缓存机制** - 避免重复加载大型profile文件
- **并行处理** - 支持多进程profile修改（未来版本）
- **增量更新** - 仅修改变更的profile字段

## 注意事项

1. **文件路径** - 所有路径使用绝对路径，确保跨环境兼容性
2. **版本控制** - 建议对实验配置文件进行版本控制
3. **备份原始数据** - 系统自动备份原始profile数据
4. **资源管理** - 大规模实验时注意内存和存储空间
5. **安全性** - LLM生成的内容需要验证，避免注入攻击

## 最终统一配置结构设计

### 完全职责分离的配置架构

经过系统性重构，实现了完全无重复的配置结构：

#### ✅ **最终架构设计**

##### 1. `experiment_config.json` - 纯实验设计
**职责**: 仅包含实验元数据、分组定义、指标配置
```
experiment_config.json:
├── experiment_info                    # ✅ 实验元信息
├── base_scenario                      # ✅ 基础场景配置
├── experimental_groups                # ✅ 实验组定义
│   ├── control_group                  # ✅ 对照组
│   ├── treatment_groups[]             # ✅ 治疗组 (无干预细节)
│   └── replicates[]                   # ✅ 重复实验组
├── metrics                           # ✅ 指标定义
└── analysis_config                   # ✅ 分析配置
```

##### 2. `intervention_specifications.json` - 完整干预规范
**职责**: 包含所有干预的完整定义，包括目标选择
```
intervention_specifications.json:
├── treatment_groups                   # ✅ 按治疗组组织
│   ├── treatment_group_001
│   │   ├── profile_modifications[]   # ✅ 预模拟干预
│   │   │   ├── target_selection     # ✅ 每个干预的目标选择
│   │   │   ├── modifications[]      # ✅ 具体修改操作
│   │   │   └── llm_prompt_template  # ✅ LLM提示模板
│   │   └── runtime_modifications[]  # ✅ 运行时干预
│   │       ├── intervention_type    # ✅ 干预类型
│   │       ├── target_type          # ✅ 目标实体类型
│   │       ├── target_selection     # ✅ 目标选择策略
│   │       ├── timing               # ✅ 执行时机
│   │       └── config               # ✅ 干预配置
└── target_selection_strategies       # ✅ 选择策略定义
```

**运行时干预配置自动生成**:
- 如果某个治疗组包含`runtime_modifications`，系统会自动生成`groups/{group_id}/intervention_config.json`
- 同时在`groups/{group_id}/config.json`中添加`intervention_config`配置项指向该文件

### 🎯 **关键设计原则**

#### 统一结构模式
两种干预类型完全采用统一的字段结构：

**Profile Modifications**:
```json
{
  "target_selection": {...},           // 统一的目标选择
  "modifications": [...],              // 字段修改列表
  "llm_prompt_template": "..."         // LLM模板
}
```

**Runtime Modifications**:
```json
{
  "intervention_type": "broadcast|attribute_modification",
  "target_type": "agent|environment",
  "target_selection": {...},           // 统一的目标选择
  "timing": {...},
  "config": {...}
}
```

#### 完全职责分离
- **experiment_config.json**: 零干预细节，纯实验设计
- **intervention_specifications.json**: 包含所有干预定义和目标选择

#### 灵活性最大化
- 每个具体干预都有自己的目标选择策略
- 支持一个治疗组包含多种不同的干预
- 支持不同干预针对不同的目标群体

### 📋 **干预类型规范**

**干预类型** (`intervention_type`):
- `broadcast`: 广播消息类干预
- `attribute_modification`: 属性修改类干预

**目标实体** (`target_type`):
- `agent`: 针对智能体
- `environment`: 针对环境

## 故障排除

### 常见问题

1. **ModuleNotFoundError** - 确保正确设置Python路径
2. **配置验证失败** - 检查JSON格式和必需字段
3. **Profile修改失败** - 验证字段名称和数据类型
4. **LLM生成错误** - 检查模型配置和网络连接
5. **配置文件职责混乱** - 确保experiment_config.json仅包含引用，详细规范在intervention_specifications.json中

### 调试技巧

- 启用详细日志记录
- 使用验证功能检查配置
- 检查生成的中间文件
- 使用simple_test.py进行独立测试
- 验证两个配置文件间的ID引用一致性