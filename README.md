# mabinogi-mobile-helper

瑪奇 Mobile（台服 PC 版）辅助工具。数据来源只有**只读内存**与**客户区截图 / 模板 / OCR**；不修改游戏数值、不注入非目标进程、不以抓包作为数据源。

## 功能

核心是常驻桌面的「玛奇状态栏」（`watch`）：

- **界面识别**：按 UI 栈 + 模板 / OCR 显示当前所在页面（主界面、製作、加工设施、技能详情、地图等），并显示停止态、版本与读取诊断。
- **生活备料**：读取背包材料、设施队列、工具耐久与生活技能，自动生成计划并按队列执行采集、投料、领取。
  - 普通「生活备料」：失败留证后跳过，继续下一项。
  - 「生活備料（調試）」：失败暂停并保留当前项与队列，修复后「恢复」。
- **一键修理**：前往杜巴頓修理 NPC，走完对话链完成修理。
- **执行记录**：保存最近 12 小时的计划、阶段与事件明细（`logs/life_execution_12h.json`）。
- **网络提示处理**：遇到「網路不穩定。」只读等待消失；「網路連線狀態不穩定」弹窗按间隔重新验图后点「重新嘗試」。

所有键鼠输入都经过统一安全门控：输入前核对前台、遮挡、客户区与新鲜画面；同一步最多两次动作后必须验页；F8、失焦、超时立即停止并释放鼠标。

## 环境要求

- Windows 10 / 11，Python ≥ 3.11
- 瑪奇 Mobile 台服 PC 客户端，窗口客户区 1920×1440（见 `config.yaml`）
- 内存读取只支持已登记的客户端版本（`knowledge/ui/config/il2cpp_client_builds.json`）；客户端更新后会显示「待验证」并拒绝读取，需重新核对偏移后登记

## 安装

```bat
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

## 使用

启动状态栏（会先关闭本仓旧状态栏再拉起新的）：

```bat
scripts\start_state_watch.bat
```

或手动：

```bat
set PYTHONPATH=src
python -m mm_assist doctor --json
python -m mm_assist watch --restart --detach --interval 0.5
python -m mm_assist watch --install-shortcut   & rem 创建桌面快捷方式
```

鼠标异常（按键卡住 / 光标被锁）时：

```bat
set PYTHONPATH=src
python -m mm_assist mouse-unlock
```

其它子命令（采证、校验、基准、切页验收）见 `python -m mm_assist -h` 与 `src/mm_assist/cli.py`。

## 配置

`config.yaml`：

| 键 | 说明 |
|---|---|
| `window.title_substr` | 游戏窗口标题匹配 |
| `window.capture_method` | `auto` 使用 PrintWindow 渲染图 |
| `window.input_method` | 长挂默认 `unity_hook` |
| `window.input_trace` | 鼠标轨迹诊断，写入执行记录 |
| `read_only_only` | 只读档位 |

窗口坐标与 ROI 统一按客户区缩放，配置在 `knowledge/ui/config/game_frame.json`。

## 目录

```
src/mm_assist/
  memory/        IL2CPP 只读内存读取（UI 栈、设施、背包、技能、场景、位置）
  perception/    截图识别：模板、OCR、页面 / 按钮 / 弹窗定位
  life_plan/     生活备料计划器、调度控制器、状态栏 UI
  life_execute/  采集 / 投料 / 领取 / 修理执行器与页面导航
  nav/           地图定位与城镇导航
  safety/        输入门控、租约、停止与鼠标释放
  platform/      窗口、截图、输入后端
  jobs/          状态栏（state_watch）与采证任务
knowledge/
  ui/config/     IL2CPP 偏移与界面配置（整组参与版本校验摘要，勿单独修改）
  ui/contracts/  界面切换图与流程契约
  templates/     识别模板与登记元数据
  maps/          地图图集
  items/         配方与材料目录
scripts/         启动脚本与效率核对脚本
tests/           离线测试
```

## 测试

```bat
python -m pytest
```

离线测试不连接游戏。本仓库公开版未包含实机截图夹具（`tests/fixtures/`）和文档目录，依赖它们的测试会失败或需自行补采。

## 声明

仅供个人学习与研究。使用自动化工具可能违反游戏服务条款，风险自负。
