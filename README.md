# Class2 Railway 部署版：企业智能客服

这是第二课 Web 智能客服的 Railway 部署版本，不包含 RAG。

部署成功后，Railway 会提供一个公网地址，同学之间可以互相访问。

## 1. 项目结构

```text
class2-railway/
  app.py
  static/
    index.html
  requirements.txt
  railway.toml
  Procfile
  runtime.txt
  README.md
```

说明：

- `app.py`：FastAPI 后端
- `static/index.html`：前端聊天页面
- `requirements.txt`：Python 依赖
- `railway.toml`：Railway 部署配置
- `Procfile`：告诉 Railway 如何启动服务
- `runtime.txt`：指定 Python 版本

## 2. 本地运行

进入目录：

```bash
cd class2-railway
```

安装依赖：

```bash
pip3 install -r requirements.txt
```

配置环境变量：

```bash
export DASHSCOPE_API_KEY="sk-你的阿里云百炼APIKey"
export LLM_PROVIDER="qwen"
```

启动：

```bash
python3 -m uvicorn app:app --reload
```

打开：

```text
http://127.0.0.1:8000
```

如果想使用 DeepSeek：

```bash
export DEEPSEEK_API_KEY="sk-你的DeepSeekAPIKey"
export LLM_PROVIDER="deepseek"
```

## 3. Railway 部署流程

推荐每个学生把 `class2-railway` 单独放进一个 GitHub 仓库。

如果使用一个大仓库，Railway 里需要把 Root Directory 设置为：

```text
class2-railway
```

部署步骤：

1. 打开 Railway。
2. 使用 GitHub 登录。
3. 点击 `New Project`。
4. 选择 `Deploy from GitHub repo`。
5. 选择自己的项目仓库。
6. 如果是大仓库，设置 Root Directory 为 `class2-railway`。
7. 在 `Variables` 里添加环境变量。

使用通义千问：

```text
DASHSCOPE_API_KEY=sk-你的阿里云百炼APIKey
LLM_PROVIDER=qwen
```

使用 DeepSeek：

```text
DEEPSEEK_API_KEY=sk-你的DeepSeekAPIKey
LLM_PROVIDER=deepseek
```

8. Railway 自动部署。
9. 进入服务的 `Settings` 或 `Networking`，生成公网域名。
10. 打开 Railway 提供的网址测试聊天功能。

## 4. 启动命令

Railway 会优先读取 `railway.toml`：

```toml
startCommand = "python -m uvicorn app:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
```

项目里也保留了 `Procfile`：

```text
web: python -m uvicorn app:app --host 0.0.0.0 --port $PORT
```

重点是：

- `--host 0.0.0.0`：允许外部访问
- `--port $PORT`：使用 Railway 分配的端口

本地开发时仍然可以用：

```bash
python3 -m uvicorn app:app --reload
```

## 5. 常见问题

如果页面能打开，但聊天返回“没有读取到环境变量”，说明 Railway Variables 没有配置 API Key。

如果部署失败，先检查：

- `requirements.txt` 是否在项目根目录
- `Procfile` 是否在项目根目录
- Root Directory 是否设置正确
- Railway 日志里是否提示 Python 依赖安装失败

如果访问网址显示 502 或启动失败，通常是启动命令没有使用 Railway 的 `$PORT`。

如果只想展示页面，不调用大模型，也可以不配置 API Key，但聊天接口会返回错误提示。
