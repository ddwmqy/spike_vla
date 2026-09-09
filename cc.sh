#!/usr/bin/env bash
# 在新容器里启动 Claude Code 的统一入口:自动挂载 /data 上的持久配置目录,
# 这样聊天记录、memory、设置都写到 /data,容器重建后不丢。
# 用法: ./cc.sh            (新会话)
#       ./cc.sh --resume    (恢复历史对话)
#       ./cc.sh -c          (继续最近一次对话)
source /data/260010028/dwh_vla/claude_env.sh
export PATH="/data/260010028/dwh_vla/node_modules/.bin:$PATH"
exec claude "$@"
