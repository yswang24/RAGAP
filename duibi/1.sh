#!/bin/bash
echo "=== 完整代理测试 ==="

echo "1. 检查端口状态:"
netstat -tulpn | grep 7980

echo "2. 测试基础代理连接:"
timeout 10 curl --socks5 127.0.0.1:7980 -I https://www.google.com/

echo "3. 测试 GitHub Copilot:"
timeout 10 curl --socks5 127.0.0.1:7980 -I https://api.githubcopilot.com/

echo "4. 详细错误信息:"
curl -v --socks5 127.0.0.1:7980 --max-time 10 https://www.google.com/ 2>&1 | tail -20