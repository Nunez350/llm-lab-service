#!/bin/bash
# Check if a port is in use and what process is using it

PORT=${1:-9020}

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [port_number]"
    echo "Example: $0 9020"
    exit 1
fi

echo "🔍 Checking port $PORT..."
echo ""

# Check if port is listening
echo "📊 Listening status:"
if command -v ss >/dev/null 2>&1; then
    if ss -tuln | grep -q ":$PORT "; then
        echo "✅ Port $PORT is LISTENING"
        ss -tuln | grep ":$PORT "
    else
        echo "❌ Port $PORT is NOT listening"
    fi
elif command -v netstat >/dev/null 2>&1; then
    if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
        echo "✅ Port $PORT is LISTENING"
        netstat -tuln | grep ":$PORT "
    else
        echo "❌ Port $PORT is NOT listening"
    fi
else
    echo "⚠️ Neither ss nor netstat found"
fi

echo ""

# Check which process is using it
echo "🔎 Process information:"
if command -v lsof >/dev/null 2>&1; then
    if sudo lsof -i :$PORT 2>/dev/null; then
        :
    else
        echo "   (Requires sudo for process details)"
        echo "   Run: sudo lsof -i :$PORT"
    fi
else
    echo "   lsof not found, install with: sudo apt-get install lsof"
fi

echo ""

# Test if port responds to connections
echo "🧪 Connection test:"
if command -v nc >/dev/null 2>&1; then
    if nc -zv localhost $PORT 2>&1 | grep -q "succeeded"; then
        echo "✅ Port $PORT is accepting connections"
        nc -zv localhost $PORT 2>&1
    else
        echo "❌ Port $PORT is not accepting connections"
    fi
elif command -v curl >/dev/null 2>&1; then
    if curl -s --connect-timeout 2 http://localhost:$PORT >/dev/null 2>&1; then
        echo "✅ Port $PORT is responding to HTTP requests"
        echo "   Server: $(curl -s -I http://localhost:$PORT 2>/dev/null | grep -i '^server:' | cut -d: -f2 | tr -d '\r')"
    else
        echo "❌ Port $PORT is not responding to HTTP"
    fi
else
    echo "   nc or curl not available for connection test"
fi

echo ""

# Summary
if ss -tuln 2>/dev/null | grep -q ":$PORT " || netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
    echo "📝 Summary: Port $PORT appears to be in use"
    exit 0
else
    echo "📝 Summary: Port $PORT appears to be available"
    exit 1
fi
