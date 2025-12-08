#!/bin/bash
# Test Phi Mini model on port 9020

PORT=${1:-9020}
MODEL="phi3-mini"

echo "🧪 Testing Phi Mini model on port $PORT"
echo ""

# Check if port is available
if ! curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
    echo "❌ Port $PORT is not responding"
    exit 1
fi

echo "✅ Server is responding"
echo ""

# List available models
echo "📋 Available models:"
curl -s http://localhost:$PORT/v1/models | python3 -m json.tool | grep -A 2 '"id"'
echo ""

# Test 1: Simple math
echo "🧮 Test 1: Simple Math"
echo "Question: What is 2+2? Answer briefly."
curl -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer briefly.\"}],
    \"max_tokens\": 50,
    \"temperature\": 0.7
  }" 2>/dev/null | python3 -m json.tool | grep -A 10 '"choices"'
echo ""

# Test 2: Creative writing
echo "📝 Test 2: Creative Writing"
echo "Question: Write a haiku about artificial intelligence."
curl -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Write a haiku about artificial intelligence.\"}],
    \"max_tokens\": 100,
    \"temperature\": 0.8
  }" 2>/dev/null | python3 -m json.tool | grep -A 10 '"choices"'
echo ""

# Test 3: Reasoning
echo "🧠 Test 3: Reasoning"
echo "Question: Explain why the sky is blue in one sentence."
curl -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Explain why the sky is blue in one sentence.\"}],
    \"max_tokens\": 100,
    \"temperature\": 0.5
  }" 2>/dev/null | python3 -m json.tool | grep -A 10 '"choices"'
echo ""

echo "✅ Testing complete!"
