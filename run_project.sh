#!/bin/bash

set -euo pipefail

# RAGFlow & MinIO Start Script (Multimodal RAG Project)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAGFLOW_ROOT="${RAGFLOW_ROOT:-$(cd "$PROJECT_DIR/../.." && pwd)}"

# 1. Enter the RAGFlow root directory
echo "Starting RAGFlow main services in ${RAGFLOW_ROOT}..."
cd "$RAGFLOW_ROOT"

# 2. Start services in background
# This will fix the "Network Exception" in RAGFlow UI
docker-compose up -d

# 3. Wait for services to be ready
echo "Waiting for RAGFlow backend (8080) to be ready..."
until curl -s http://localhost:8080 > /dev/null; do
  sleep 5
  echo -n "."
done
echo "Backend is UP!"

# 4. Enter the Multimodal Project directory
echo "Setting up local environment..."
cd "$PROJECT_DIR"

# 5. Create and activate virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 6. Install dependencies
echo "Installing Python dependencies (this may take a minute)..."
pip install --upgrade pip
pip install -r requirements.txt

# 7. Run the processing script
echo "Running the Multimodal RAG Processor..."
cd src
python3 ragflow_pdf_processor.py ../data/pdf/excavator_repair_case_en.pdf
cd ..

echo "Done!"
echo ""
echo "Next steps:"
echo "  Start chat server:  cd src && uvicorn chat_server:app --port 7860 --reload"
echo "  Start MCP server:   cd src && python mcp_server.py"
echo ""
echo "  RAGFlow UI:         http://localhost:8080"
echo "  Chat UI:            http://localhost:7860"
echo ""
echo "Claude Desktop config (~/.config/claude/claude_desktop_config.json):"
echo '  {'
echo '    "mcpServers": {'
echo '      "excavator-maintenance": {'
echo '        "command": "'"$(pwd)"'/venv/bin/python",'
echo '        "args": ["'"$(pwd)"'/src/mcp_server.py"]'
echo '      }'
echo '    }'
echo '  }'
