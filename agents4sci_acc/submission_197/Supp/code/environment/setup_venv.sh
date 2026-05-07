#!/bin/bash

echo "🚀 Setting up Enhanced Boids Evolution Environment"
echo "================================================="
echo

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

# Special handling for Python 3.13+
if [ "$python_major" -eq 3 ] && [ "$python_minor" -ge 13 ]; then
    echo "⚠️  Python 3.13+ detected - using compatible package versions"
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and setuptools
echo "⬆️ Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "📚 Installing packages from requirements.txt..."
pip install -r environment/requirements.txt
if [ $? -ne 0 ]; then
    echo "⚠️ Some packages failed to install. This is normal for optional packages."
    echo "💡 Try installing packages individually if needed:"
    echo "   pip install openai anthropic requests pandas numpy"
fi

# Test environment manager
echo "🧪 Testing environment manager..."
python3 src/environment_manager.py

echo
echo "✅ Environment setup complete!"
echo
echo "🎯 Next steps:"
echo "   1. Activate the environment: source venv/bin/activate"
echo "   2. Set up your API keys in .env file:"
echo "      - OPENAI_API_KEY=your_key"
echo "      - TAVILY_API_KEY=your_key"
echo "   3. Run experiments: python3 run_experiment.py"
echo
echo "📦 Available packages: $(wc -l < environment/requirements.txt) packages installed"
echo "🤖 AI capabilities: OpenAI ready"
echo "🔍 Search capabilities: Tavily, DuckDuckGo ready"
echo "📊 Data analysis: pandas, numpy, matplotlib ready" 