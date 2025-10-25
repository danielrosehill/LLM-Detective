# LLM Detective

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**Note: This is an experimental project exploring AI agent capabilities for LLM evaluation.**

An undercover AI agent that goes incognito to test and evaluate other large language models. LLM Detective conducts systematic investigations to assess capabilities, biases, guardrails, and behavioral patterns of target LLMs.

## Features

- **Pre-Investigation Research**: Automatically fetches model cards, capabilities, and web search results before testing
- **Stealth Mode**: Simulates human interaction patterns with realistic typing delays and casual language
- **Multi-Provider Support**: Works with Ollama (local), OpenAI, Anthropic, and any OpenAI-compatible API
- **Comprehensive Testing**: 9 different test categories to evaluate LLM capabilities
- **Automated Analysis**: Rates and analyzes responses across multiple dimensions
- **Detailed Reports**: Generates JSON reports with full investigation results including model context

## Test Categories

| Test Category | Description |
|--------------|-------------|
| Knowledge Cutoff | Tests temporal knowledge boundaries |
| Vision Capability | Evaluates multimodal image understanding |
| Audio Capability | Tests audio processing beyond simple transcription |
| Bias Detection | Probes for political or ideological biases |
| Censorship Test | Identifies content filtering patterns |
| Guardrail Test | Triggers safety mechanisms |
| Conspiracy Theory | Tests critical thinking and fact-checking |
| Positive Reinforcement | Detects excessive or inappropriate enthusiasm |
| Agentic Capability | Evaluates tool use and execution abilities |

## Table of Contents

- [Features](#features)
- [Test Categories](#test-categories)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Command Line Options](#command-line-options)
- [Configuration File](#configuration-file)
- [Report Format](#report-format)
- [Understanding the Rating System](#understanding-the-rating-system)
- [Use Cases](#use-cases)
- [Architecture](#architecture)
- [Development](#development)
- [Ethical Considerations](#ethical-considerations)

## Installation

```bash
# Clone the repository
git clone https://github.com/danielrosehill/LLM-Detective.git
cd LLM-Detective

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using Ollama (Local)

```bash
# Make sure Ollama is running
ollama serve

# Run investigation with default settings
python llm_detective.py --provider ollama --model qwen2.5:14b-instruct-q5_K_M
```

### Using OpenAI API

```bash
python llm_detective.py \
  --provider openai \
  --model gpt-4 \
  --api-key your-api-key-here
```

### Using Anthropic Claude

```bash
python llm_detective.py \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --api-key your-api-key-here
```

### Using OpenRouter or Other APIs

```bash
python llm_detective.py \
  --provider openai \
  --model meta-llama/llama-3.1-70b-instruct \
  --base-url https://openrouter.ai/api/v1 \
  --api-key your-openrouter-key
```

## Advanced Usage

### Select Specific Tests

Run only certain investigation tasks:

```bash
python llm_detective.py \
  --provider ollama \
  --tasks knowledge_cutoff bias_detection guardrail_test
```

### Disable Human Simulation

For faster testing without realistic delays:

```bash
python llm_detective.py \
  --provider ollama \
  --no-human-sim
```

### Skip Model Research Phase

To skip the pre-investigation research (model card fetching and web search):

```bash
python llm_detective.py \
  --provider ollama \
  --no-research
```

### Custom Output Location

```bash
python llm_detective.py \
  --provider ollama \
  --output reports/investigation_$(date +%Y%m%d_%H%M%S).json
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | LLM provider: ollama, openai, anthropic | ollama |
| `--model` | Model name (provider-specific) | qwen2.5:14b-instruct-q5_K_M |
| `--api-key` | API key for cloud providers | None |
| `--base-url` | Base URL for API endpoints | Provider-specific |
| `--no-human-sim` | Disable human behavior simulation | False |
| `--no-research` | Skip pre-investigation model research phase | False |
| `--output` | Output file path | investigation_report.json |
| `--tasks` | Specific tasks to run (space-separated) | All tasks |

## Configuration File

Copy `config.example.json` to `config.json` and customize:

```json
{
  "provider": "ollama",
  "model": "qwen2.5:14b-instruct-q5_K_M",
  "simulate_human": true,
  "tasks": [
    "knowledge_cutoff",
    "bias_detection",
    "guardrail_test"
  ]
}
```

## Report Format

Investigation reports are saved as JSON with the following structure:

```json
{
  "model_name": "ollama/qwen2.5:14b",
  "provider": "OllamaProvider",
  "overall_rating": 7.2,
  "summary": "Investigation completed...",
  "timestamp": "2025-10-23T10:30:00",
  "model_context": {
    "model_name": "ollama/qwen2.5:14b",
    "claimed_capabilities": ["text_generation"],
    "knowledge_cutoff": null,
    "model_card_url": "https://huggingface.co/Qwen/Qwen2.5-14B",
    "model_card_content": "...",
    "web_search_results": [
      {
        "query": "qwen2.5 AI model capabilities",
        "source": "duckduckgo",
        "status": "success"
      }
    ],
    "metadata": {}
  },
  "tasks": [
    {
      "task_type": "knowledge_cutoff",
      "prompt": "What major tech...",
      "response": "I don't have...",
      "analysis": "Response analyzed...",
      "rating": 7.5,
      "metadata": {
        "response_time": 2.3,
        "response_length": 150
      }
    }
  ]
}
```

## Understanding the Rating System

Responses are rated on a 0-10 scale:

- **8-10**: Excellent - Model performs ideally for this test
- **6-7**: Good - Model shows appropriate behavior
- **4-5**: Average - Mixed or neutral results
- **2-3**: Poor - Concerning behavior or capabilities
- **0-1**: Critical - Major issues detected

## Use Cases

- **Model Selection**: Evaluate LLMs before choosing for production
- **Safety Testing**: Assess guardrails and content filtering
- **Capability Assessment**: Verify claimed multimodal capabilities
- **Bias Detection**: Identify political or ideological leanings
- **Benchmarking**: Compare models across standardized tests
- **Research**: Study LLM behavior patterns and limitations

## Architecture

| Component | Purpose |
|-----------|---------|
| `ModelResearcher` | Fetches model cards, capabilities, and web search results before investigation |
| `LLMProvider` | Abstract interface for different LLM backends |
| `HumanSimulator` | Adds realistic delays and interaction patterns |
| `DetectiveTasks` | Library of investigation prompts |
| `LLMDetective` | Main agent that coordinates testing |
| `TaskResult` | Structured response analysis and rating |
| `ModelContext` | Stores researched information about the target model |

### Supported Providers

| Provider | Class | Authentication |
|----------|-------|----------------|
| Ollama (Local) | `OllamaProvider` | None required |
| OpenAI | `OpenAIProvider` | API key |
| Anthropic | `AnthropicProvider` | API key |
| OpenRouter | `OpenAIProvider` | API key + base URL |

## Development

### Adding New Providers

Extend the `LLMProvider` abstract class:

```python
class CustomProvider(LLMProvider):
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        # Your implementation
        pass

    def get_model_name(self) -> str:
        return "custom/model-name"
```

### Adding New Test Tasks

Add to `DetectiveTasks.get_tasks()`:

```python
TaskType.YOUR_TEST: [
    "Test prompt 1",
    "Test prompt 2",
]
```

## Ethical Considerations

This tool is designed for:
- Legitimate security testing
- Model evaluation and selection
- Research purposes
- Educational demonstrations

**Not for:**
- Circumventing safety measures for malicious purposes
- Harassment or abuse of AI services
- Violating terms of service

Always ensure you have authorization to test target systems.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR.

## Roadmap

- [ ] Add more sophisticated analysis using meta-LLM evaluation
- [ ] Support for image/audio file uploads in multimodal tests
- [ ] Interactive CLI mode with conversation continuity
- [ ] Web dashboard for report visualization
- [ ] Plugin system for custom test modules
- [ ] Batch testing across multiple models
- [ ] Statistical comparison reports

## Credits

Created by [Daniel Rosehill](https://danielrosehill.com)

[![GitHub](https://img.shields.io/badge/GitHub-danielrosehill%2FLLM--Detective-181717?logo=github)](https://github.com/danielrosehill/LLM-Detective)