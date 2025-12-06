# LLM Lab Service - Dataset Preparation & Training Infrastructure

Complete pipeline for preparing custom datasets and fine-tuning large language models using LoRA/QLoRA via a REST API.

## Overview

This service provides a Java-based REST API for managing LLM training jobs, with support for:
- Custom dataset preparation and merging
- HuggingFace dataset integration
- LoRA/QLoRA fine-tuning with Unsloth
- GPU resource management
- Process monitoring and logging

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Spring Boot REST API                      │
│                      (Port 8083)                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─> TrainingController (REST endpoints)
                     │
                     ├─> TrainingService (Process management)
                     │
                     └─> Python Training Scripts
                         ├─> train_lora.py (HuggingFace datasets)
                         └─> train_custom_dataset.py (Custom JSONL)
```

## Features

### Dataset Preparation Pipeline

The `scripts/prepare_dataset.py` provides a complete pipeline for merging and preparing datasets:

- **Multi-source dataset merging**: Combine datasets from HuggingFace
- **Quality filtering**:
  - Character count limits (50-6000 chars)
  - Minimum conversation turns (2+)
  - Role validation (user/assistant/system)
  - English language detection
- **Format normalization**: Convert to standard messages format
- **Deduplication**: SHA-256 content hashing
- **Train/validation split**: Configurable ratio (default 90/10)
- **JSONL output**: Compatible with training scripts

#### Dataset Statistics

Recent merge: OpenHermes 2.5 + SlimOrca
- **Training samples**: 372,354
- **Validation samples**: 41,372
- **Average length**: 1,492 characters per conversation
- **Average turns**: 2.5 turns per conversation
- **Duplicates removed**: 0
- **Quality**: All samples pass filtering criteria

### Training Infrastructure

Two training modes supported:

#### 1. HuggingFace Datasets (`train_lora.py`)

```bash
python scripts/train_lora.py \
  --model_path /path/to/model \
  --dataset_name HuggingFaceH4/ultrachat_200k \
  --output_dir ./output \
  --dataset_split "train_sft[:10%]" \
  --num_epochs 1
```

#### 2. Custom JSONL Datasets (`train_custom_dataset.py`)

```bash
python scripts/train_custom_dataset.py \
  --model_path /path/to/model \
  --train_file ./merged_train.jsonl \
  --val_file ./merged_val.jsonl \
  --output_dir ./output \
  --num_epochs 3
```

### REST API Integration

Start training via HTTP API:

```bash
curl -X POST http://localhost:8083/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "modelPath": "/home/rnu/mnt/models/phi_models/phi3-medium/",
    "trainFile": "/mnt/models/llm-lab-service/scripts/datasets/merged_train.jsonl",
    "valFile": "/mnt/models/llm-lab-service/scripts/datasets/merged_val.jsonl",
    "outputDir": "/mnt/datasets/lora_dataset/phi3-openhermes-slimorca",
    "maxSeqLength": 4096,
    "batchSize": 2,
    "numEpochs": 3,
    "loraRank": 64,
    "gpuId": "0"
  }'
```

Response:
```json
{
  "success": true,
  "processId": 935546,
  "message": "Training started successfully",
  "outputDir": "/mnt/datasets/lora_dataset/phi3-openhermes-slimorca"
}
```

## API Endpoints

### Start Training Job

**POST** `/api/training/start`

Required parameters:
- `modelPath`: Path to base model
- `datasetName` OR `trainFile`: HuggingFace dataset or custom JSONL file
- `outputDir`: Where to save the fine-tuned model

Optional parameters (with defaults):
- `maxSeqLength`: 2048 (can override with any value, e.g., 4096)
- `batchSize`: 2
- `gradientAccumulationSteps`: 8
- `learningRate`: 0.0002
- `numEpochs`: 1
- `loraRank`: 64
- `loraAlpha`: 16
- `loraDropout`: 0.0
- `gpuId`: "0"

HuggingFace dataset-only parameters:
- `datasetSplit`: "train_sft[:10%]"
- `datasetField`: "messages"
- `maxSteps`: -1
- `warmupSteps`: 100
- `lrSchedulerType`: "linear"
- `weightDecay`: 0.01
- `loggingSteps`: 10
- `saveSteps`: 500
- `saveTotalLimit`: 3
- `numProc`: 4
- `targetModules`: "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
- `optimizer`: "adamw_8bit"

### Check Training Status

**GET** `/api/training/status/{pid}`

```bash
curl http://localhost:8083/api/training/status/935546
```

Response:
```json
{
  "processId": 935546,
  "isRunning": true
}
```

### Stop Training Job

**DELETE** `/api/training/{pid}`

```bash
curl -X DELETE http://localhost:8083/api/training/935546
```

## Usage Workflow

### 1. Prepare Custom Dataset

```bash
cd scripts
python prepare_dataset.py
```

This creates:
- `scripts/datasets/merged_train.jsonl` (372,354 samples)
- `scripts/datasets/merged_val.jsonl` (41,372 samples)
- `scripts/datasets/merged_stats.json` (statistics)

### 2. Start Spring Boot Service

```bash
./mvnw clean package -DskipTests
nohup java -jar target/llm-lab-service-1.0.0.jar --server.port=8083 > spring-boot.log 2>&1 &
```

### 3. Start Training via API

```bash
curl -X POST http://localhost:8083/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "modelPath": "/home/rnu/mnt/models/phi_models/phi3-medium/",
    "trainFile": "/mnt/models/llm-lab-service/scripts/datasets/merged_train.jsonl",
    "valFile": "/mnt/models/llm-lab-service/scripts/datasets/merged_val.jsonl",
    "outputDir": "/mnt/datasets/lora_dataset/phi3-openhermes-slimorca",
    "maxSeqLength": 4096,
    "numEpochs": 3,
    "loraRank": 64,
    "gpuId": "0"
  }'
```

### 4. Monitor Training

Check logs:
```bash
tail -f /mnt/datasets/lora_dataset/phi3-openhermes-slimorca/training.log
```

Check status via API:
```bash
curl http://localhost:8083/api/training/status/{PID}
```

Check Spring Boot logs:
```bash
tail -f spring-boot.log
```

### 5. Stop Training (if needed)

```bash
curl -X DELETE http://localhost:8083/api/training/{PID}
```

## Configuration

### Application Properties

`src/main/resources/application.properties`:

```properties
server.port=8083

# Python environment
training.python.venv=${user.home}/mnt/models/unsloth/.venv/bin/python

# Training scripts
training.python.script=${user.home}/mnt/models/llm-lab-service/scripts/train_lora.py
training.python.custom.script=${user.home}/mnt/models/llm-lab-service/scripts/train_custom_dataset.py

# Database (PostgreSQL)
spring.datasource.url=jdbc:postgresql://localhost:5432/llmlab
spring.datasource.username=llmlab
spring.datasource.password=llmlab123
spring.jpa.hibernate.ddl-auto=update
spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.PostgreSQL10Dialect
```

### Dataset Preparation Configuration

Edit `scripts/prepare_dataset.py`:

```python
# Target sample counts
TARGET_OH = 300_000  # OpenHermes samples
TARGET_SO = 150_000  # SlimOrca samples

# Quality filters
MAX_CHARS_PER_CONVO = 6000
MIN_CHARS_PER_CONVO = 50
MIN_TURNS = 2

# Train/validation split
VAL_RATIO = 0.1
```

### Training Defaults

Edit `src/main/java/com/example/llmlab/training/dto/TrainingConfig.java`:

```java
public static TrainingConfig getDefaults() {
    return TrainingConfig.builder()
        .maxSeqLength(2048)        // Can override via API
        .loadIn4bit(true)
        .loraRank(64)
        .loraAlpha(16)
        .loraDropout(0.0)
        .batchSize(2)
        .gradientAccumulationSteps(8)
        .learningRate(2e-4)
        .numEpochs(1)
        .gpuId("0")
        .bf16(true)
        .optimizer("adamw_8bit")
        .build();
}
```

## GPU Memory Management

### Memory Requirements

For Phi-3 Medium (14B parameters) with 4-bit quantization:
- **Model**: ~8-10GB
- **Training overhead**: ~10-15GB per GPU
- **Total**: ~20-25GB minimum

### Optimization Strategies

1. **Reduce sequence length**:
   ```json
   "maxSeqLength": 1024
   ```

2. **Reduce batch size**:
   ```json
   "batchSize": 1
   ```

3. **Increase gradient accumulation**:
   ```json
   "gradientAccumulationSteps": 16
   ```

4. **Free GPU memory**:
   ```bash
   docker stop vllm-phi3-medium
   docker stop vllm-other-models
   ```

5. **Select specific GPU**:
   ```json
   "gpuId": "1"
   ```

## File Structure

```
llm-lab-service/
├── src/main/java/com/example/llmlab/
│   └── training/
│       ├── TrainingController.java       # REST API endpoints
│       ├── TrainingService.java          # Process management & command building
│       └── dto/
│           ├── TrainingConfig.java       # Configuration with defaults
│           ├── TrainingRequest.java      # API request DTO
│           └── TrainingResponse.java     # API response DTO
├── scripts/
│   ├── prepare_dataset.py                # Dataset preparation pipeline
│   ├── train_lora.py                     # HuggingFace dataset training
│   ├── train_custom_dataset.py           # Custom JSONL training
│   └── datasets/
│       ├── merged_train.jsonl            # Training data (372K samples)
│       ├── merged_val.jsonl              # Validation data (41K samples)
│       └── merged_stats.json             # Dataset statistics
├── target/
│   └── llm-lab-service-1.0.0.jar        # Compiled application
└── spring-boot.log                       # Application logs
```

## Technical Details

### Dataset Format

JSONL format with messages structure:

```json
{"messages": [
  {"role": "user", "content": "What is machine learning?"},
  {"role": "assistant", "content": "Machine learning is..."}
]}
```

### Training Technologies

- **Unsloth**: Fast training library
- **LoRA**: Low-Rank Adaptation for parameter-efficient fine-tuning
- **QLoRA**: 4-bit quantized LoRA
- **BitsAndBytes**: 4-bit quantization library
- **HuggingFace Transformers**: Model loading and training
- **PEFT**: Parameter-Efficient Fine-Tuning library

### Java Components

- **Spring Boot 3.1.5**: Web framework
- **Lombok**: Boilerplate reduction
- **PostgreSQL**: Database (optional, for future job tracking)
- **ProcessBuilder**: Python process management
- **CompletableFuture**: Async process monitoring

## Troubleshooting

### Training fails with "unrecognized arguments"

The service automatically routes to the correct training script based on dataset type:
- Custom JSONL files → `train_custom_dataset.py`
- HuggingFace datasets → `train_lora.py`

Each script accepts different parameters. The service filters parameters accordingly.

### GPU out of memory

1. Check GPU usage: `nvidia-smi`
2. Stop other processes using GPU
3. Reduce `maxSeqLength` or `batchSize` in API request
4. Use smaller model or switch GPU

### Process not starting

1. Check Spring Boot logs: `tail -f spring-boot.log`
2. Check training logs: `tail -f {outputDir}/training.log`
3. Verify Python environment: `which python`
4. Test script manually: `python scripts/train_custom_dataset.py --help`

### Can't connect to API

1. Check if service is running: `ps aux | grep llm-lab-service`
2. Check port: `netstat -tulpn | grep 8083`
3. Check logs: `tail -f spring-boot.log`
4. Restart service: `kill {PID} && nohup java -jar target/llm-lab-service-1.0.0.jar --server.port=8083 > spring-boot.log 2>&1 &`

## Future Enhancements

- [ ] Multi-GPU data parallelism support
- [ ] Job queue and scheduling
- [ ] Training metrics API endpoint
- [ ] Model registry integration
- [ ] Checkpoint management
- [ ] Automated hyperparameter tuning
- [ ] Web UI dashboard
- [ ] Training progress streaming via WebSocket
- [ ] Model evaluation pipeline
- [ ] Distributed training support

## License

MIT

## Contributors

Built with Claude Code
