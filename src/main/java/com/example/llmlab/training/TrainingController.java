package com.example.llmlab.training;

import com.example.llmlab.training.dto.TrainingConfig;
import com.example.llmlab.training.dto.TrainingRequest;
import com.example.llmlab.training.dto.TrainingResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * REST API for managing LLM training jobs
 */
@Slf4j
@RestController
@RequestMapping("/api/training")
@RequiredArgsConstructor
public class TrainingController {

    private final TrainingService trainingService;

    /**
     * Start a new training job
     *
     * POST /api/training/start
     *
     * Example request body:
     * {
     *   "modelPath": "/home/rnu/mnt/models/phi_models/phi3-medium/",
     *   "datasetName": "HuggingFaceH4/ultrachat_200k",
     *   "outputDir": "./phi3-finetuned",
     *   "maxSeqLength": 2048,
     *   "batchSize": 2,
     *   "gradientAccumulationSteps": 8,
     *   "learningRate": 0.0002,
     *   "numEpochs": 1,
     *   "loraRank": 64,
     *   "datasetSplit": "train_sft[:10%]",
     *   "gpuId": "0"
     * }
     */
    @PostMapping("/start")
    public ResponseEntity<TrainingResponse> startTraining(@RequestBody TrainingRequest request) {
        try {
            // Validate required fields
            if (request.getModelPath() == null || request.getModelPath().isEmpty()) {
                return ResponseEntity.badRequest()
                    .body(TrainingResponse.error("modelPath is required"));
            }

            // Either datasetName (HuggingFace) OR trainFile (custom) must be provided
            boolean hasHuggingFaceDataset = request.getDatasetName() != null && !request.getDatasetName().isEmpty();
            boolean hasCustomDataset = request.getTrainFile() != null && !request.getTrainFile().isEmpty();

            if (!hasHuggingFaceDataset && !hasCustomDataset) {
                return ResponseEntity.badRequest()
                    .body(TrainingResponse.error("Either datasetName or trainFile is required"));
            }

            if (request.getOutputDir() == null || request.getOutputDir().isEmpty()) {
                return ResponseEntity.badRequest()
                    .body(TrainingResponse.error("outputDir is required"));
            }

            // Build configuration with defaults
            TrainingConfig config = buildConfig(request);

            // Start training
            Long pid = trainingService.startTraining(config);

            // Return success response
            return ResponseEntity.ok(TrainingResponse.success(pid, config.getOutputDir()));

        } catch (Exception e) {
            log.error("Failed to start training", e);
            return ResponseEntity.internalServerError()
                .body(TrainingResponse.error(e.getMessage()));
        }
    }

    /**
     * Check if a training process is still running
     *
     * GET /api/training/status/{pid}
     */
    @GetMapping("/status/{pid}")
    public ResponseEntity<TrainingStatusResponse> getStatus(@PathVariable Long pid) {
        boolean isRunning = trainingService.isProcessRunning(pid);
        return ResponseEntity.ok(new TrainingStatusResponse(pid, isRunning));
    }

    /**
     * Stop a running training process
     *
     * DELETE /api/training/{pid}
     */
    @DeleteMapping("/{pid}")
    public ResponseEntity<TrainingResponse> stopTraining(@PathVariable Long pid) {
        boolean killed = trainingService.killProcess(pid);
        if (killed) {
            return ResponseEntity.ok(TrainingResponse.builder()
                .success(true)
                .message("Training process " + pid + " terminated")
                .processId(pid)
                .build());
        } else {
            return ResponseEntity.notFound().build();
        }
    }

    /**
     * Build training configuration from request, applying defaults where needed
     */
    private TrainingConfig buildConfig(TrainingRequest request) {
        TrainingConfig defaults = TrainingConfig.getDefaults();

        return TrainingConfig.builder()
            // Required
            .modelPath(request.getModelPath())
            .datasetName(request.getDatasetName())
            .outputDir(request.getOutputDir())

            // Custom dataset files
            .trainFile(request.getTrainFile())
            .valFile(request.getValFile())

            // Optional with defaults
            .maxSeqLength(getOrDefault(request.getMaxSeqLength(), defaults.getMaxSeqLength()))
            .batchSize(getOrDefault(request.getBatchSize(), defaults.getBatchSize()))
            .gradientAccumulationSteps(getOrDefault(request.getGradientAccumulationSteps(), defaults.getGradientAccumulationSteps()))
            .learningRate(getOrDefault(request.getLearningRate(), defaults.getLearningRate()))
            .numEpochs(getOrDefault(request.getNumEpochs(), defaults.getNumEpochs()))
            .loraRank(getOrDefault(request.getLoraRank(), defaults.getLoraRank()))
            .datasetSplit(getOrDefault(request.getDatasetSplit(), defaults.getDatasetSplit()))
            .gpuId(getOrDefault(request.getGpuId(), defaults.getGpuId()))

            // Advanced optional
            .maxSteps(getOrDefault(request.getMaxSteps(), defaults.getMaxSteps()))
            .warmupSteps(getOrDefault(request.getWarmupSteps(), defaults.getWarmupSteps()))
            .lrSchedulerType(getOrDefault(request.getLrSchedulerType(), defaults.getLrSchedulerType()))
            .weightDecay(getOrDefault(request.getWeightDecay(), defaults.getWeightDecay()))
            .loraAlpha(getOrDefault(request.getLoraAlpha(), defaults.getLoraAlpha()))
            .loraDropout(getOrDefault(request.getLoraDropout(), defaults.getLoraDropout()))
            .targetModules(getOrDefault(request.getTargetModules(), defaults.getTargetModules()))
            .datasetField(getOrDefault(request.getDatasetField(), defaults.getDatasetField()))
            .loggingSteps(getOrDefault(request.getLoggingSteps(), defaults.getLoggingSteps()))
            .saveSteps(getOrDefault(request.getSaveSteps(), defaults.getSaveSteps()))
            .optimizer(getOrDefault(request.getOptimizer(), defaults.getOptimizer()))

            // Use defaults for these
            .loadIn4bit(defaults.getLoadIn4bit())
            .numProc(defaults.getNumProc())
            .saveTotalLimit(defaults.getSaveTotalLimit())
            .bf16(defaults.getBf16())
            .fp16(defaults.getFp16())

            .build();
    }

    private <T> T getOrDefault(T value, T defaultValue) {
        return value != null ? value : defaultValue;
    }

    /**
     * Response DTO for training status
     */
    record TrainingStatusResponse(Long processId, boolean isRunning) {}
}
