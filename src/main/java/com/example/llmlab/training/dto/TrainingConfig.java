package com.example.llmlab.training.dto;

import lombok.Builder;
import lombok.Data;

/**
 * Configuration for LoRA training job
 */
@Data
@Builder
public class TrainingConfig {

    // Required parameters
    private String modelPath;
    private String datasetName;  // For HuggingFace datasets
    private String outputDir;

    // Custom dataset files (alternative to datasetName)
    private String trainFile;    // Path to training JSONL file
    private String valFile;      // Path to validation JSONL file

    // Model configuration
    private Integer maxSeqLength;
    private Boolean loadIn4bit;

    // LoRA configuration
    private Integer loraRank;
    private Integer loraAlpha;
    private Double loraDropout;
    private String targetModules;

    // Training configuration
    private Integer batchSize;
    private Integer gradientAccumulationSteps;
    private Double learningRate;
    private Integer numEpochs;
    private Integer maxSteps;
    private Integer warmupSteps;
    private String lrSchedulerType;
    private Double weightDecay;

    // Dataset configuration
    private String datasetSplit;
    private String datasetField;
    private Integer numProc;

    // Logging and saving
    private Integer loggingSteps;
    private Integer saveSteps;
    private Integer saveTotalLimit;

    // GPU configuration
    private String gpuId;

    // Advanced options
    private Boolean bf16;
    private Boolean fp16;
    private String optimizer;

    /**
     * Get default configuration values
     */
    public static TrainingConfig getDefaults() {
        return TrainingConfig.builder()
            .maxSeqLength(2048)
            .loadIn4bit(true)
            .loraRank(64)
            .loraAlpha(16)
            .loraDropout(0.0)
            .targetModules("q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
            .batchSize(2)
            .gradientAccumulationSteps(8)
            .learningRate(2e-4)
            .numEpochs(1)
            .maxSteps(-1)
            .warmupSteps(100)
            .lrSchedulerType("linear")
            .weightDecay(0.01)
            .datasetSplit("train_sft[:10%]")
            .datasetField("messages")
            .numProc(4)
            .loggingSteps(10)
            .saveSteps(500)
            .saveTotalLimit(3)
            .gpuId("0")
            .bf16(true)
            .fp16(false)
            .optimizer("adamw_8bit")
            .build();
    }
}
