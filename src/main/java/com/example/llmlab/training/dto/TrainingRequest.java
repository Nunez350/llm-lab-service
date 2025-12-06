package com.example.llmlab.training.dto;

import lombok.Data;

/**
 * REST API request for starting a training job
 */
@Data
public class TrainingRequest {

    // Required fields
    private String modelPath;
    private String datasetName;  // For HuggingFace datasets
    private String outputDir;

    // Custom dataset files (alternative to datasetName)
    private String trainFile;    // Path to training JSONL file
    private String valFile;      // Path to validation JSONL file

    // Optional fields with defaults
    private Integer maxSeqLength;
    private Integer batchSize;
    private Integer gradientAccumulationSteps;
    private Double learningRate;
    private Integer numEpochs;
    private Integer loraRank;
    private String datasetSplit;
    private String gpuId;

    // Advanced optional fields
    private Integer maxSteps;
    private Integer warmupSteps;
    private String lrSchedulerType;
    private Double weightDecay;
    private Integer loraAlpha;
    private Double loraDropout;
    private String targetModules;
    private String datasetField;
    private Integer loggingSteps;
    private Integer saveSteps;
    private String optimizer;
}
