package com.example.llmlab.training.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * REST API response for training job operations
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TrainingResponse {

    private boolean success;
    private String message;
    private Long processId;
    private String outputDir;
    private String logFile;

    public static TrainingResponse success(Long pid, String outputDir) {
        return TrainingResponse.builder()
            .success(true)
            .message("Training started successfully")
            .processId(pid)
            .outputDir(outputDir)
            .logFile(outputDir + "/training.log")
            .build();
    }

    public static TrainingResponse error(String errorMessage) {
        return TrainingResponse.builder()
            .success(false)
            .message("Failed to start training: " + errorMessage)
            .build();
    }
}
