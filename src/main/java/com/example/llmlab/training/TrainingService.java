package com.example.llmlab.training;

import com.example.llmlab.training.dto.TrainingConfig;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Service for managing LLM training jobs
 */
@Slf4j
@Service
public class TrainingService {

    @Value("${training.python.venv:${user.home}/mnt/models/unsloth/.venv/bin/python}")
    private String pythonExecutable;

    @Value("${training.python.script:${user.home}/mnt/models/llm-lab-service/scripts/train_lora.py}")
    private String trainingScript;

    @Value("${training.python.custom.script:${user.home}/mnt/models/llm-lab-service/scripts/train_custom_dataset.py}")
    private String customDatasetScript;

    /**
     * Start a new training job
     *
     * @param config Training configuration
     * @return Process ID of the started training job
     * @throws IOException if the process cannot be started
     */
    public Long startTraining(TrainingConfig config) throws IOException {
        log.info("Starting training job:");
        log.info("  Model: {}", config.getModelPath());
        log.info("  Dataset: {}", config.getDatasetName());
        log.info("  Output: {}", config.getOutputDir());

        // Build command
        List<String> command = buildCommand(config);

        // Create output directory
        File outputDir = new File(config.getOutputDir());
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        // Setup process builder
        ProcessBuilder processBuilder = new ProcessBuilder(command);

        // Redirect output to log file
        File logFile = new File(config.getOutputDir(), "training.log");
        processBuilder.redirectOutput(ProcessBuilder.Redirect.appendTo(logFile));
        processBuilder.redirectError(ProcessBuilder.Redirect.appendTo(logFile));

        // Start process
        Process process = processBuilder.start();
        long pid = process.pid();

        log.info("Training process started with PID: {}", pid);
        log.info("Logs: {}", logFile.getAbsolutePath());

        // Monitor process asynchronously
        CompletableFuture.runAsync(() -> monitorProcess(process, pid, config.getOutputDir()));

        return pid;
    }

    /**
     * Build the command to execute the Python training script
     */
    private List<String> buildCommand(TrainingConfig config) {
        List<String> command = new ArrayList<>();

        // Determine if using custom dataset files or HuggingFace dataset
        boolean useCustomDataset = config.getTrainFile() != null && !config.getTrainFile().isEmpty();

        // Python executable
        command.add(pythonExecutable);

        // Training script (custom or standard)
        command.add(useCustomDataset ? customDatasetScript : trainingScript);

        // Required parameters
        command.add("--model_path");
        command.add(config.getModelPath());

        // Dataset parameters (different for custom vs HuggingFace)
        if (useCustomDataset) {
            command.add("--train_file");
            command.add(config.getTrainFile());

            if (config.getValFile() != null && !config.getValFile().isEmpty()) {
                command.add("--val_file");
                command.add(config.getValFile());
            }
        } else {
            command.add("--dataset_name");
            command.add(config.getDatasetName());
        }

        command.add("--output_dir");
        command.add(config.getOutputDir());

        // Common optional parameters (supported by both scripts)
        addOptionalParam(command, "--max_seq_length", config.getMaxSeqLength());
        addOptionalParam(command, "--batch_size", config.getBatchSize());
        addOptionalParam(command, "--gradient_accumulation_steps", config.getGradientAccumulationSteps());
        addOptionalParam(command, "--learning_rate", config.getLearningRate());
        addOptionalParam(command, "--num_epochs", config.getNumEpochs());
        addOptionalParam(command, "--lora_r", config.getLoraRank());
        addOptionalParam(command, "--lora_alpha", config.getLoraAlpha());
        addOptionalParam(command, "--lora_dropout", config.getLoraDropout());

        if (config.getGpuId() != null) {
            command.add("--gpu_id");
            command.add(config.getGpuId());
        }

        // HuggingFace dataset script only parameters
        if (!useCustomDataset) {
            addOptionalParam(command, "--max_steps", config.getMaxSteps());
            addOptionalParam(command, "--warmup_steps", config.getWarmupSteps());
            addOptionalParam(command, "--weight_decay", config.getWeightDecay());
            addOptionalParam(command, "--logging_steps", config.getLoggingSteps());
            addOptionalParam(command, "--save_steps", config.getSaveSteps());
            addOptionalParam(command, "--save_total_limit", config.getSaveTotalLimit());
            addOptionalParam(command, "--num_proc", config.getNumProc());

            if (config.getTargetModules() != null) {
                command.add("--target_modules");
                command.add(config.getTargetModules());
            }

            if (config.getDatasetSplit() != null) {
                command.add("--dataset_split");
                command.add(config.getDatasetSplit());
            }

            if (config.getDatasetField() != null) {
                command.add("--dataset_field");
                command.add(config.getDatasetField());
            }

            if (config.getLrSchedulerType() != null) {
                command.add("--lr_scheduler_type");
                command.add(config.getLrSchedulerType());
            }

            if (config.getOptimizer() != null) {
                command.add("--optim");
                command.add(config.getOptimizer());
            }
        }

        log.debug("Training command: {}", String.join(" ", command));

        return command;
    }

    /**
     * Add optional parameter to command if value is not null
     */
    private void addOptionalParam(List<String> command, String flag, Object value) {
        if (value != null) {
            command.add(flag);
            command.add(String.valueOf(value));
        }
    }

    /**
     * Monitor training process and log when it completes
     */
    private void monitorProcess(Process process, long pid, String outputDir) {
        try {
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                log.info("Training process {} completed successfully", pid);
                log.info("Model saved to: {}", outputDir);
            } else {
                log.error("Training process {} failed with exit code: {}", pid, exitCode);
                log.error("Check logs at: {}/training.log", outputDir);
            }
        } catch (InterruptedException e) {
            log.error("Training process monitoring interrupted", e);
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Check if a process is still running
     */
    public boolean isProcessRunning(long pid) {
        return ProcessHandle.of(pid)
            .map(ProcessHandle::isAlive)
            .orElse(false);
    }

    /**
     * Kill a running training process
     */
    public boolean killProcess(long pid) {
        return ProcessHandle.of(pid)
            .map(processHandle -> {
                processHandle.destroy();
                log.info("Sent termination signal to process: {}", pid);
                return true;
            })
            .orElse(false);
    }
}
