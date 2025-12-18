package com.example.bpmn;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.List;

public class CsvWriter {
    private final String csvContent;
    private String fileName;
    private String anonLevel;
    private String group;

    public CsvWriter(String csvContent, List<BpmnXml> models) {
        this.csvContent = csvContent;
        generateNames(models);
    }

    private void generateNames(List<BpmnXml> models) {
        if (models.isEmpty()) {
            throw new IllegalArgumentException("Model list is empty!");
        }

        BpmnXml sample = models.get(0);  // All models belong to same group & anonLevel

        // Extract group
        String groupPart = (sample.getGroup() == com.example.bpmn.Group.Meins) ? "Prototype" : sample.getGroup().name();

        // Extract AnonLevel
        String anonPart = switch (sample.getAnonLevel()) {
            case None -> "A0";
            case Anon1 -> "A1";
            case Anon2 -> "A2";
        };

        this.fileName = "OutputMetrics_" + groupPart + "_" + anonPart + ".csv";
        this.anonLevel = anonPart;
        this.group = groupPart;
    }

    public void writeToFile() {
        Path dirPath = Paths.get("src", "main", "resources", "Output", group, anonLevel);
        Path outputPath = dirPath.resolve(fileName);

        try {
            Files.createDirectories(dirPath);
            Files.writeString(outputPath, csvContent, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
            System.out.println("CSV written: " + outputPath);
        } catch (IOException e) {
            System.err.println("Failed to write CSV: " + fileName);
            e.printStackTrace();
        }
    }



}
