package com.example.bpmn;

import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import org.camunda.bpm.model.bpmn.Bpmn;
import org.camunda.bpm.model.bpmn.BpmnModelInstance;
import org.camunda.bpm.model.bpmn.instance.*;

/**
 * Stores all the relevant information of each BpmnXml
 */
public class BpmnXml {

    //region Variables
    private Path filePath;
    private Group group;
    private AnonLevel anonLevel;
    private String fullFilename;
    private String fileName;
    private Boolean hasErrors;
    private HashMap stats;
    private BpmnModelInstance modelInstance;
//endregion

    //region Constructor
    public BpmnXml(Path filePath, Group group, AnonLevel anonLevel) {
        this.filePath = filePath;
        this.hasErrors = false;
        this.group = group;
        this.anonLevel = anonLevel;
        this.stats = new HashMap<>();

        this.fullFilename = filePath.getFileName().toString();
        this.fileName = getCorrectName(fullFilename);

        try {
            this.modelInstance = Bpmn.readModelFromFile(filePath.toFile());
        } catch (Exception e) {
            // das ist ein bisschen ein Saugriff --> Es gibt bewusst Modelle mit Fehlern und das ist der einfachste Weg der mir eingefallen ist um sie zu handeln.
            this.modelInstance = null;
            this.hasErrors = true;
        }
        // Auswertung starten
        analyse();
    }
//endregion

    //region Getter & Setter
    public Path getFilePath() {
        return filePath;
    }

    public String getfullFilename() {
        return fullFilename;
    }

    public Group getGroup() {
        return group;
    }

    public AnonLevel getAnonLevel() {
        return anonLevel;
    }

    public String getFileName() {
        return fileName;
    }

    public BpmnModelInstance getModelInstance() {
        return modelInstance;
    }

    public boolean getHasErrors() {
        return  hasErrors;
    }

//endregion

    //region Methods

    /**
    Vergibt den richtigen Namen (wichtig zum Matchen). Macht auch property hasErrors
     */
    private String getCorrectName(String fullFilename) {
        String returnValue = fullFilename;
        switch (group) {
            case experts:
                returnValue = returnValue.split("-")[1].replace(".bpmn", "");
                break;
            case Aalst:
                if(anonLevel == AnonLevel.None){
                    returnValue = returnValue.replace(".bpmn", "");
                }else if (anonLevel == AnonLevel.Anon1){
                    returnValue = returnValue.replace("_Anonymisierung1.bpmn", "");
                }else{
                    returnValue = returnValue.replace("_Anonymisierung2.bpmn", "");
                }
                break;
            case Aau:
                if(anonLevel == AnonLevel.None){
                    returnValue = returnValue.replace(".xml", "");
                }else if (anonLevel == AnonLevel.Anon1){
                    returnValue = returnValue.replace("_Anonymisierung1.xml", "");
                }else{
                    returnValue = returnValue.replace("_Anonymisierung2.xml", "");
                }
                break;
            case Meins:
                if (returnValue.startsWith("ERROR_")){
                    hasErrors = true;
                    returnValue = returnValue.replace("ERROR_", "");
                }
                if(anonLevel == AnonLevel.None){
                    returnValue = returnValue.replace("Graphical_", "");
                    returnValue = returnValue.replace(".bpmn", "");
                }else if (anonLevel == AnonLevel.Anon1){
                    returnValue = returnValue.replace("Graphical_Ano1_", "");
                    returnValue = returnValue.replace(".bpmn", "");
                }else{
                    returnValue = returnValue.replace("Graphical_Ano2_", "");
                    returnValue = returnValue.replace(".bpmn", "");
                }
                break;
            default:
                returnValue = "ERROR";
        }

        return returnValue;
    }

    //region Analysing
    private void analyse(){
        // was wollen wir da noch analysieren
        if (modelInstance == null) return;
        analyseGateways();
    }

    private void analyseGateways() {
        stats.put("Tasks", modelInstance.getModelElementsByType(Task.class).size());
        stats.put("Events", modelInstance.getModelElementsByType(Event.class).size());
        stats.put("Sequence Flows", modelInstance.getModelElementsByType(SequenceFlow.class).size());
        stats.put("Gateways", modelInstance.getModelElementsByType(Gateway.class).size());

        int xorSplit = 0, xorJoin = 0;
        int andSplit = 0, andJoin = 0;

        for (ExclusiveGateway gateway : modelInstance.getModelElementsByType(ExclusiveGateway.class)) {
            if (gateway.getIncoming().size() == 1 && gateway.getOutgoing().size() > 1)
                xorSplit++;
            else if (gateway.getOutgoing().size() == 1 && gateway.getIncoming().size() > 1)
                xorJoin++;
        }

        for (ParallelGateway gateway : modelInstance.getModelElementsByType(ParallelGateway.class)) {
            if (gateway.getIncoming().size() == 1 && gateway.getOutgoing().size() > 1)
                andSplit++;
            else if (gateway.getOutgoing().size() == 1 && gateway.getIncoming().size() > 1)
                andJoin++;
        }

        stats.put("XOR_Split", xorSplit);
        stats.put("XOR_Join", xorJoin);
        stats.put("AND_Split", andSplit);
        stats.put("AND_Join", andJoin);

        Map<String, Integer> gatewayTypes = new HashMap<>();
        gatewayTypes.put("Exclusive", modelInstance.getModelElementsByType(ExclusiveGateway.class).size());
        gatewayTypes.put("Parallel", modelInstance.getModelElementsByType(ParallelGateway.class).size());
        gatewayTypes.put("Inclusive", modelInstance.getModelElementsByType(InclusiveGateway.class).size());
        gatewayTypes.put("Event-Based", modelInstance.getModelElementsByType(EventBasedGateway.class).size());

        int totalGateways = gatewayTypes.values().stream().mapToInt(Integer::intValue).sum();
        stats.put("GatewaysTotal", totalGateways);
        stats.put("GatewayTypes", gatewayTypes);
    }

    //endregion

    //region Ausgabe
    public void printStats() {
        System.out.println("📄 " + fileName);
        System.out.println("    Hat Fehler: " + ((hasErrors)? "Ja" : "Nein"));
        System.out.println("    Stats:");
        System.out.printf("     - %-22s : %d%n", "Tasks", stats.getOrDefault("Tasks", 0));
        System.out.printf("     - %-22s : %d%n", "Events", stats.getOrDefault("Events", 0));
        System.out.printf("     - %-22s : %d%n", "Sequence Flows", stats.getOrDefault("Sequence Flows", 0));
        System.out.printf("     - %-22s : %d%n", "XOR_Split", stats.getOrDefault("XOR_Split", 0));
        System.out.printf("     - %-22s : %d%n", "XOR_Join", stats.getOrDefault("XOR_Join", 0));
        System.out.printf("     - %-22s : %d%n", "AND_Split", stats.getOrDefault("AND_Split", 0));
        System.out.printf("     - %-22s : %d%n", "AND_Join", stats.getOrDefault("AND_Join", 0));
        System.out.printf("     - %-22s : %d%n", "Gateways", stats.getOrDefault("Gateways", 0));

        if (stats.containsKey("GatewayTypes")) {
            System.out.println("         - Gateway Types:");
            @SuppressWarnings("unchecked")
            Map<String, Integer> gatewayTypes = (Map<String, Integer>) stats.get("GatewayTypes");
            gatewayTypes.entrySet().stream()
                    .filter(entry -> entry.getValue() > 0)
                    .forEach(entry ->
                            System.out.printf("             - %-15s : %d%n", entry.getKey(), entry.getValue())
                    );
        }

        System.out.println();
    }

    public String getStatsCsvString() {
        if (hasErrors) {
            return "";
        }
        return String.format("%d;%d;%d;%d;%d;%d;%d",
                stats.getOrDefault("Tasks", 0),
                stats.getOrDefault("Events", 0),
                stats.getOrDefault("Sequence Flows", 0),
                stats.getOrDefault("XOR_Split", 0),
                stats.getOrDefault("XOR_Join", 0),
                stats.getOrDefault("AND_Split", 0),
                stats.getOrDefault("AND_Join", 0)
        );
    }


    //endregion

//endregion
//region misc
public boolean hasModel() {
    return modelInstance != null;
}

    //endregion
}
