package com.example.bpmn;

import org.apache.commons.text.similarity.JaroWinklerSimilarity;
import org.camunda.bpm.model.bpmn.instance.*;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class Main {
    private List<BpmnXml> experts;
    private List<BpmnXml> aalst;
    private List<BpmnXml> aau;
    private List<BpmnXml> meins;
    private List<BpmnXml> aalst_ano1;
    private List<BpmnXml> aau_ano1;
    private List<BpmnXml> meins_ano1;
    private List<BpmnXml> aalst_ano2;
    private List<BpmnXml> aau_ano2;
    private List<BpmnXml> meins_ano2;

    private final JaroWinklerSimilarity similarity = new JaroWinklerSimilarity();
    //for fitness and precision. (Name matching JaroWinklerSimilarity %age)
    private static final double SIMILARITY_THRESHOLD = 0.85;
    //für FallbackTaskMatching
    private double myContainmentRation = 0.95;

    private final StringBuilder csvBuilder = new StringBuilder();

    //matching maps for Tasks & Gateways --> important for Sequence Flows
    private Map<String, Set<String>> mapForward;
    private Map<String, Set<String>> mapReverse;


    public static void main(String[] args) {
        Main m = new Main();
        m.readFiles();
        /*
        for (BpmnXml expert : m.meins_ano1) {
            System.out.println(expert.getFileName() + " || " + expert.getGroup());
        }

         */
        //  m.printStats(m.meins);
        //  m.compareModels(m.experts, m.meins);
        //String csvContent = m.compareModels(m.experts, m.meins);
        //System.out.println(csvContent);

        // Write CSV file:
        m.writeAll();
    }

    /**
     * Reads the relevant files and fills the lists of BpmnXml.
     */
    public void readFiles() {
        experts = loadBpmnFiles("src/main/resources/bpmn/experts", Group.experts, AnonLevel.None);

        aalst = loadBpmnFiles("src/main/resources/bpmn/Aalst/Ohne Anonymisierung", Group.Aalst, AnonLevel.None);
        aau = loadBpmnFiles("src/main/resources/bpmn/Aau/Ohne Anonymisierung", Group.Aau, AnonLevel.None);
        meins = loadBpmnFiles("src/main/resources/bpmn/Meins/Ohne Anonymisierung", Group.Meins, AnonLevel.None);

        aalst_ano1 = loadBpmnFiles("src/main/resources/bpmn/Aalst/Anonymisierung_1", Group.Aalst, AnonLevel.Anon1);
        aau_ano1 = loadBpmnFiles("src/main/resources/bpmn/Aau/Anonymisierung_1", Group.Aau, AnonLevel.Anon1);
        meins_ano1 = loadBpmnFiles("src/main/resources/bpmn/Meins/Anonymisierung_1", Group.Meins, AnonLevel.Anon1);

        aalst_ano2 = loadBpmnFiles("src/main/resources/bpmn/Aalst/Anonymisierung_2", Group.Aalst, AnonLevel.Anon2);
        aau_ano2 = loadBpmnFiles("src/main/resources/bpmn/Aau/Anonymisierung_2", Group.Aau, AnonLevel.Anon2);
        meins_ano2 = loadBpmnFiles("src/main/resources/bpmn/Meins/Anonymisierung_2", Group.Meins, AnonLevel.Anon2);
    }

    /**
     * Returns the bpmn xmls of the Folder as a List of BpmnXml according to folderPath.
     */
    private List<BpmnXml> loadBpmnFiles(String folderPath, Group group, AnonLevel anonLevel) {
        List<BpmnXml> list = new ArrayList<>();
        String pathEnd = (group == Group.Aau) ? ".xml" : ".bpmn";

        try {

            List<Path> files = Files.walk(Paths.get(folderPath))
                    .filter(p -> p.toString().endsWith(pathEnd))
                    .collect(Collectors.toList());

            for (Path file : files) {
                BpmnXml model = new BpmnXml(file, group, anonLevel);
                list.add(model);
            }
        } catch (IOException e) {
            System.err.println("Failed to read from folder: " + folderPath);
            e.printStackTrace();
        }
        return list;
    }

    ///region Ausgabe
    public void writeAll() {
        // Process: Meins (Prototype)
        processGroup(meins);
        processGroup(meins_ano1);
        processGroup(meins_ano2);

        // Process: Aalst
        processGroup(aalst);
        processGroup(aalst_ano1);
        processGroup(aalst_ano2);

        // Process: Aau
        processGroup(aau);
        processGroup(aau_ano1);
        processGroup(aau_ano2);
    }

    public void writeOne(List<BpmnXml> testModels) {
        processGroup(testModels);
    }

    // Helper function to call compareModels() + CsvWriter
    private void processGroup(List<BpmnXml> testModels) {
        if (testModels.isEmpty()) {
            System.out.println("Skipping empty test list.");
            return;
        }

        // Extract group/anon from first element (they're identical for whole list)
        BpmnXml sample = testModels.get(0);
        String groupName = (sample.getGroup() == Group.Meins) ? "Prototype" : sample.getGroup().name();
        String anonName = sample.getAnonLevel().name();

        System.out.printf("Processing: %s %s...%n", groupName, anonName);

        String csvContent = compareModels(experts, testModels);
        CsvWriter writer = new CsvWriter(csvContent, testModels);
        writer.writeToFile();
    }


    private void printStats(List<BpmnXml> list) {
        if (list.isEmpty()) return;

        BpmnXml sample = list.get(0);
        System.out.println("==============================");
        System.out.println("Group     : " + sample.getGroup());
        System.out.println("AnonLevel : " + sample.getAnonLevel());
        System.out.println("==============================");

        for (BpmnXml bpmn : list) {
            bpmn.printStats();
        }
    }
    //endregion

    //region Statistics

    /**
     * compares by model name
     */
    private String compareModels(List<BpmnXml> referenceList, List<BpmnXml> testList) {
        Map<String, BpmnXml> referenceMap = referenceList.stream()
                .collect(Collectors.toMap(BpmnXml::getFileName, b -> b));

        StringBuilder csvBuilder = new StringBuilder();

        // CSV header
        csvBuilder.append("Name;Group;AnonLevel;HatFehler;Tasks;Events;SequenceFlows;XOR (Split);XOR (Join);AND (Split);AND (Join);")
                .append("task_fitness;task_precision;event_fitness;event_precision;gatewayPositioning_fitness;gatewayPositioning_precision;")
                .append("gatewayContent_fitness;gatewayContent_precision;flow_fitness;flow_precision\n");

        for (BpmnXml testModel : testList) {
            BpmnXml referenceModel = referenceMap.get(testModel.getFileName());
            if (referenceModel != null) {

                if (!testModel.hasModel() || !referenceModel.hasModel()) {
                    String row = String.format("%s;%s;%s;1;%s;;;;;;;;;;;;;;;;\n",
                            testModel.getFileName(),
                            testModel.getGroup(),
                            testModel.getAnonLevel(),
                            testModel.getStatsCsvString());
                    csvBuilder.append(row);
                    continue;
                }

                Map<String, Double> task = calculateTaskScores(referenceModel, testModel);
                Map<String, Double> event = calculateEventScores(referenceModel, testModel);
                Map<String, Double> gatewayPositioning = calculateGatewayPositioningScores(referenceModel, testModel);
                Map<String, Double> gatewayContent = calculateGatewayContentScores(referenceModel, testModel);
                Map<String, Double> flow = calculateFlowScores(referenceModel, testModel);

                String row = String.format("%s;%s;%s;0;%s;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f\n",
                        testModel.getFileName(),
                        testModel.getGroup(),
                        testModel.getAnonLevel(),
                        testModel.getStatsCsvString(),
                        task.get("task_fitness"), task.get("task_precision"),
                        event.get("event_fitness"), event.get("event_precision"),
                        gatewayPositioning.get("gatewayPositioning_fitness"), gatewayPositioning.get("gatewayPositioning_precision"),
                        gatewayContent.get("gatewayContent_fitness"), gatewayContent.get("gatewayContent_precision"),
                        flow.get("flow_fitness"), flow.get("flow_precision")
                );

                csvBuilder.append(row);
            } else {
                System.out.println("No matching reference model for: " + testModel.getFileName());
            }
        }

        return csvBuilder.toString();
    }


    //region Fitness + Precision
    //region Matching
    private Map<String, Set<String>> matchTasks(BpmnXml primary, BpmnXml secondary) {
        List<Task> primaryTasks = new ArrayList<>(primary.getModelInstance().getModelElementsByType(Task.class));
        List<Task> secondaryTasks = new ArrayList<>(secondary.getModelInstance().getModelElementsByType(Task.class));

        Map<String, Set<String>> taskMatchMap = new HashMap<>();
        Set<Integer> matchedIndexes = new HashSet<>();

        for (int i = 0; i < primaryTasks.size(); i++) {

            Task refTask = primaryTasks.get(i);
            String refId = refTask.getId();
            String refName = refTask.getName();


            if (refName == null || refId == null)
                continue;


            // 1:1 matching
            for (int j = 0; j < secondaryTasks.size(); j++) {
                if (matchedIndexes.contains(j))
                    continue;
                Task testTask = secondaryTasks.get(j);
                String testName = testTask.getName();
                if (testName == null) continue;

                if (similarity.apply(refName.toLowerCase(), testName.toLowerCase()) >= SIMILARITY_THRESHOLD) {
                    taskMatchMap.computeIfAbsent(refId, k -> new HashSet<>()).add(testTask.getId());
                    matchedIndexes.add(j);
                    break;
                }
            }

        }
        return taskMatchMap;
    }

    private void matchTasksFallback(BpmnXml base, BpmnXml compare, Map<String, Set<String>> forwardMap, Map<String, Set<String>> reverseMap) {
        Collection<Task> baseTasks = base.getModelInstance().getModelElementsByType(Task.class);
        Collection<Task> compareTasks = compare.getModelInstance().getModelElementsByType(Task.class);

        // Build sets of already matched test task IDs for forward map
        Set<String> matchedTestIds = forwardMap.values().stream()
                .flatMap(Set::stream)
                .collect(Collectors.toSet());

        for (Task refTask : baseTasks) {
            String refId = refTask.getId();
            String refName = refTask.getName();
            if (refName == null) continue;

            // Skip already matched reference tasks
            if (forwardMap.containsKey(refId))
                continue;

            for (Task testTask : compareTasks) {
                String testId = testTask.getId();
                String testName = testTask.getName();
                if (testName == null || matchedTestIds.contains(testId))
                    continue;

                int lcs = longestCommonSubsequence(refName.toLowerCase(), testName.toLowerCase());
                double ratio = (double) lcs / refName.length();

                if (ratio >= myContainmentRation) {
                    forwardMap.computeIfAbsent(refId, k -> new HashSet<>()).add(testId);
                    matchedTestIds.add(testId);
                    break;
                }
            }
        }
        /*
        // Build sets of already matched reference task IDs for reverse map
        Set<String> matchedRefIds = reverseMap.values().stream()
                .flatMap(Set::stream)
                .collect(Collectors.toSet());

        for (Task testTask : compareTasks) {
            String testId = testTask.getId();
            String testName = testTask.getName();
            if (testName == null) continue;

            // Skip already matched test tasks
            if (reverseMap.containsKey(testId))
                continue;

            for (Task refTask : baseTasks) {
                String refId = refTask.getId();
                String refName = refTask.getName();
                if (refName == null || matchedRefIds.contains(refId))
                    continue;

                int lcs = longestCommonSubsequence(testName.toLowerCase(), refName.toLowerCase());
                double ratio = (double) lcs / testName.length();

                if (ratio >= myContainmentRation) {
                    reverseMap.computeIfAbsent(testId, k -> new HashSet<>()).add(refId);
                    matchedRefIds.add(refId);
                    break;
                }
            }
        }

         */
    }


    private int longestCommonSubsequence(String a, String b) {
        int m = a.length(), n = b.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (a.charAt(i) == b.charAt(j)) {
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                } else {
                    dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
                }
            }
        }
        return dp[m][n];
    }

    private Map<String, Set<String>> matchGatewayPositioning(BpmnXml primary, BpmnXml secondary, Map<String, Set<String>> taskMatchMap) {
        Map<String, Set<String>> gatewayMatchMap = new HashMap<>();
        Set<String> matchedSecondaryGateways = new HashSet<>();

        // Process each type of gateway separately
        Class<?>[] gatewayTypes = new Class[]{
                ExclusiveGateway.class,
                ParallelGateway.class,
                InclusiveGateway.class,
                EventBasedGateway.class
        };

        for (Class<?> gatewayType : gatewayTypes) {
            Collection<? extends FlowNode> primaryGateways = primary.getModelInstance().getModelElementsByType((Class<? extends FlowNode>) gatewayType);
            Collection<? extends FlowNode> secondaryGateways = secondary.getModelInstance().getModelElementsByType((Class<? extends FlowNode>) gatewayType);

            for (FlowNode refGateway : primaryGateways) {
                String refId = refGateway.getId();
                boolean isSplit = refGateway.getIncoming().size() == 1 && refGateway.getOutgoing().size() > 1;

                for (FlowNode testGateway : secondaryGateways) {
                    String testId = testGateway.getId();

                    if (matchedSecondaryGateways.contains(testId))
                        continue;

                    boolean match = isSplit
                            ? gatewaysMatchByPredecessor(refGateway, testGateway, primary, secondary, taskMatchMap)
                            : gatewaysMatchBySuccessor(refGateway, testGateway, primary, secondary, taskMatchMap);

                    if (match) {
                        gatewayMatchMap.computeIfAbsent(refId, k -> new HashSet<>()).add(testId);
                        matchedSecondaryGateways.add(testId);
                        break;
                    }
                }
            }
        }

        return gatewayMatchMap;
    }

    private boolean gatewaysMatchBySuccessor(FlowNode refGateway, FlowNode testGateway,
                                             BpmnXml primary, BpmnXml secondary, Map<String, Set<String>> taskMatchMap) {

        FlowNode refSuccessor = findClosestMatchedSuccessor(refGateway, primary, taskMatchMap.keySet());
        FlowNode testSuccessor = findClosestMatchedSuccessor(testGateway, secondary, flattenValues(taskMatchMap));

        // --- SPECIAL-CASE: EndEvent on both sides should match even if IDs differ ---
        if (refSuccessor instanceof EndEvent && testSuccessor instanceof EndEvent) {
            return true; // do not consult taskMatchMap for terminal events
        }

        if (refSuccessor == null || testSuccessor == null)
            return false;

        Set<String> matchedTargets = taskMatchMap.getOrDefault(refSuccessor.getId(), Set.of());
        return matchedTargets.contains(testSuccessor.getId());
    }

    private boolean gatewaysMatchByPredecessor(FlowNode refGateway, FlowNode testGateway,
                                               BpmnXml primary, BpmnXml secondary, Map<String, Set<String>> taskMatchMap) {

        FlowNode refPredecessor = findClosestMatchedPredecessor(refGateway, primary, taskMatchMap.keySet());
        FlowNode testPredecessor = findClosestMatchedPredecessor(testGateway, secondary, flattenValues(taskMatchMap));

        // --- SPECIAL-CASE: StartEvent on both sides should match even if IDs differ ---
        if (refPredecessor instanceof StartEvent && testPredecessor instanceof StartEvent) {
            return true; // do not consult taskMatchMap for terminal events
        }

        if (refPredecessor == null || testPredecessor == null)
            return false;

        Set<String> matchedSources = taskMatchMap.getOrDefault(refPredecessor.getId(), Set.of());
        return matchedSources.contains(testPredecessor.getId());
    }

    private Set<String> flattenValues(Map<String, Set<String>> map) {
        return map.values().stream().flatMap(Set::stream).collect(Collectors.toSet());
    }

    private FlowNode findClosestMatchedSuccessor(FlowNode gateway, BpmnXml model, Set<String> matchedTaskIds) {
        Queue<FlowNode> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();

        for (SequenceFlow outgoing : gateway.getOutgoing()) {
            queue.add(outgoing.getTarget());
        }

        while (!queue.isEmpty()) {
            FlowNode node = queue.poll();
            if (visited.contains(node.getId()))
                continue;

            visited.add(node.getId());


            if (node instanceof Task && matchedTaskIds.contains(node.getId())) return node;
            if (node instanceof EndEvent) return node; // allow returning EndEvent as terminal witness

            for (SequenceFlow outgoing : node.getOutgoing()) {
                queue.add(outgoing.getTarget());
            }
        }
        return null;
    }


    private FlowNode findClosestMatchedPredecessor(FlowNode gateway, BpmnXml model, Set<String> matchedTaskIds) {
        Queue<FlowNode> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();

        for (SequenceFlow incoming : gateway.getIncoming()) {
            queue.add(incoming.getSource());
        }

        while (!queue.isEmpty()) {
            FlowNode node = queue.poll();
            if (visited.contains(node.getId()))
                continue;

            visited.add(node.getId());

            if (node instanceof Task && matchedTaskIds.contains(node.getId())) return node;
            if (node instanceof StartEvent) return node; // allow returning StartEvent as terminal witness

            for (SequenceFlow incoming : node.getIncoming()) {
                queue.add(incoming.getSource());
            }
        }
        return null;
    }


    private int matchSequenceFlows(BpmnXml primary, BpmnXml secondary, Map<String, Set<String>> elementMatchMap) {
        Collection<SequenceFlow> primaryFlows = primary.getModelInstance().getModelElementsByType(SequenceFlow.class);
        Collection<SequenceFlow> secondaryFlows = secondary.getModelInstance().getModelElementsByType(SequenceFlow.class);

        Set<String> testFlowPairs = secondaryFlows.stream()
                .map(f -> f.getSource().getId() + "->" + f.getTarget().getId())
                .collect(Collectors.toSet());

        int matchedCount = 0;
        Set<String> alreadyMatched = new HashSet<>();

        // Collect start/end event IDs
        Set<String> primaryStartIds = primary.getModelInstance().getModelElementsByType(StartEvent.class).stream()
                .map(FlowElement::getId).collect(Collectors.toSet());
        Set<String> primaryEndIds = primary.getModelInstance().getModelElementsByType(EndEvent.class).stream()
                .map(FlowElement::getId).collect(Collectors.toSet());
        Set<String> secondaryStartIds = secondary.getModelInstance().getModelElementsByType(StartEvent.class).stream()
                .map(FlowElement::getId).collect(Collectors.toSet());
        Set<String> secondaryEndIds = secondary.getModelInstance().getModelElementsByType(EndEvent.class).stream()
                .map(FlowElement::getId).collect(Collectors.toSet());

        for (SequenceFlow refFlow : primaryFlows) {
            String refSource = refFlow.getSource().getId();
            String refTarget = refFlow.getTarget().getId();
            if (refSource == null || refTarget == null)
                continue;

            Set<String> sourceMatches;
            Set<String> targetMatches;

            // Source matching
            if (primaryStartIds.contains(refSource)) {
                sourceMatches = new HashSet<>(secondaryStartIds);
            } else if (elementMatchMap.containsKey(refSource)) {
                sourceMatches = elementMatchMap.get(refSource);
            } else if (primaryEndIds.contains(refSource)) {
                sourceMatches = new HashSet<>(secondaryEndIds);
            } else {
                sourceMatches = Set.of(refSource);
            }

            // Target matching
            if (primaryStartIds.contains(refTarget)) {
                targetMatches = new HashSet<>(secondaryStartIds);
            } else if (elementMatchMap.containsKey(refTarget)) {
                targetMatches = elementMatchMap.get(refTarget);
            } else if (primaryEndIds.contains(refTarget)) {
                targetMatches = new HashSet<>(secondaryEndIds);
            } else {
                targetMatches = Set.of(refTarget);
            }

            for (String src : sourceMatches) {
                for (String tgt : targetMatches) {
                    String pair = src + "->" + tgt;
                    if (testFlowPairs.contains(pair) && !alreadyMatched.contains(pair)) {
                        matchedCount++;
                        alreadyMatched.add(pair);
                        break;
                    }
                }
            }
        }

        return matchedCount;
    }



    //endregion
    //mei fitness bzw precision werden base und compare einfach umgedreht
    //region Calculating
    private Map<String, Double> calculateTaskScores(BpmnXml base, BpmnXml compare) {
        this.mapForward = matchTasks(base, compare);
        this.mapReverse = matchTasks(compare, base);
        matchTasksFallback(base, compare, this.mapForward, this.mapReverse);
        matchTasksFallback(compare, base, this.mapReverse, this.mapForward);

        int matched = mapForward.size();
        int total = base.getModelInstance().getModelElementsByType(Task.class).size();
        int matchedReverse = mapReverse.size();
        int totalReverse = compare.getModelInstance().getModelElementsByType(Task.class).size();

        double fitness = total == 0 ? 1.0 : (double) matched / total;
        double precision = totalReverse == 0 ? 1.0 : (double) matchedReverse / totalReverse;

        return Map.of("task_fitness", fitness, "task_precision", precision);
    }


    private Map<String, Double> calculateFlowScores(BpmnXml base, BpmnXml compare) {

        int matched = matchSequenceFlows(base, compare, mapForward);
        int total = base.getModelInstance().getModelElementsByType(SequenceFlow.class).size();
        int matchedReverse = matchSequenceFlows(compare, base, mapReverse);
        int totalReverse = compare.getModelInstance().getModelElementsByType(SequenceFlow.class).size();

        double fitness = total == 0 ? 1.0 : (double) matched / total;
        double precision = totalReverse == 0 ? 1.0 : (double) matchedReverse / totalReverse;

        return Map.of("flow_fitness", fitness, "flow_precision", precision);
    }


    private Map<String, Double> calculateEventScores(BpmnXml base, BpmnXml compare) {
        Collection<StartEvent> baseStart = base.getModelInstance().getModelElementsByType(StartEvent.class);
        Collection<EndEvent> baseEnd = base.getModelInstance().getModelElementsByType(EndEvent.class);
        Collection<StartEvent> compareStart = compare.getModelInstance().getModelElementsByType(StartEvent.class);
        Collection<EndEvent> compareEnd = compare.getModelInstance().getModelElementsByType(EndEvent.class);

        int baseCount = baseStart.size() + baseEnd.size();
        int compareCount = compareStart.size() + compareEnd.size();

        int matched = Math.min(baseStart.size(), compareStart.size()) + Math.min(baseEnd.size(), compareEnd.size());

        double fitness = baseCount == 0 ? 1.0 : (double) matched / baseCount;
        double precision = compareCount == 0 ? 1.0 : (double) matched / compareCount;
        return Map.of("event_fitness", fitness, "event_precision", precision);
    }

    private Map<String, Double> calculateGatewayPositioningScores(BpmnXml base, BpmnXml compare) {
        Map<String, Set<String>> gatewayForward = matchGatewayPositioning(base, compare, mapForward);
        Map<String, Set<String>> gatewayReverse = matchGatewayPositioning(compare, base, mapReverse);

        // Update global maps
        gatewayForward.forEach((refId, testIds) ->
                mapForward.computeIfAbsent(refId, k -> new HashSet<>()).addAll(testIds));
        gatewayReverse.forEach((testId, refIds) ->
                mapReverse.computeIfAbsent(testId, k -> new HashSet<>()).addAll(refIds));

        int total = base.getModelInstance().getModelElementsByType(Gateway.class).size();
        int matched = gatewayForward.size();

        int totalReverse = compare.getModelInstance().getModelElementsByType(Gateway.class).size();
        int matchedReverse = gatewayReverse.size();

        double fitness = total == 0 ? 1.0 : (double) matched / total;
        double precision = totalReverse == 0 ? 1.0 : (double) matchedReverse / totalReverse;

        return Map.of("gatewayPositioning_fitness", fitness, "gatewayPositioning_precision", precision);
    }

    private Map<String, Double> calculateGatewayContentScores(BpmnXml reference, BpmnXml test) {


        BlockContent refBlock = extractSplitBlocks(reference);
        BlockContent testBlock = extractSplitBlocks(test);

        int refTasks = refBlock.tasks.size();
        int testTasks = testBlock.tasks.size();


        int matchedTasksRef = 0;

        for (String refTaskId : refBlock.tasks) {
            if (this.mapForward.containsKey(refTaskId)) {
                matchedTasksRef += this.mapForward.get(refTaskId).size();
            }
        }

        int matchedTasksTest = 0;
        for (String testTaskId : testBlock.tasks) {
            if (this.mapReverse.containsKey(testTaskId)) {
                matchedTasksTest += this.mapReverse.get(testTaskId).size();
            }
        }

        double fitness = refTasks == 0 ? 1.0 : (double) matchedTasksRef / refTasks;
        double precision = testTasks == 0 ? 1.0 : (double) matchedTasksTest / testTasks;

        return Map.of("gatewayContent_fitness", fitness, "gatewayContent_precision", precision);
    }


    public class BlockContent {
        Set<String> tasks;
        Set<String> gateways;

        public BlockContent(Set<String> tasks, Set<String> gateways) {
            this.tasks = tasks;
            this.gateways = gateways;
        }
    }

    private BlockContent extractSplitBlocks(BpmnXml model) {
        Set<String> taskIds = new HashSet<>();
        Set<String> gatewayIds = new HashSet<>();

        Collection<Gateway> allGateways = model.getModelInstance().getModelElementsByType(Gateway.class);

        for (Gateway split : allGateways) {
            if (split.getIncoming().size() == 1 && split.getOutgoing().size() > 1) {
                extractBlockRecursive(split, taskIds, gatewayIds);
            }
        }

        return new BlockContent(taskIds, gatewayIds);
    }

    private void extractBlockRecursive(FlowNode current, Set<String> taskIds, Set<String> gatewayIds) {
        Queue<FlowNode> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();

        queue.addAll(current.getOutgoing().stream()
                .map(SequenceFlow::getTarget)
                .toList());

        while (!queue.isEmpty()) {
            FlowNode node = queue.poll();
            if (!visited.add(node.getId()))
                continue; // skip if already visited

            // Collect nodes
            if (node instanceof Task) {
                taskIds.add(node.getId());
            } else if (node instanceof Gateway) {
                gatewayIds.add(node.getId());

                // If nested split, recursively process it
                if (node.getIncoming().size() == 1 && node.getOutgoing().size() > 1) {
                    extractBlockRecursive(node, taskIds, gatewayIds);
                    continue; // don't traverse again, handled by recursion
                }
            }

            // Stop if it's a join (incoming > 1 and outgoing <= 1)
            if (node.getIncoming().size() > 1 && node.getOutgoing().size() <= 1) {
                continue;
            }

            // Continue traversal
            queue.addAll(node.getOutgoing().stream()
                    .map(SequenceFlow::getTarget)
                    .toList());
        }
    }

/*
    private Map<String, Double> calculateFlowScores(BpmnXml base, BpmnXml compare) {
        int matched = matchSequenceFlows(base, compare);
        int total = base.getModelInstance().getModelElementsByType(SequenceFlow.class).size();
        int matchedReverse = matchSequenceFlows(compare, base);
        int totalReverse = compare.getModelInstance().getModelElementsByType(SequenceFlow.class).size();
        double fitness = total == 0 ? 1.0 : (double) matched / total;
        double precision = totalReverse == 0 ? 1.0 : (double) matchedReverse / totalReverse;
        return Map.of("flow_fitness", fitness, "flow_precision", precision);
    }

 */
    //endregion

}
//endregion
//endregion
