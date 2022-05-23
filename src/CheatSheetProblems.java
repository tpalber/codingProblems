import datastructures.*;
import java.util.*;

public class CheatSheetProblems {

    /**
     * Traversing a LinkedNode using iterative and recursive call.
     * Use pointer manipulation when necessary, like to reverse a LinkedNode.
     * Time O(n) & Space O(1)
     */
    public static void linkedNodeIterate(LinkedNode head) {
        LinkedNode current = head;
        while (current != null) {
            System.out.println(current.valInt);
            current = current.next;
        }
    }

    public static void linkedNodeRecursive(LinkedNode head) {
        if (head == null) {
            return;
        }
        System.out.println(head.valInt);
        linkedNodeRecursive(head.next);
    }

    public static LinkedNode reverseLinkedNode(LinkedNode head) {
        LinkedNode current = head;
        LinkedNode previous = null;
        while (current != null) {
            LinkedNode nextTemp = current.next;
            current.next = previous;
            previous = current;
            current = nextTemp;
        }
        return previous;
    }

    /**
     * Traversing a Tree using BFS and DFS. We are using BinaryTree here, so there is only
     * left and right child. If we are using tree with many children, we can just iterate through
     * the children by making the DFS call or add it to the Queue.
     *
     * Many problems include traversing the tree and doing something with the value. Like the
     * max tree path sum, to return the max sum from root to a leaf.
     * Time O(n) since we only go through each of the nodes
     * Space O(n) since we make recursive call at each node
     */
    public static void treeBFS(BinaryTree root) {
        Queue<Integer> queue = new LinkedList<>();
        queue.add(root.val);

        while (!queue.isEmpty()) {
            Integer val = queue.poll();
            System.out.println(val);
            if (root.left != null) {
                queue.add(root.left.val);
            }
            if (root.right != null) {
                queue.add(root.right.val);
            }
        }
    }

    public static void treeDFS(BinaryTree root) {
        if (root == null) {
            return;
        }
        System.out.println(root.val);
        treeDFS(root.left);
        treeDFS(root.right);
    }

    public static int maxTreePathSum(BinaryTree root) {
        if (root == null) {
            return 0;
        }
        int leftPathSum = maxTreePathSum(root.left);
        int rightPathSum = maxTreePathSum(root.right);
        return root.val + Math.max(leftPathSum, rightPathSum);
    }

    /**
     * For graph problems, we first need to create an adjacent Map. For each node, what are all
     * of its pointed nodes. Once we have that, we can iterate through the graph if we are given
     * the head, otherwise we iterate through each of the keys in the adjacent Map.
     * We need to also take care of scenarios where we have connected graphs. That should be considered
     * in our base case like in the largestComponent() problem.
     * Time O(e) where e is number of edges. Count of input relationships.
     * Space O(n) where n is the largest number of adjacent node for one node.
     *
     * Some Matrix problems can also be solved using tree structure, where at each position, we
     * have limited number of paths to take (left, right, up, down). We should also track our visited
     * positions, so we don't visit them again. We can usually save space by updating the existing matrix.
     * Otherwise, create a new boolean[][].
     */
    public static void graphDFS() {
        Map<String, List<String>> adjacent = new HashMap<>();
        adjacent.put("a", Arrays.asList("b","c"));
        adjacent.put("b", Collections.singletonList("d"));
        adjacent.put("f", Collections.emptyList());

        graphDFS("a", adjacent);
    }

    private static void graphDFS(String head, Map<String, List<String>> adjacent) {
        System.out.println(head);
        for (String next : adjacent.get(head)) {
            System.out.println(next);
        }
    }

    public static void graphBFS() {
        Map<String, List<String>> adjacent = new HashMap<>();
        adjacent.put("a", Arrays.asList("b","c"));
        adjacent.put("b", Collections.singletonList("d"));
        adjacent.put("f", Collections.emptyList());

        Queue<String> queue = new LinkedList<>();
        queue.add("a");
        while (!queue.isEmpty()) {
            String result = queue.poll();
            System.out.println(result);
            queue.addAll(adjacent.get(result));
        }
    }

    public static int largestComponent(Map<String, List<String>> graph) {
        int maxCount = Integer.MIN_VALUE;
        Set<String> visited = new HashSet<>();
        for (String node : graph.keySet()) {
            int currentCount = componentCount(graph, visited, node);
            maxCount = Math.max(maxCount, currentCount);
        }
        return maxCount;
    }

    private static int componentCount(Map<String, List<String>> graph, Set<String> visited, String start) {
        if (visited.contains(start)) {
            return 0;
        }
        visited.add(start);
        int count = 1;
        for (String node : graph.get(start)) {
            count += componentCount(graph, visited, node);
        }
        return count;
    }

    /**
     * Backtracking problems are essentially DFS problems, but we might not accept the final running solution,
     * so we will have to back track and check for the next running solution.
     * In the number of unique ways to climb n step stairs, given all the possible steps you can take. We track
     * the running solution (current) and result. We only add to result when we meet our valid base case and
     * once we are returned to the caller, we remove, (current.remove()), the newly added possible step from running solution to check
     * for next possible solution.
     * Time O(m^n) since we have m possible steps, and we iterate until n == 0
     * Space O(n*m^n) since we have call stack plus the running solution, if our possible steps was [1], then we
     * need to make n many recursive calls.
     */
    public static List<List<Integer>> uniqueClimbs(int n, Set<Integer> possibleSteps) {
        List<List<Integer>> result = new LinkedList<>();
        uniqueClimbDfs(n, possibleSteps, result, new LinkedList<>());
        return result;
    }

    private static void uniqueClimbDfs(int n, Set<Integer> possibleSteps, List<List<Integer>> result, List<Integer> current) {
        if (n == 0) {
            result.add(current);
            return;
        }
        if (n < 0) {
            return;
        }
        for (Integer steps : possibleSteps) {
            current.add(steps);
            uniqueClimbDfs(n-steps, possibleSteps, result, new LinkedList<>(current));
            current.remove(current.size()-1);
        }
    }
}
