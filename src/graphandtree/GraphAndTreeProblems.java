package graphandtree;

import datastructures.BinaryTree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Traversing Binary Trees using in-order (LNR), pre-order (NLR), or post-order (LRN).
 * Time complexity of O(n)
 * Space complexity of O(1)
 *
 * Traversing Graphs using DFS and BFS. Traversal orders above can be used on DFS algorithm.
 * For DFS, we can implement with a Stack (First in Last out) or recursive call.
 * For BFS, we can implement with a Queue (First in First out).
 *
 * Traversing Matrices as if they were Graphs. If we treat each point within a matrix as a node
 * and each of adjacent points (up, down, left, right) as its edges.
 * Time Complexity O(l*w), where l is length and w is width of matrix. This should be same as graph
 *      O(e), where e is number of edges, since position within graph is an edge.
 *
 * For Graphs and Matrices, to prevent visiting nodes that has been already visited (cycles), we need
 * to either track the visited nodes in either an Array or Set of size nodes. If possible, we can
 * manipulate the existing data structure (matrix) to indicate visited.
 */
public class GraphAndTreeProblems {
    public static void main(String[] args) {
        BinaryTree root = new BinaryTree(10,
                new BinaryTree(5, new BinaryTree(3,new BinaryTree(1, null,null),null), new BinaryTree(7,new BinaryTree(6, null,null),null)),
                new BinaryTree(15, null, new BinaryTree(18,null,null)));

        System.out.println(treeMaxPathSum(root));
    }

    /**
     * Stack or recursive call to traverse a binary tree using DFS.
     * Time O(n) where n is number of nodes.
     * Space O(n) for creating a stack or keeping track of recursive call.
     */
    public static void treeDFS(TreeNode root) {
        if (root == null) {
            return;
        }
        System.out.println(root.valString);
        treeDFS(root.left);
        treeDFS(root.right);
    }

    /**
     * Queue to traverse a binary tree using BFS.
     * Time O(n) where n is number of nodes.
     * Space O(n) for creating a queue.
     */
    public static void treeBFS(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode current = queue.poll();
            System.out.println(current.valString);
            if (current.left != null) {
                queue.add(current.left);
            }
            if (current.right != null) {
                queue.add(current.right);
            }
        }
    }

    /**
     * Check if binary tree contains the target value.
     * Same time and space complexity as a tree traversal from above.
     */
    public static boolean treeIncludes(TreeNode root, String target) {
        if (root == null) {
            return false;
        }
        if (root.valString.equals(target)) {
            return true;
        }

        return treeIncludes(root.left, target) || treeIncludes(root.right, target);
    }

    /**
     * Find the total sum of the tree, this could return int sum if node values are an int.
     * Same time and space complexity as tree traversal from above.
     */
    public static String treeSum(TreeNode root) {
        if (root == null) {
            return "";
        }

        return root.valString + treeSum(root.left) + treeSum(root.right);
    }

    /**
     * Find the min value of the total tree. Since we are dealing with String values, we will just do string comparison.
     * Same time and space complexity as tree traversal from above.
     */
    public static String treeMinDFS(TreeNode root) {
        if (root == null) {
            return "zz";
        }

        // If comparing numbers, we can use Math.min(root.valString, treeMinDFS(root.left), treeMinDFS(root.right));
        String minVal = root.valString;
        String current = treeMinDFS(root.left);
        if (minVal.compareTo(current) > 0) {
            minVal = current;
        }

        current = treeMinDFS(root.right);
        if (minVal.compareTo(current) > 0) {
            minVal = current;
        }
        return minVal;
    }

    /**
     * Find the max of all the values added up from root to leaf node.
     */
    public static int treeMaxPathSum(BinaryTree root) {
        if (root.left == null && root.right == null) {
            return root.val;
        }
        int leftMathPathSum = Integer.MIN_VALUE;
        if (root.left != null) {
            leftMathPathSum = treeMaxPathSum(root.left);
        }
        int rightMathPathSum = Integer.MIN_VALUE;
        if (root.right != null) {
            rightMathPathSum = treeMaxPathSum(root.right);
        }
        return Math.max(leftMathPathSum, rightMathPathSum) + root.val;
    }

    public static int maxTreePathSum(datastructures.BinaryTree root) {
        if (root == null) {
            return 0;
        }
        int leftPathSum = maxTreePathSum(root.left);
        int rightPathSum = maxTreePathSum(root.right);
        return root.val + Math.max(leftPathSum, rightPathSum);
    }

    /**
     * Unival tree is a subtree where all the values (Node, Left, and Right) all have the same value or is null.
     * O(n) We can use post-order traversal and add on to our existing counter if our subtree matches our requirement.
     */
    public static int univalTreeCount(TreeNode node) {
        if (node == null) {
            return 1;
        }
        int count = 0;
        if (node.left != null && node.right != null) {
            if (node.left.val == node.val && node.right.val == node.val) {
                count++;
            }
            count += univalTreeCount(node.left);
            count += univalTreeCount(node.right);
        } else if (node.left != null) {
            if (node.left.val == node.val) {
                count++;
            }
            count += univalTreeCount(node.left);
        } else if (node.right != null) {
            if (node.right.val == node.val) {
                count++;
            }
            count += univalTreeCount(node.right);
        } else {
            count++;
        }
        return count;
    }

    public static boolean isTreeSymmetric(BinaryTree node1, BinaryTree node2) {
        if (node1 == null && node2 == null) {
            return true;
        } else if (node1 == null || node2 == null || node1.val != node2.val) {
            return false;
        }
        return isTreeSymmetric(node1.left, node2.right) && isTreeSymmetric(node1.right, node2.left);
    }

    /**
     * Find the max sum of sequential path along a Binary Tree.
     * MaxSum is max of following:
     *      1. left tree + current node
     *      2. right tree + current node
     *      3. left tree + right tree + current node
     *      4. previous max
     * The key here is to update the max but return the max of option 1 or 2.
     * Hence, we are using AtomicInteger.
     * Integer variable that can be updated whenever, we can also a reference class.
     */
    public static int maxSumPath(BinaryTree root, AtomicInteger maxSum) {
        if (root == null) {
            return 0;
        }

        int leftSum = maxSumPath(root.left, maxSum);
        int rightSum = maxSumPath(root.right, maxSum);

        int maxChildSum = Math.max(Math.max(leftSum, rightSum) + root.val, root.val);
        int maxTotal = Math.max(maxChildSum, leftSum + rightSum + root.val);

        maxSum.set(Math.max(maxSum.get(), maxTotal));
        return maxChildSum;
    }

    /**
     * Very similar to maxSumPath() but we keep track of the number of nodes with same value.
     */
    public static int maxSamePath(BinaryTree root) {
        AtomicInteger maxResult = new AtomicInteger(Integer.MIN_VALUE);
        dfs(root, maxResult);
        return maxResult.get();
    }

    public static int dfs(BinaryTree root, AtomicInteger maxResult) {
        if (root == null) {
            return 0;
        }
        int maxSameLeftPath = dfs(root.left, maxResult);
        int maxSameRightPath = dfs(root.right, maxResult);
        if (root.left != null && root.left.val == root.val) {
            maxSameLeftPath++;
        }
        if (root.right != null && root.right.val == root.val) {
            maxSameRightPath++;
        }

        if (maxSameLeftPath + maxSameRightPath > maxResult.get()) {
            maxResult.set(maxSameLeftPath + maxSameRightPath);
        }
        return Math.max(maxSameLeftPath, maxSameRightPath);
    }

    /**
     * Given a Binary Search Tree, which means nodes of left < root and nodes on right > root.
     * Instead of traversing every node, brute force, we just check the nodes that are potentially in range
     * since we know which nodes are greater or lesser than current node.
     * Time O(logn) since we don't look at every node.
     * Space O(1)
     */
    public static int sumBetweenRange(BinaryTree root, int low, int high) {
        // val is less than low, then only check right
        // val is greater than high, then only check left
        // otherwise check both
        if (root == null) {
            return 0;
        }

        if (root.val >= low && root.val <= high) {
            int leftSum = sumBetweenRange(root.left, low, high);
            int rightSum = sumBetweenRange(root.right, low, high);
            return root.val + leftSum + rightSum;
        } else if (root.val < low) {
            return sumBetweenRange(root.right, low, high);
        } else {
            return sumBetweenRange(root.left, low, high);
        }
    }

    /**
     * Given a matrix, at each position, we have two traversal options matrix[i+1][j] or matrix[i][j+1]
     * if those values are greater than the current position. Find the max path length starting at 0,0.
     *
     * Time O(logn)
     */
    public static int longestMatrixPath(int[][] matrix) {
        // At each position, we have two options matrix[i+1][j] or matrix[i][j+1]
        return 1 + longestMatrixPath(matrix, 0, 0);
    }

    public static int longestMatrixPath(int[][] matrix, int rowIndex, int colIndex) {
        int nextRow = 0;
        if (rowIndex+1 < matrix.length && matrix[rowIndex+1][colIndex] > matrix[rowIndex][colIndex]) {
            nextRow = 1 + longestMatrixPath(matrix, rowIndex+1, colIndex);
        }
        int nextCol = 0;
        if (colIndex+1 < matrix[rowIndex].length && matrix[rowIndex][colIndex+1] > matrix[rowIndex][colIndex]) {
            nextCol = 1+ longestMatrixPath(matrix, rowIndex, colIndex+1);
        }

        return Math.max(nextRow, nextCol);
    }

    /**
     * DFS traversal of a complete graph (all nodes are connected and no cycles) using Stack
     * O(e) If we implement a visited array and don't visit it again,
     *      therefore we only iterate once through each of the graph's edges.
     */
    public static void graphDFS() {
        Map<String, List<String>> adjacent = new HashMap<>();
        adjacent.put("a", Arrays.asList("b","c"));
        adjacent.put("b", Collections.singletonList("d"));
        adjacent.put("c", Collections.singletonList("e"));
        adjacent.put("d", Collections.singletonList("f"));
        adjacent.put("e", Collections.emptyList());
        adjacent.put("f", Collections.emptyList());

        Stack<String> stack = new Stack<>();
        stack.add("a");
        while (!stack.isEmpty()) {
            String current = stack.pop();
            System.out.println(current);
            // If order doesn't matter, otherwise we can add to stack in the correct order.
            stack.addAll(adjacent.get(current));
        }
    }

    /**
     * BFS traversal of a complete graph (all nodes are connected and no cycles) using Queue
     * O(e) If we implement a visited array and don't visit it again,
     *      therefore we only iterate once through each of the graph's edges.
     */
    public static void graphBFS() {
        Map<String, List<String>> adjacent = new HashMap<>();
        adjacent.put("a", Arrays.asList("b","c"));
        adjacent.put("b", Collections.singletonList("d"));
        adjacent.put("c", Collections.singletonList("e"));
        adjacent.put("d", Collections.singletonList("f"));
        adjacent.put("e", Collections.emptyList());
        adjacent.put("f", Collections.emptyList());

        Queue<String> queue = new LinkedList<>();
        queue.add("a");
        while (!queue.isEmpty()) {
            String current = queue.poll();
            System.out.println(current);
            queue.addAll(adjacent.get(current));
        }
    }

    /**
     * Given start and end, check if there is a path from start to end within a graph.
     * Time complexity O(e), where e is number of edges
     * Space complexity O(n), where n is the number of nodes
     */
    public static boolean hasPath(Map<String, List<String>> adjacent, String start, String end) {
        Stack<String> stack = new Stack<>();
        stack.add(start);

        while (!stack.isEmpty()) {
            String current = stack.pop();
            if (current.equals(end)) {
                return true;
            }

            stack.addAll(adjacent.get(current));
        }
        return false;
    }

    /**
     * O(1) space complexity. Better than creating a Stack like the above version, which will have O(n).
     */
    public static boolean hasPathRecursive(Map<String, List<String>> adjacent, String start, String end) {
        if (start.equals(end)) {
            return true;
        }
        for (String neighbor : adjacent.get(start)) {
            if (hasPathRecursive(adjacent, neighbor, end)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Given a cyclic graph, where there could be a possible infinite loop. We check to see if there exists a path
     *      from start to end. Only difference from above method is that we now add an isVisited Set,
     *      which return true if we have already visited the node otherwise add to the Set. If we are given
     *      a list of edges (Check shortestPath() for how to build graph from edges), where we get list of
     *      connections between two nodes, we need to first turn it into an adjacent list. We can achieve that by
     *      iterating the edges list and adding both nodes of an edge.
     * Same time and space complexity as hasPathRecursive().
     */
    public static boolean hasPathRecursiveInCyclicGraph(Map<String, List<String>> adjacent, String start, String end) {
        Set<String> isVisited = new HashSet<>();
        return hasPathRecursiveInCyclicGraph(adjacent, start, end, isVisited);
    }

    public static boolean hasPathRecursiveInCyclicGraph(Map<String, List<String>> adjacent,
                                                        String start,
                                                        String end,
                                                        Set<String> isVisited) {
        if (start.equals(end)) {
            return true;
        }
        if (isVisited.contains(start)) {
            return false;
        }

        isVisited.add(start);
        for (String neighbor : adjacent.get(start)) {
            if (hasPathRecursiveInCyclicGraph(adjacent, neighbor, end, isVisited)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Find the count of different connected graphs. We can iterate through each of the nodes and check
     *      to see if it has been visited. Otherwise, we DFS traverse (recursive) and add all the nodes within
     *      the connected graph to the visited Set.
     * Space complexity O(e), where e is number of edges.
     * Time complexity O(n), where n is number of nodes for the isVisited Set.
     */
    public static int connectedComponentsCount(Map<String, List<String>> adjacent) {
        Set<String> isVisited = new HashSet<>();
        int count = 0;
        for (String key : adjacent.keySet()) {
            if (!isVisited.contains(key)) {
                setVisited( adjacent, isVisited, key);
                count ++;
            }
        }
        return count;
    }

    public static void setVisited(Map<String, List<String>> adjacent, Set<String> isVisited, String start) {
        if (isVisited.contains(start)) {
            return;
        }

        isVisited.add(start);
        for (String neighbor : adjacent.get(start)) {
            setVisited( adjacent, isVisited, neighbor);
        }
    }

    /**
     * Find the count of the largest connected component within graph containing multiple connected components.
     *      Similar step as before, but now we return the count of nodes within the connected component and then
     *      compare it with the existing max count.
     * Space complexity O(e), where e is number of edges.
     * Time complexity O(n), where n is number of nodes for the isVisited Set.
     */
    public static int largestComponent(Map<String, List<String>> graph) {
        int maxCount = -1;
        Set<String> visited = new HashSet<>();

        for (String node : graph.keySet()) {
            int currentCount = componentCount(graph, visited, node);
            if (currentCount > maxCount) {
                maxCount = currentCount;
            }
        }
        return maxCount;
    }

    public static int componentCount(Map<String, List<String>> graph, Set<String> visited, String start) {
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
     * Find the shortest path between two nodes. BFS will always search the shortest distance first,
     *      hence no need to track min shortest path.
     *
     * 1. Convert edges to adjacent Map
     * 2. BFS and add level of BFS plus the node to the queue
     * 3. Keep track of visited Set, so we don't go into loop
     * 4. Poll from the queue and if we find that the current matches end,
     *      then we add to it and return.
     * 5. Else we add all the neighbors to the queue with the current length incremented by 1.
     */
    public static int shortestPathBFS(List<String[]> edges, String start, String end) {
        Map<String, List<String>> adjacentMap = new HashMap<>();
        for (String[] edge : edges) {
            List<String> adjacentNodes = adjacentMap.getOrDefault(edge[0], new ArrayList<>());
            adjacentNodes.add(edge[1]);
            adjacentMap.put(edge[0], adjacentNodes);

            adjacentNodes = adjacentMap.getOrDefault(edge[1], new ArrayList<>());
            adjacentNodes.add(edge[0]);
            adjacentMap.put(edge[1], adjacentNodes);
        }

        Queue<String[]> queue = new LinkedList<>();
        queue.add(new String[] {start, "0"});
        Set<String> visited = new HashSet<>();
        while (!queue.isEmpty()) {
            String[] current = queue.poll();
            visited.add(current[0]);
            if (current[0].equals(end)) {
                return Integer.parseInt(current[1]);
            }
            for (String neighbor : adjacentMap.get(current[0])) {
                if (!visited.contains(neighbor)) {
                    queue.add(new String[]{neighbor, String.valueOf(Integer.parseInt(current[1]) + 1)});
                }
            }
        }
        return -1;
    }

    public static int shortestPathDFS(List<String[]> edges, String start, String end) {
        Map<String, List<String>> adjacentMap = new HashMap<>();
        for (String[] edge : edges) {
            List<String> adjacentNodes = adjacentMap.getOrDefault(edge[0], new ArrayList<>());
            adjacentNodes.add(edge[1]);
            adjacentMap.put(edge[0], adjacentNodes);

            adjacentNodes = adjacentMap.getOrDefault(edge[1], new ArrayList<>());
            adjacentNodes.add(edge[0]);
            adjacentMap.put(edge[1], adjacentNodes);
        }

        int shortestLength = Integer.MAX_VALUE;
        for (String node : adjacentMap.get(start)) {
            Set<String> visited = new HashSet<>();
            visited.add(start);
            int currentLength = pathLength(adjacentMap, visited, node, end);
            if (currentLength > 0 && currentLength < shortestLength) {
                shortestLength = currentLength;
            }
        }
        return shortestLength;
    }

    public static int pathLength(Map<String, List<String>> adjacentMap, Set<String> visited, String start, String end) {
        if (visited.contains(start)) {
            return 0;
        }
        visited.add(start);
        int count = 1;
        if (start.equals(end)) {
            return count;
        }

        for (String node : adjacentMap.get(start)) {
            count += pathLength(adjacentMap, visited, node, end);
        }
        return count;
    }

    /**
     * Course Schedule, given number of courses and prereqs, we need to check if we are able
     * to complete the courses (check for cycles).
     * O(n+p) = O(n) where n is number of classes and p is number of prerequisites.
     * - We create a visited Set that holds all the courses we are visiting.
     * - We create HashMap<Integer, Integer[]> of course and its prerequisites.
     * -
     * numCourses = 3   [1,0][0,3]
     *
     * Create adjacent Map of courses and their prereqs. Which we will use to iterate. After
     * each iteration, we clear the visited array/map to we can count for scenarios where
     * multiple courses are relied on one course. To optimize this solution, we can create a
     * marked array, which will return false for checkCycles and is only set when the visited
     * course has no cycles.
     */
    public static boolean courseSchedule(int numCourses, int[][] prerequisites) {
        Map<Integer, List<Integer>> adjacent = new HashMap<>();
        for (int i = 0; i < numCourses; i++) {
            adjacent.put(i, new ArrayList<>());
        }
        for (int i = 0; i < prerequisites.length; i++) {
            adjacent.get(prerequisites[i][0]).add(prerequisites[i][1]);
        }

        for (Integer course : adjacent.keySet()) {
            boolean[] visited = new boolean[numCourses];
            if (!visited[course]) {
                if (checkCycles(adjacent, course, visited)) {
                    return false;
                }
            }
        }
        return true;
    }

    public static boolean checkCycles(Map<Integer, List<Integer>> adjacent, int course, boolean[] visited) {
        if (visited[course]) {
            return true;
        }
        visited[course] = true;
        for (Integer edge : adjacent.get(course)) {
            if (checkCycles(adjacent, edge, visited)) {
                return true;
            }
        }
        return false;
    }

    public static List<Integer> findOrder(int numCourses, int[][] prerequisites) {
        Map<Integer, List<Integer>> adjacentMap = createAdjacentMap(numCourses, prerequisites);
        // <0,[1]>
        // <1,[]>

        List<Integer> result = new LinkedList<>();
        List<Integer> visited = new LinkedList<>();
        for (Map.Entry<Integer, List<Integer>> entry : adjacentMap.entrySet()) {
            if (!visited.contains(entry.getKey())) {
                visited.add(entry.getKey());
                List<Integer> children = findOrder(entry.getKey(), visited, adjacentMap);
                if (children != null) {
                    result.addAll(children);
                } else {
                    visited.remove(visited.size()-1);
                }
            }
        }

        if (result.size() == numCourses) {
            return result;
        }
        return new LinkedList<>();
    }

    public static List<Integer> findOrder(int course, List<Integer> visited, Map<Integer, List<Integer>> adjacentMap) {
        List<Integer> prereqs = adjacentMap.get(course);
        if (prereqs.size() == 0) {
            return visited;
        }

        for (Integer prereq : prereqs) {
            if (!visited.contains(prereq)) {
                visited.add(prereq);
                List<Integer> result = findOrder(prereq, new LinkedList<>(visited), adjacentMap);
                if (result != null) {
                    return result;
                }
                visited.remove(visited.size()-1);
            }
        }
        return null;
    }


    public static Map<Integer, List<Integer>> createAdjacentMap(int numCourses, int[][] prerequisites) {
        Map<Integer, List<Integer>> adjList = new HashMap<>();
        for (int i = 0; i < prerequisites.length; i++) {
            int dest = prerequisites[i][0];
            int src = prerequisites[i][1];
            List<Integer> lst = adjList.getOrDefault(src, new ArrayList<Integer>());
            lst.add(dest);
            adjList.put(src, lst);
        }
        return adjList;
    }

    /**
     * Number of islands is same problem as number of connected graphs. We consider all out of bound indexes and 0
     *      neighbors as nodes without child. We can improve on our graph implementation by setting the values within
     *      matrix to 0 if we visit it instead of keep a visited matrix.
     * O(m*n) since we only visit each position within the matrix once.
     *
     * 1. Iterate through each position within matrix that isn't a water (value set to 0).
     * 2. Make a DFS from that position to each of its edges (up, down, left, right).
     * 3. At the position, we make sure rowIndex and colIndex aren't out of bounds. If so, set value to 0.
     * 4. Once we are out of DFS, we increment our island counter.
     */
    public static int numberOfIslands(int[][] matrix) {
        int islandCounter = 0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] == 1) {
                    dfsIsland(matrix, i, j);
                    islandCounter++;
                }
            }
        }
        return islandCounter;
    }

    public static void dfsIsland(int[][] matrix, int rowIndex, int colIndex) {
        if (rowIndex < 0 || rowIndex >= matrix.length || colIndex < 0
                || colIndex >= matrix[rowIndex].length || matrix[rowIndex][colIndex] == 0) {
            return;
        }
        matrix[rowIndex][colIndex] = 0;
        dfsIsland(matrix, rowIndex-1, colIndex);
        dfsIsland(matrix, rowIndex+1, colIndex);
        dfsIsland(matrix, rowIndex, colIndex-1);
        dfsIsland(matrix, rowIndex, colIndex+1);
    }

    /**
     * Min island size is same problem as short path DFS implementation but with matrix instead of a graph.
     *
     * 1. Iterate through each position within matrix that isn't a water (value set to 0).
     * 2. Make a DFS from that position to each of its edges (up, down, left, right), which returns a count of nodes.
     * 3. At the position, we make sure rowIndex and colIndex aren't out of bounds or 0. If so, set value to 0 and return 1.
     * 4. We add the DFS result count to current running count and compare that count to the global min count.
     */
    public static int minIslandSize(int[][] matrix) {
        int minIslandSize = Integer.MAX_VALUE;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] == 1) {
                    int currentSize = dfsIslandSize(matrix, i, j);
                    if (currentSize > 0 && currentSize < minIslandSize) {
                        minIslandSize = currentSize;
                    }
                }
            }
        }
        return minIslandSize;
    }

    public static int dfsIslandSize(int[][] matrix, int i, int j) {
        if (i < 0 || j < 0 || i >= matrix.length || j >= matrix[i].length || matrix[i][j] == 0) {
            return 0;
        }
        matrix[i][j] = 0;
        int count = 1;
        count += dfsIslandSize(matrix, i-1, j);
        count += dfsIslandSize(matrix, i+1, j);
        count += dfsIslandSize(matrix, i, j-1);
        count += dfsIslandSize(matrix, i, j+1);
        return count;
    }

    /**
     * Mixed Examples ==================================================================
     */

    /**
     * We create a decision tree, where we can select the current or next value.
     *      Then for the next value, we can't select the value right after the current one,
     *      but we can select the one after that.
     * O(2^n) since we can either select the next or one after next.
     *
     * [9,2,4,5] -> 14
     *    9          2
     *  4   5        5
     *  --------------
     * 13 , 14 ,     7
     */
    public static int rob(int[] nums) {
        return traverse(nums, 0, 0);
    }

    public static int traverse(int[] nums, int currentSum, int start) {
        if (start >= nums.length) {
            return currentSum;
        }

        int leftSum = 0;
        if (start < nums.length) {
            leftSum = traverse(nums, nums[start] + currentSum, start + 2); // 13
        }
        int rightSum = 0;
        if (start + 1 < nums.length) {
            rightSum = traverse(nums, nums[start + 1] + currentSum, start + 3); // 14
        }

        return Math.max(leftSum, rightSum);
    }

    /**
     * Assuming val can't contain comma char and string null, we use it to denote end of node and empty child.
     *          root
     *        left  right
     *    left.left
     *
     * "root, left, right, left.left, null, null, null, null, null,"
     *    0 ,   1 ,   2  ,    3     ,   4 ,  5  ,  6  ,  7  ,  8
     *
     *    1, 4 = 3
     *    2, 6 = 4
     *    3, 8 = 5
     *
     * Algorithm to find the next left and right children given your index
     * left = index*2 + 1;
     * right = (index*3 % 2 == 0) ? (index*3) : (index*3 + 1);
     */
    public static String serialize(TreeNode root) {
        StringBuilder result = new StringBuilder();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int currentQueueSize = queue.size();
            System.out.println(currentQueueSize);

            for (int i = 0; i < currentQueueSize; i++) {
                TreeNode current = queue.poll();
                result.append(current.valString == null ? "null" : current.val).append(',');
                if (current.valString == null) {
                    continue;
                }

                TreeNode emptyNode = new TreeNode(null, null, null);
                if (current.left != null) {
                    queue.add(current.left);
                } else {
                    queue.add(emptyNode);
                }

                if (current.right != null) {
                    queue.add(current.right);
                } else {
                    queue.add(emptyNode);
                }
            }
        }
        return result.toString();
    }

    public static TreeNode deserialize(String nodeString) {
        String[] values = nodeString.split(",");
        TreeNode root = new TreeNode(values[0], null, null);

        root.left = deserializeHelper(values, 1);
        root.right = deserializeHelper(values, 2);
        return root;
    }

    public static TreeNode deserializeHelper(String[] values, int index) {
        if (index >= values.length || Objects.equals(values[index], "null")) {
            return null;
        }

        TreeNode current = new TreeNode(values[index], null, null);
        current.left = deserializeHelper(values, index*2 + 1);
        current.right = deserializeHelper(values, (index*3 % 2 == 0) ? (index*3) : (index*3 + 1));
        return current;
    }

    /**
     * O(n) we need to keep track of the min and max at each node.
     *      since we could get into tree structure where the immediate child are less than root but
     *      the child's child has a value greater than the root.
     */
    public static boolean isValidBST(TreeNode root) {
        return isValidBST(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }

    public static boolean isValidBST(TreeNode root, int min, int max) {
        if (root == null) {
            return true;
        }
        if (root.val <= min && root.val >= max) {
            return false;
        }

        return isValidBST(root.left, min, root.val) && isValidBST(root.right, root.val, max);
    }

    /**
     * O(n^n) We need to understand how to structure the tree that helps with our backtracking
     * solution. Children of each node are the possible subsets without including the node itself
     * So first nodes are 0, 00, 001, 0012
     * Then 0 -> 0, 01, 012, which are all the possible subsets without including the node itself
     */
    public static boolean splitString(String s) {
        // 0012
        //          0                   00              001               0012
        //    0     01    012          1  12             2
        //  1 12    2                 2
        // 2

        //[0, 0, 1, 2] [0, 0, 12] [0, 01, 2] [0, 012] [00, 1, 2]  [00, 12] [001, 2]
        return dfs(s, 0, new LinkedList<>());
    }

    public static boolean dfs(String s, int substringIndex, List<Integer> currentSubsets) {
        if (substringIndex >= s.length() && currentSubsets.size() > 1 && isDescending(currentSubsets)) {
            return true;
        }

        for (int i = substringIndex; i < s.length(); i++) {
            String currentString = s.substring(substringIndex, i+1);
            int currentValue = Integer.parseInt(currentString);
            currentSubsets.add(currentValue);
            if (dfs(s, i+1, currentSubsets)) {
                return true;
            }
            currentSubsets.remove(currentSubsets.size()-1);
        }
        return false;
    }

    public static boolean isDescending(List<Integer> currentResult) {
        Integer prevNum = currentResult.get(0);
        for (int i = 1; i < currentResult.size(); i++) {
            if (prevNum != (currentResult.get(i) + 1)) {
                return false;
            }
            prevNum = currentResult.get(i);
        }
        return true;
    }

    public static class Tree {
        char val;
        Set<Tree> children = new HashSet<>();

        public Tree(char val) {
            this.val = val;
        }

        public Tree(char val, Set<Tree> children) {
            this.val = val;
            this.children.addAll(children);
        }

        public Tree(char val, Tree child) {
            this.val = val;
            this.children.add(child);
        }

        public static Tree insert(Tree tree, String query, int startIndex) {
            if (startIndex >= query.length()) {
                return tree;
            }

            for (Tree child : tree.children) {
                if (child.val == query.charAt(startIndex)) {
                    return insert(child, query, startIndex+1);
                }
            }
            Tree newChild = new Tree(query.charAt(startIndex));
            tree.children.add(newChild);
            return insert(newChild, query, startIndex+1);
        }

        public static Set<String> findWithPrefix(Tree tree, String query) {
            Set<String> autoCompletes = new HashSet<>();
            findWithPrefix(tree, query, 1, autoCompletes, "d");
            return autoCompletes;
        }

        private static void findWithPrefix(Tree tree, String query, int startIndex, Set<String> autoCompletes, String currentString) {
            if (tree.children == null || tree.children.isEmpty()) {
                autoCompletes.add(currentString);
                return;
            }
            for (Tree child : tree.children) {
                if (startIndex >= query.length()) {
                    currentString = currentString + child.val;
                    findWithPrefix(child, query, startIndex+1, autoCompletes, currentString);
                    currentString = currentString.substring(0, currentString.length()-1);
                } else if (child.val == query.charAt(startIndex)) {
                    currentString = currentString + child.val;
                    findWithPrefix(child, query, startIndex+1, autoCompletes, currentString);
                    break;
                }
            }
        }
    }

    public static class Trie {
        char val;
        Map<Character, Trie> children = new HashMap<>();

        public Trie() {}

        public Trie(char val) {
            this.val = val;
        }

        public void insert(String word) {
            this.val = word.charAt(0);
            insert(word, 1);
        }

        private void insert(String word, int startIndex) {
            if (startIndex >= word.length()) {
                this.children.put(null, null);
                return;
            }

            if (this.children.containsKey(word.charAt(startIndex))) {
                Trie matchingChild = this.children.get(word.charAt(startIndex));
                matchingChild.insert(word, startIndex+1);
            } else {
                Trie newChild = new Trie(word.charAt(startIndex));
                this.children.put(word.charAt(startIndex), newChild);
                newChild.insert(word, startIndex+1);
            }
        }

        public boolean search(String word) {
            if (word.charAt(0) == this.val) {
                return search(word, 1);
            }
            return false;
        }

        private boolean search(String word, int startIndex) {
            if (startIndex >= word.length()) {
                return this.children.containsKey(null);
            }
            if (this.children.containsKey(word.charAt(startIndex))) {
                Trie matchingChild = this.children.get(word.charAt(startIndex));
                return matchingChild.search(word, startIndex+1);
            }
            return false;
        }

        public boolean startsWith(String prefix) {
            if (prefix.charAt(0) == this.val) {
                return startsWith(prefix, 1);
            }
            return false;
        }

        private boolean startsWith(String prefix, int startIndex) {
            if (startIndex >= prefix.length()) {
                return true;
            }
            if (this.children.containsKey(prefix.charAt(startIndex))) {
                Trie matchingChild = this.children.get(prefix.charAt(startIndex));
                return matchingChild.startsWith(prefix, startIndex+1);
            }
            return false;
        }
    }

    public static class TreeNode {
        //        TreeNode root = new TreeNode(3, new TreeNode(9), new TreeNode(20, new TreeNode(15), new TreeNode(7)));
        int val;

        String valString;
        TreeNode left;
        TreeNode right;
        TreeNode next;

        TreeNode random;

        Integer randomIndex;

        TreeNode() {}
        TreeNode(int val) { this.val = val; }

        TreeNode(int val, TreeNode next, Integer random) {
            this.val = val;
            this.next = next;
            this.randomIndex = random;
        }

        TreeNode(String valString, TreeNode left, TreeNode right) {
            this.valString = valString;
            this.left = left;
            this.right = right;
        }

        TreeNode(int val, TreeNode next, TreeNode random) {
            this.val = val;
            this.next = next;
            this.random = random;
        }

        TreeNode(int val, TreeNode next, TreeNode left, TreeNode right, TreeNode random) {
            this.val = val;
            this.next = next;
            this.left = left;
            this.right = right;
            this.random = random;
        }
    }
    public static class BinaryTree {
        private int val;
        private BinaryTree left;
        private BinaryTree right;

        BinaryTree(int val, BinaryTree left, BinaryTree right) {
            this.val = val;
            this.right = right;
            this.left = left;
        }
    }
}
