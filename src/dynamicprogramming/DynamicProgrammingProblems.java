package dynamicprogramming;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Use results from sub-problems to generate your final solution. Try to visualize your problem as trees.
 * Use memoization to speed up your solution since results of the sub-problems will be cached.
 *
 * Time O(2^n) for making two recursive calls at each level
 * Space O(n) since we reuse the stacks once it reaches the base condition, which is usually height of tree
 */
public class DynamicProgrammingProblems {

    public static void main(String[] args) {
        DynamicProgrammingProblems run = new DynamicProgrammingProblems();

//        int result = run.countConstruct("purple", Arrays.asList("purp", "p", "ur", "le", "purpl"), new HashMap<>());
        int result = run.gridTraveler(3,3, new HashMap<>());
        System.out.println(result);
    }

    /**
     * Given a target value and list of numbers, we need to find the shortest list of numbers that adds up to target.
     * Recursive call each time and compare the result to the running shortest result.
     * We can speed this algorithm up by adding to a cache and retrieving if it already exists in cache.
     * Time O(n*m) where n is size of number list and m is the targetSum since number list can just be made up of 1s.
     * Space O(m^2) where m is the size of cache which will be at max targetSum.
     */
    public List<Integer> bestSum(int targetSum, List<Integer> numbers, List<Integer> current, Map<Integer, List<Integer>> cache) {
        if (cache.containsKey(targetSum)) {
            return cache.get(targetSum);
        }
        if (targetSum == 0) {
            return current.isEmpty() ? null : current;
        }
        if (targetSum < 0) {
            return null;
        }

        List<Integer> result = null;
        for (Integer num : numbers) {
            current.add(num);
            List<Integer> subResult = bestSum(targetSum-num, numbers, new LinkedList<>(current), cache);
            if (result == null && subResult != null) {
                result = subResult;
            } else if (result != null && subResult != null) {
                if (subResult.size() < result.size()) {
                    result = subResult;
                }
            }
            current.remove(current.size()-1);
        }
        if (result != null) {
            cache.put(targetSum, result);
        }
        return result;
    }

    /**
     * Time O(n*m^2), where n is size of numbers and m is targetSum.
     *      We have m*m (or m^2) because targetSum amount of recursive calls is all prefixes are just one char.
     *      Also, the second m comes from the String.startsWith() method.
     * Space O(m^2), targetSum number of potential entries in cache and recursive calls within stack.
     */
    public int countConstruct(String target, List<String> vals, Map<String, Integer> cache) {
        if (cache.containsKey(target)) {
            return cache.get(target);
        }
        if (target == null || target.isEmpty()) {
            return 1;
        }

        int result = 0;
        for (String val : vals) {
            if (target.startsWith(val)) {
                result += countConstruct(target.substring(val.length()), vals, cache);
            }
        }
        cache.put(target, result);
        return result;
    }

    /**
     * allConstruct uses dynamic programming and memoization, whereas allConstruct2 uses backtracking.
     * allConstruct builds the solution from bottom up, whereas allConstruct2 builds it from top down.
     * Hence, the bottom up solution is able to take advantage of memoization.
     *
     * Given n is size of vals and m is the target value:
     * Time O(m*n^m) for both solutions. Memoization doesn't help for worst case scenario since we need to generate the output.
     * Space O(m) for both solutions.
     */
    public List<List<String>> allConstruct(String target, List<String> vals, Map<String, List<List<String>>> cache) {
        if (cache.containsKey(target)) {
            return cache.get(target);
        }
        if (target == null || target.isEmpty()) {
            return new LinkedList<>();
        }

        List<List<String>> result = new LinkedList<>();
        for (String val : vals) {
            if (target.startsWith(val)) {
                List<List<String>> subResults = allConstruct(target.substring(val.length()), vals, cache);
                if (subResults != null) {
                    if (subResults.isEmpty()) {
                        List<String> subResult = new LinkedList<>();
                        subResult.add(val);
                        result.add(subResult);
                        continue;
                    }
                    for (List<String> subResult : subResults) {
                        subResult.add(val);
                        result.add(new LinkedList<>(subResult));
                    }
                }
            }
        }

        if (result.isEmpty()) {
            cache.put(target, null);
            return null;
        }
        cache.put(target, result);
        return result;
    }

    public void allConstruct2(String target, List<String> vals, List<String> current, List<List<String>> result) {
        if ((target == null || target.isEmpty()) && !current.isEmpty()) {
            result.add(current);
            return;
        }

        for (String val : vals) {
            if (target.startsWith(val)) {
                current.add(val);
                allConstruct2(target.substring(val.length()), vals, new LinkedList<>(current), result);
                current.remove(current.size()-1);
            }
        }
    }

    /**
     * Given m*n grid, if we start at (0,0) position, and we can only go down or right, how many ways to reach (m,n) position.
     * We make recursive call with the sub problem until we reach the base case of 0 or 1.
     * We can also cache the result.
     * Time O(m*n)
     * Space O(m+n) since in worst case scenario, we need to recurse until m and n are 0, one of our base case.
     */
    public int gridTraveler(int m, int n, Map<String, Integer> cache) {
        if (m == 0 || n == 0) {
            return 0;
        }
        if (m == 1 || n == 1) {
            return 1;
        }
        String key = m + "," + n;
        if (cache.containsKey(key)) {
            return cache.get(key);
        }

        int downResult = gridTraveler(m-1, n, cache);
        int rightResult = gridTraveler(m, n-1, cache);
        cache.put(key, downResult+rightResult);
        return downResult+rightResult;
    }
}
