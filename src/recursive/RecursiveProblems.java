package recursive;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class RecursiveProblems {
    public static void main(String[] args) {
        RecursiveProblems run = new RecursiveProblems();

        int result = run.numSubMatrices(new int[][]{{0,1,0},{1,1,1},{0,1,0}}, 0, 3, 0, 3, 0, 2);
        System.out.println(result);
    }

    public static int[] mergeSort(int[] values) {
        divideAndConcur(values, 0, values.length-1);
        return values;
    }

    private static void divideAndConcur(int[] values, int start, int end) {
        if (start >= end) {
            return;
        }
        int mid = ((end - start) / 2) + start;
        divideAndConcur(values, start, mid);
        divideAndConcur(values, mid+1, end);
        merge(values, start, mid, end);
    }

    private static void merge(int[] values, int start, int mid, int end) {
        int[] temp = new int[end-start+1];

        int firstIndex = start;
        int secondIndex = mid+1;
        int tempIndex = 0;
        while(firstIndex <= mid && secondIndex <= end) {
            if (values[firstIndex] <= values[secondIndex]) {
                temp[tempIndex] = values[firstIndex];
                firstIndex++;
            } else {
                temp[tempIndex] = values[secondIndex];
                secondIndex++;
            }
            tempIndex++;
        }
        while(firstIndex <= mid) {
            temp[tempIndex] = values[firstIndex];
            firstIndex++;
            tempIndex++;
        }
        while(secondIndex <= end) {
            temp[tempIndex] = values[secondIndex];
            secondIndex++;
            tempIndex++;
        }

        for (int j : temp) {
            values[start] = j;
            start++;
        }
    }

    public static int sumOfNaturalNumbers(int n) {
        if (n == 0) {
            return 0;
        }
        return sumOfNaturalNumbers(n-1) + n;
    }

    public static int fibOptimized(int n) {
        Map<Integer, Integer> fibCache = new HashMap<>();
        fibCache.put(0, 0);
        fibCache.put(1, 1);

        return fib(n, fibCache);
    }

    private static int fib(int n, Map<Integer, Integer> fibCache) {
        if (fibCache.containsKey(n)) {
            return fibCache.get(n);
        }
        return fib(n-1, fibCache) + fib(n-2, fibCache);
    }

    /**
     * Find the numbers with n digits that if rotated 180 degrees, it will be
     * the same number. 69 or 181
     * Since only 0,1,9,8,6 make a valid number when flipped, those are our only candidates
     * when we iterate and try to add to our final solution. Keep calling the recursive function
     * until size of string match the n input.
     * isStrobo() check is done by iterating the string backwards and getting its flipped value.
     *
     * Time O(n^2) since we recursively make n calls until the string matches the input and for each possible
     *      solution, we iterate through to check if it's strobo.
     *      Optimal solution O(nlogn), we can make the isStrobo check in logn if we have two pointers
     *      and check from left and right side of the value.
     * Space O(1)
     */
    public static List<Integer> findStrobos(int n) {
        String[] candidates = new String[]{"0","1","6","8","9"};
        Map<String, String> flippedMap = new HashMap<>();
        flippedMap.put("0", "0");
        flippedMap.put("1", "1");
        flippedMap.put("6", "9");
        flippedMap.put("8", "8");
        flippedMap.put("9", "6");
        List<Integer> result = new LinkedList<>();
        for (int i = 1; i < flippedMap.size(); i++) {
            findStrobos(n, flippedMap, result, candidates[i]);
        }
        return result;
    }

    public static void findStrobos(int n, Map<String, String> flippedMap, List<Integer> result, String current) {
        if (current.length() == n) {
            if (isStrobo(current, flippedMap)) {
                result.add(Integer.parseInt(current));
            }
            return;
        }

        for (String candidate : flippedMap.keySet()) {
            findStrobos(n, flippedMap, result, current + candidate);
        }
    }

    public static boolean isStrobo(String val, Map<String, String> flippedMap) {
        // reverse order because we need to flip and reverse to rotate by 180.
        StringBuilder result = new StringBuilder();
        for (int i = val.length()-1; i >= 0; i--) {
            result.append(flippedMap.get(String.valueOf(val.charAt(i))));
        }

        // If the result equals the original val then it's strobogrammatic number.
        return val.equals(result.toString());
    }

    // Figure out how to create sub matrices
    // if m = 3, then we make sub matrices of 2, and then 1 (m-1 at each recursion)
    // Given a sub matrix, we check to see if the sum equals target (backtracking)
    public int numSubMatrices(int[][] matrix, int rowStart, int rowEnd, int colStart, int colEnd, int target, int size) {
        int currentSum = 0;
        for (int i = rowStart; i < rowEnd; i++) {
            for (int j = colStart; j < colEnd; j++) {
                currentSum += matrix[i][j];
            }
        }
        if (size == 0) {
//            System.out.println(rowStart + "," + rowEnd + " | " + colStart + "," + colEnd);
        }
        if (currentSum == target) {
            System.out.println(rowStart + "," + rowEnd + " | " + colStart + "," + colEnd);
            return 1;
        }
        // Find the sub matrices
        int result = 0;
        for (int i = rowStart; i < rowEnd; i++) {
            for (int j = colStart; j < colEnd; j++) {
                int tempRowEnd = i+size;
                int tempColEnd = j+size;
                if (tempColEnd <= colEnd && tempRowEnd <= rowEnd) {
                    result += numSubMatrices(matrix, i, i + size, j, j + size, target, size - 1);
                }
            }
        }
        return result;
    }
}
