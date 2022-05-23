package backtracking;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Backtracking problems require you to add the possible solution to your current solution list and if
 * it doesn't result in the final solution, we need to revert (backtrack) that added solution from current
 * solution list. This way we can try another possible solution.
 *
 * Template for back tracking problems
 *
 *    void search(T state, M solutions) {
 *         if (isValidState(state)) {
 *             solutions.add(new T(state));
 *             // return if we need one solution
 *         }
 *
 *         for (N candidate : getCandidates(state)) {
 *             state.add(candidate);
 *             search(state, solutions);
 *             state.remove(candidate);
 *         }
 *     }
 *
 *     M solve() {
 *         M solutions = new M();
 *         T state = new T();
 *         search(state, solutions);
 *         return solutions
 *     }
 */
public class BackTrackingProblems {

    public static void main(String[] args) {
        List<List<Character>> result = generateParentheses(6);
        System.out.println(result.size());
    }

    /**
     * Generate valid parens given the number of parens we can use. Backtracking problem with 2 candidates ( and ).
     * Time O(2^n) since we have 2 candidates at each step
     * Space O(n*2^n) since we have call stack plus the running solution
     */
    public static List<List<Character>> generateParentheses(int n) {
        // two possible selections ( or )
        // base case is the current running solution's size == n && running sum == 0
        // only add to current running solution if we meet these conditions
        //          - If (, then we always add (increment sum)
        //          - If ), then we can only add if running sum > 0 (decrement sum)

        List<List<Character>> solutions = new LinkedList<>();
        generateParentheses(n, solutions, new LinkedList<>(), 0);
        return solutions;
    }

    public static void generateParentheses(int n, List<List<Character>> solutions, List<Character> current, int sum) {
        if (current.size() == n) {
            if (sum == 0) {
                solutions.add(current);
            }
            return;
        }

        current.add('(');
        generateParentheses(n, solutions, new LinkedList<>(current), sum+1);
        current.remove(current.size()-1);

        if (sum > 0) {
            current.add(')');
            generateParentheses(n, solutions, new LinkedList<>(current), sum-1);
            current.remove(current.size()-1);
        }
    }

    /**
     * Given possible steps at a time you can take, how many unique ways can you climb n steps?
     * Tree and backtracking problem. We just count of the current steps taken to see if it matches
     * the solution. If it's same, we add to solution; If it's greater, we return from recursive call;
     * If it's less, we continue on with the recursive call.
     */
    public static List<List<Integer>> uniqueClimbs(int n, Set<Integer> possibleSteps) {
        List<List<Integer>> solutions = new LinkedList<>();
        dfs(new LinkedList<>(), solutions, n, possibleSteps);
        return solutions;
    }

    public static void dfs(List<Integer> current, List<List<Integer>> solutions, int n, Set<Integer> possibleSteps) {
        int sum = sum(current);
        if (sum == n) {
            solutions.add(current);
            return;
        } else if (sum > n) {
            return;
        }

        for (Integer step : possibleSteps) {
            current.add(step);
            dfs(new LinkedList<>(current), solutions, n, possibleSteps);
            current.remove(current.size()-1);
        }
    }

    public static int sum(List<Integer> current) {
        int sum = 0;
        for (Integer val : current) {
            sum += val;
        }
        return sum;
    }

    /**
     * Find all the subsets from the values in the array.
     * We iterate through each of different subset sizes, 0 to 3, then if our current solution
     * matches the iteration size and doesn't already exist in our final solution, then we add
     * it to the final solution. Backtracking comes in play when we have added our value in the
     * current solution but still hasn't hit our size limit, we recursively call the method again.
     * When we return back from the recursive call, we need to backtrack to the initial state.
     */
    public static List<List<Integer>> subsets(int[] nums) {
        List<Set<Integer>> subsets = new LinkedList<>();
        for (int i = 0; i <= nums.length; i++) {
            solve(nums, i, new HashSet<>(), subsets);
        }

        List<List<Integer>> result = new LinkedList<>();
        for (Set<Integer> subset : subsets) {
            result.add(new LinkedList<>(subset));
        }
        return result;
    }

    public static void solve(int[] nums, int index, Set<Integer> current, List<Set<Integer>> subsets) {
        if (current.size() == index && !subsets.contains(current)) {
            subsets.add(current);
            return;
        }

        // Iterate through candidates
        for (int i = 0; i < nums.length; i++) {
            if (!current.contains(nums[i])) {
                current.add(nums[i]);
                solve(nums, index, new HashSet<>(current), subsets);
                current.remove(nums[i]);
            }
        }
    }

    /**
     * Hard backtracking problem that requires you to create a valid sudoku board.
     */
    public static void solveSudoku(Integer[][] board) {
//        Integer[][] board = new Integer[][] {
//                {5,3,null,null,7,null,null,null,null},
//                {6,null,null,1,9,5,null,null,null},
//                {null,9,8,null,null,null,null,6,null},
//                {8,null,null,null,6,null,null,null,3},
//                {4,null,null,8,null,3,null,null,1},
//                {7,null,null,null,2,null,null,null,6},
//                {null,6,null,null,null,null,2,8,null},
//                {null,null,null,4,1,9,null,null,5},
//                {null,null,null,null,8,null,null,7,9}
//        };

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == null) {
                    if (fillValidBoard(board, i, j)) {
                        return;
                    }
                }
            }
        }
    }

    public static boolean fillValidBoard(Integer[][] board, int rowIndex, int colIndex) {
        if (isBoardFilled(board)) {
            if (isValidSudoku(board) && isBoardFilled(board)) {
                return true;
            }
        }
//        Integer[][] newBoard = new Integer[board.length][board[0].length];
//        for (int i = 0; i < board.length; i++) {
//            for (int j = 0; j < board[i].length; j++) {
//                newBoard[i][j] = board[i][j];
//            }
//        }

        for (int digit = 1; digit <= 9; digit++) {
            board[rowIndex][colIndex] = digit;
            if (isValidSudoku(board)) {
                for (int i = 0; i < board.length; i++) {
                    for (int j = 0; j < board[i].length; j++) {
                        if (board[i][j] == null) {
                            if (fillValidBoard(board, i, j)) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        board[rowIndex][colIndex] = null;
        return false;
    }

    public static boolean isBoardFilled(Integer[][] board) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == null) {
                    return false;
                }
            }
        }
        return true;
    }

    public static boolean isValidSudoku(Integer[][] board) {
        for (int i = 0; i < board.length; i++) {
            if (!checkValidColumn(board, i) || !checkValidRow(board, i)) {
                return false;
            }
        }

        for (int i = 0; i < board.length; i = i + 3) {
            for (int j = 0; j < board[i].length; j = j + 3) {
                // 0,0 -> 0,3 -> 0,6
                // 3,0 -> 3,3 -> 3,6
                // 6,0 -> 6,3 -> 6,6
                if (!checkValidBox(board, i, j)) {
                    return false;
                }
            }
        }
        return true;
    }

    public static boolean checkValidColumn(Integer[][] board, int columnIndex) {
        Set<Integer> digits = new HashSet<>();
        digits.addAll(Arrays.asList(1,2,3,4,5,6,7,8,9));
        for (Integer[] integers : board) {
            if (digits.contains(integers[columnIndex])) {
                digits.remove(integers[columnIndex]);
            } else if (integers[columnIndex] != null) {
                return false;
            }
        }
        return true;
    }

    public static boolean checkValidRow(Integer[][] board, int rowIndex) {
        Set<Integer> digits = new HashSet<>();
        digits.addAll(Arrays.asList(1,2,3,4,5,6,7,8,9));
        for (int i = 0; i < board.length; i++) {
            if (digits.contains(board[rowIndex][i])) {
                digits.remove(board[rowIndex][i]);
            } else if (board[rowIndex][i] != null) {
                return false;
            }
        }
        return true;
    }

    /**
     * More Optimal Solution
     * To check if each of the box within board is valid, we can do a clever math to find the
     * row and column index of the box. Box index = Math.floor(board index {1 - 9} divided by 3).
     * That should give us the row and column index of the box within the board.
     * Given that, we can create a HashMap<String(boxRow, boxColumn), Set<Integer>> to check if
     * value within the box already exits or not.
     */
    public static boolean checkValidBox(Integer[][] board, int rowIndex, int columnIndex) {
        Set<Integer> digits = new HashSet<>();
        digits.addAll(Arrays.asList(1,2,3,4,5,6,7,8,9));
        for (int i = rowIndex; i < rowIndex+3; i++) {
            for (int j = columnIndex; j < columnIndex+3; j++) {
                if (digits.contains(board[i][j])) {
                    digits.remove(board[i][j]);
                } else if (board[i][j] != null) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Solve N Queens is asking given n number of queens on an n X n board, how many distinct
     * queens positions are there where queens can't attack each other.
     *
     * 1. Define the state of the board and current list of solutions.
     * 2. Iterate through each of the indexes on the board to see if a solution exists.
     * 3. Make a recursive call to check if a solution exists.
     */
    public static List<List<String>> solveNQueens(int numberOfQueens) {
        List<List<String>> solutions = new LinkedList<>();

        for (int i = 0 ; i < numberOfQueens; i++) {
            for (int j = 0; j < numberOfQueens; j++) {
                List<String> solution = new LinkedList<>();
                solution.add(i + "," + j);
                int[][] board = new int[numberOfQueens][numberOfQueens];
                board[i][j] = 1;
                if (solutionExists(board, solutions, solution, numberOfQueens, i, j)) {
                    solutions.add(solution);
                } else {
                    solution.remove(solution.size()-1);
                }
            }
        }
        return solutions;
    }

    /**
     * Main backtracking program. If the solution doesn't exist, we undo the marked queen position.
     * Also, we need to create a new board with the original state plus latest invalid states. Since, if the
     * solution doesn't exist, we can still backtrack and use the original board to find the next possible solution.
     */
    public static boolean solutionExists(int[][] board, List<List<String>> solutions, List<String> solution,
                                         int numberOfQueens, int rowIndex, int colIndex) {
        if (solution.size() == numberOfQueens) {
            return true;
        }
        int[][] newBoard = updateInvalidStates(board, rowIndex, colIndex);
        for (int i = 0; i < newBoard.length; i++) {
            for (int j = 0; j < newBoard.length; j++) {
                if (newBoard[i][j] == 0 && notInSolution(i, j, solutions)) {
                    solution.add(i + "," + j);
                    newBoard[i][j] = 1;
                    if (solutionExists(newBoard, solutions, solution, numberOfQueens, i, j)
                            && solution.size() == numberOfQueens) {
                        return true;
                    }
                    solution.remove(solution.size()-1);
                    newBoard[i][j] = 0;
                }
            }
        }
        return false;
    }

    /**
     * Set the board to 1s for all the path that the queen can take from rowIndex and colIndex.
     */
    public static int[][] updateInvalidStates(int[][] board, int rowIndex, int colIndex) {
        int[][] newBoard = new int[board.length][board.length];

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                newBoard[i][j] = board[i][j];
            }
        }

        for (int i = 0; i < newBoard.length; i++) {
            for (int j = 0; j < newBoard[i].length; j++) {
                // up and down
                if (i == rowIndex || j == colIndex) {
                    newBoard[i][j] = 1;
                }
                // Diagonals
                if (i-j == rowIndex-colIndex || i+j == rowIndex+colIndex) {
                    newBoard[i][j] = 1;
                }
            }
        }
        return newBoard;
    }

    /**
     * To prevent duplicates of the solution.
     */
    public static boolean notInSolution(int rowIndex, int colIndex, List<List<String>> solutions) {
        for (List<String> solution : solutions) {
            if (solution.contains(rowIndex + "," + colIndex)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Given a validation criteria (rotate int to a new valid int), we count how many of those exists between
     * 1 and n. We need to iterate each of the possible digits like candidates from a node until we reach
     * n. Once we have crafted the list of digits that might be our solution, we run it against our validation
     * function. If successful, we can add it to our result count, otherwise we don't.
     *
     * Time O(nlogn) & Space O(m), where n is input and m is the digit counts within n.
     */
    public static int numOfConfusingNumbers(int n) {
        Set<Integer> confusingNumbers = new HashSet<>();
        confusingNumbers.add(0);
        confusingNumbers.add(1);
        confusingNumbers.add(6);
        confusingNumbers.add(8);
        confusingNumbers.add(9);
        return dfs(n, 0, 1, confusingNumbers);
    }

    public static int dfs(int n, int startVal, int multiplication, Set<Integer> confusingNumbers) {
        int result = 0;
        for (Integer confusingNum : confusingNumbers) {
            int newVal;
            if (startVal == 0) {
                newVal = confusingNum;
            } else {
                newVal = (startVal * multiplication) + confusingNum;
            }
            if (newVal <= n && newVal > 0) {
                System.out.println(newVal);
                if (isConfusingNum(newVal)) {
                    result++;
                }
                result += dfs(n, newVal, multiplication*10, confusingNumbers);
            }
        }
        return result;
    }

    public static boolean isConfusingNum(int val) {
        Map<Integer, Integer> confusingNumbersMap = new HashMap<>();
        confusingNumbersMap.put(0, 0);
        confusingNumbersMap.put(1, 1);
        confusingNumbersMap.put(6, 9);
        confusingNumbersMap.put(8, 8);
        confusingNumbersMap.put(9, 6);

        StringBuilder result = new StringBuilder();
        int temp = val;
        while (temp > 0) {
            int digit = temp % 10;
            result.append(confusingNumbersMap.get(digit));
            temp = temp/10;
        }
        return val != Integer.parseInt(result.toString());
    }

    /**
     * Given a wordList, we need to find the shortest path from begin to end word, where path is made up of
     * words that are one character apart from the previous character. hit -> hot -> dog -> cog
     *
     * We can generateAdjacentListMap() by creating a Map of wildcard patterns as the key and set the value
     * to the list of words to match the pattern. <*it, [hit]> , <h*t, [hit, hot]>, etc.
     *
     * Once we have the adjacentListMap, then it's just a typical backtracking problem. We just traverse through
     * each candidate and check when the beginWord matches the endWord and compare the current size of path to
     * the running min size of the path.
     */
    public static List<String> wordLadder(String beginWord, String endWord, List<String> wordList) {
        Map<String, List<String>> adjacentListMap = generateAdjacentListMap(wordList);
        List<String> result = new LinkedList<>();
        List<String> current = new LinkedList<>();
        current.add(beginWord);
        for (int i = 0; i < beginWord.length(); i++) {
            for (char c = 'a'; c <= 'z'; c++) {
                // This implementation would be easier with the new definition of the adjacent list.
                String nextVal = getNextVal(beginWord, i, c);
                if (adjacentListMap.containsKey(nextVal)) {
                    current.add(nextVal);
                    wordLadderUtil(nextVal, endWord, adjacentListMap, new LinkedList<>(current), result);
                    current.remove(current.size()-1);
                }
            }
        }
        return result;
    }

    private static void wordLadderUtil(String beginWord, String endWord, Map<String, List<String>> adjacentListMap,
                                      List<String> current, List<String> result) {
        if (beginWord.equals(endWord) && (result.isEmpty() || current.size() < result.size())) {
            result = current;
            return;
        }

        for (String adjacent : adjacentListMap.get(beginWord)) {
            if (current.contains(adjacent)) {
                continue;
            }
            current.add(adjacent);
            wordLadderUtil(adjacent, endWord, adjacentListMap, new LinkedList<>(current), result);
            current.remove(current.size()-1);
        }
    }

    private static String getNextVal(String word, int index, char replacingChar) {
        return null;
    }

    private static Map<String, List<String>> generateAdjacentListMap(List<String> wordList) {
        return null;
    }
}
