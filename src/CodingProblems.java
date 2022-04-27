import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class CodingProblems {
    public static void main(String[] args) {
        char[][] input = new char[][]{{'X','X','X','X'},{'X','O','O','X'},{'X','X','O','X'},{'X','O','X','X'}};
        solve(input);
        System.out.println(input[1][1]);
    }

    /**
     * O(3^n*n^2) For each cell, we check 3 sides assuming we are coming from another cell.
     * Probably optimize it by having a visited matrix.
     *
     * O(n*m) Optimal solution is to find all the 'O' that are on the boundary of the matrix and set to 'T'. We know for sure
     * those aren't 'X'. Then we traverse (DFS/BFS) from those 'O' and also set those cells to 'T'.
     * Finally, set everything to 'X' and set 'T' to 'O'.
     */
    public static void solve(char[][] board) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == 'O') {
                    computeSurrounded(board, i, j, null);
                }
            }
        }
    }

    public static boolean computeSurrounded(char[][] board, int i, int j, String prevSide) {
        boolean left = false, right = false, top = false, bottom = false;

        if ("Left".equals(prevSide)) {
            if (i+1 < board.length) {
                if (board[i+1][j] == 'X' || computeSurrounded(board, i+1, j, "Top")) {
                    bottom = true;
                }
            }
            if (i-1 >= 0) {
                if (board[i-1][j] == 'X' || computeSurrounded(board, i-1, j, "Bottom")) {
                    top = true;
                }
            }
            left = true;
            if (j+1 < board[i].length) {
                if (board[j+1][j] == 'X' || computeSurrounded(board, i, j+1, "Left")) {
                    right = true;
                }
            }
        } else if ("Right".equals(prevSide)) {
            if (i+1 < board.length) {
                if (board[i+1][j] == 'X' || computeSurrounded(board, i+1, j, "Top")) {
                    bottom = true;
                }
            }
            if (i-1 >= 0) {
                if (board[i-1][j] == 'X' || computeSurrounded(board, i-1, j, "Bottom")) {
                    top = true;
                }
            }
            if (j-1 >= 0) {
                if (board[j-1][j] == 'X' || computeSurrounded(board, i, j-1, "Right")) {
                    left = true;
                }
            }
            right = true;
        } else if ("Top".equals(prevSide)) {
            if (i+1 < board.length) {
                if (board[i+1][j] == 'X' || computeSurrounded(board, i+1, j, "Top")) {
                    bottom = true;
                }
            }
            top = true;
            if (j-1 >= 0) {
                if (board[j-1][j] == 'X' || computeSurrounded(board, i, j-1, "Right")) {
                    left = true;
                }
            }
            if (j+1 < board[i].length) {
                if (board[j+1][j] == 'X' || computeSurrounded(board, i, j+1, "Left")) {
                    right = true;
                }
            }
        } else if ("Bottom".equals(prevSide)) {
            bottom = true;
            if (i-1 >= 0) {
                if (board[i-1][j] == 'X' || computeSurrounded(board, i-1, j, "Bottom")) {
                    top = true;
                }
            }
            if (j-1 >= 0) {
                if (board[j-1][j] == 'X' || computeSurrounded(board, i, j-1, "Right")) {
                    left = true;
                }
            }
            if (j+1 < board[i].length) {
                if (board[j+1][j] == 'X' || computeSurrounded(board, i, j+1, "Left")) {
                    right = true;
                }
            }
        } else {
            if (i+1 < board.length) {
                if (board[i+1][j] == 'X' || computeSurrounded(board, i+1, j, "Top")) {
                    bottom = true;
                }
            }
            if (i-1 >= 0) {
                if (board[i-1][j] == 'X' || computeSurrounded(board, i-1, j, "Bottom")) {
                    top = true;
                }
            }
            if (j-1 >= 0) {
                if (board[i][j-1] == 'X' || computeSurrounded(board, i, j-1, "Right")) {
                    left = true;
                }
            }
            if (j+1 < board[i].length) {
                if (board[i][j+1] == 'X' || computeSurrounded(board, i, j+1, "Left")) {
                    right = true;
                }
            }
        }

        if (left && right && top && bottom) {
            board[i][j] = 'X';
            return true;
        }
        return false;
    }

    /**
     * O(n) Use a HashSet to do a O(1) lookup to check if num exists in nums.
     *
     * Given 6,5,4,3,2,1 only the value 1 is valid for the while-loop, rest have its value - 1 in the set.
     * Given 2,5,6,7,9,11 all values don't have its value - 1 in the set but its value + 1 isn't in the set,
     *      so the while-loop isn't triggered.
     */
    public static int longestConsecutive(int[] nums) {
        Set<Integer> numsSet = new HashSet<>();
        for (int num : nums) {
            numsSet.add(num);
        }
        int longestStreak = 0;

        for (int num : nums) {
            if (!numsSet.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;

                while (numsSet.contains(currentNum + 1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }
                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }
        return longestStreak;
    }

    /**
     * O(n) we only iterate through the prices list once.
     */
    public static int maxProfit(int[] prices) {
        // [7,1,5,3,6,4] = 7
        // Set the smallest number as long as it's in decreasing order as buy day
        // Set the largest number as long as it's in increasing order as sell day
        // Compute the differences and get profit.
        boolean buy = true;
        int currentProfit = 0;
        int previousMinOrMax = Integer.MAX_VALUE;
        for (int i = 0; i <= prices.length; i++) {
            if (buy && i <= prices.length-1 && previousMinOrMax > prices[i]) {
                previousMinOrMax = prices[i];
            } else if (buy && i <= prices.length-1) {
                buy = false;
                currentProfit = currentProfit - previousMinOrMax;
                previousMinOrMax = prices[i];
            } else if (!buy && i <= prices.length-1 && previousMinOrMax < prices[i]) {
                previousMinOrMax = prices[i];
            } else if (!buy) {
                buy = true;
                currentProfit = currentProfit + previousMinOrMax;
                if (i <= prices.length-1) {
                    previousMinOrMax = prices[i];
                }
            }
        }
        return currentProfit;
    }

    /**
     * O(n) since the recursive call is outside for-loop, and we only access each node once.
     * Add each of the right sibling node as next of the left sibling node. Right most node's next will be null.
     */
    public static TreeNode connect(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);

        connectUtil(queue);
        return root;
    }

    public static void connectUtil(Queue<TreeNode> queue) {
        int queueSize = queue.size();

        TreeNode previous = null;
        for (int i = 0; i < queueSize; i++) {
            TreeNode current = queue.poll();
            if (previous != null) {
                previous.next = current;
            }
            previous = current;

            if (current.left != null) {
                queue.add(current.left);
                queue.add(current.right);
            }
        }
        if (previous != null) {
            previous.next = null;
        }

        if (!queue.isEmpty()) {
            connectUtil(queue);
        }
    }

    /**
     *                  1
     *          2               3
     *      4       5
     * Preorder (top down & left child first) : 1 2 4 5 3
     * Inorder (bottom up & left child first) : 4 2 5 1 3
     * Postorder (bottom up & children first) : 4 5 2 3 1
     *
     * Preorder traversal follows Root -> Left -> Right. Hence, given the preorder array,
     * we have easy access to the root which is preorder[0]. Root for the children nodes is
     * determined by incrementing preorderIndex global variable.
     *
     * Inorder traversal follows Left -> Root -> Right. Hence, if we know the position of root,
     * we can recursively split the entire array into two subtrees.
     */
    static int preorderIndex;
    public static TreeNode buildTree2(int[] preorder, int[] inorder) {
        preorderIndex = 0;
        return buildTree(preorder, inorder, 0, inorder.length);
    }

    public static TreeNode buildTree(int[] preorder, int[] inorder, int leftBound, int rightBound) {
        if (preorderIndex >= preorder.length) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preorderIndex++]);
        for (int i = leftBound; i < rightBound; i++) {
            if (inorder[i] == root.val) {
                root.left = buildTree(preorder, inorder, leftBound, i-1);
                root.right = buildTree(preorder, inorder, i+1, rightBound);
                break;
            }
        }
        return root;
    }

    /**
     * O(n) since even with recursive call, it's done outside the for-loop.
     * Each node is handled only once.
     */
    public static List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new LinkedList<>();
        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);

        zigzagLevelOrder(queue, result);
        return result;
    }

    public static void levelOrder(Queue<TreeNode> queue, List<List<Integer>> result) {
        List<Integer> currentLevel = new LinkedList<>();
        int currentSize = queue.size();
        for (int i = 0; i < currentSize; i++) {
            TreeNode current = queue.poll();
            currentLevel.add(current.val);
            if (current.left != null) {
                queue.add(current.left);
            }
            if (current.right != null) {
                queue.add(current.right);
            }
        }

        result.add(currentLevel);
        if (!queue.isEmpty()) {
            levelOrder(queue, result);
        }
    }

    public static void zigzagLevelOrder(Queue<TreeNode> queue, List<List<Integer>> result) {
        List<Integer> currentLevel = new LinkedList<>();
        int currentSize = queue.size();
        for (int i = 0; i < currentSize; i++) {
            TreeNode current = queue.poll();
            currentLevel.add(current.val);
            if (current.right != null) {
                queue.add(current.right);
            }
            if (current.left != null) {
                queue.add(current.left);
            }
        }

        result.add(currentLevel);
        if (!queue.isEmpty()) {
            levelOrder(queue, result);
        }
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
     * O(n*m) where n is the length of weights and m is the maxWeight.
     *
     * If j_value (weight of the current item) > j
     *      subSolutions[i-1][j]
     * Otherwise
     *      Max( subSolutions[i-1][j] , subSolutions[i-1][j-j_value] )
     */
    public static int knapsack(List<Integer> weights, List<Integer> values, int maxWeight) {
        List<Integer> weightsNew = new ArrayList<>(Collections.singletonList(0));
        weightsNew.addAll(weights);
        List<Integer> valuesNew = new ArrayList<>(Collections.singletonList(0));
        valuesNew.addAll(values);

        int[][] subSolutions = new int[weightsNew.size()][maxWeight+1];
        for (int i = 1; i < subSolutions.length; i++) {
            for (int j = 1; j < subSolutions[i].length; j++) {
                if (weightsNew.get(i) > j) {
                    subSolutions[i][j] = subSolutions[i-1][j];
                } else {
                    subSolutions[i][j] = Math.max(subSolutions[i-1][j], subSolutions[i-1][j-weightsNew.get(i)] + valuesNew.get(i));
                }
            }
        }
        return subSolutions[weightsNew.size()-1][maxWeight];
    }

    /**
     * O(n) Dynamic programming solution.
     * dp[n] = dp[n-1] + dp[n-2] , where dp[0] = 1 and dp[1] = 1 or 0 depending on if the char is 0.
     *
     * A problem is a dynamic programming problem if it satisfies two conditions:
     * 1. The problem can be divided into sub problems, and its optimal solution can be constructed from optimal
     *      solutions of the sub problems. In academic terms, this is called optimal substructure.
     * 2. The sub problems from 1) overlap and results of the sub problems can be cached and reused.
     */
    public static int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }

        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;

        for (int i = 2; i <= s.length(); i++) {
            int first = Integer.parseInt(s.substring(i - 1, i)); // 2
            int second = Integer.parseInt(s.substring(i - 2, i));// 02

            if (first >=1 && first <=9) {
                dp[i] += dp[i-1];
            }
            if (second >=10 && second <=26) {
                dp[i] += dp[i-2];
            }
        }
        return dp[s.length()];
    }

    /**
     * O(n*(3^l) ) where n is the total number of cells in the grid and l is the length of the given word to be searched.
     * 3^l since we only have three choices at a given cell, since one of the adjacent cell is where we just came from.
     *
     * Set visited to non alphabet and then set it back if substring isn't valid.
     */
    public static boolean existWord(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == word.charAt(0)) {
                    char temp = board[i][j];
                    board[i][j] = '#';
                    if (existUtil(board, word, i, j, 1)) {
                        return true;
                    }
                    board[i][j] = temp;
                }
            }
        }
        return false;
    }

    public static boolean existUtil(char[][]board, String word, int i, int j, int wordIndex) {
        if (wordIndex == word.length()) {
            return true;
        }

        // check each of the adjacent cells
        if (i-1 >= 0) {
            if (board[i-1][j] == word.charAt(wordIndex)) {
                char temp = board[i-1][j];
                board[i-1][j] = '#';
                if (existUtil(board, word, i-1, j, wordIndex+1)) {
                    return true;
                }
                board[i-1][j] = temp;
            }
        }
        if (i+1 < board.length) {
            if (board[i+1][j] == word.charAt(wordIndex)) {
                char temp = board[i+1][j];
                board[i+1][j] = '#';
                if (existUtil(board, word, i+1, j, wordIndex+1)) {
                    return true;
                }
                board[i+1][j] = temp;
            }
        }
        if (j-1 >= 0) {
            if (board[i][j-1] == word.charAt(wordIndex)) {
                char temp = board[i][j-1];
                board[i][j-1] = '#';
                if (existUtil(board, word, i, j-1, wordIndex+1)) {
                    return true;
                }
                board[i][j-1] = temp;
            }
        }
        if (j+1 < board[i].length) {
            if (board[i][j+1] == word.charAt(wordIndex)) {
                char temp = board[i][j+1];
                board[i][j+1] = '#';
                if (existUtil(board, word, i, j+1, wordIndex+1)) {
                    return true;
                }
                board[i][j+1] = temp;
            }
        }
        return false;
    }

    /**
     * O(n2^n) Since we iterate from left to right, we never have to worry about
     * duplicates being added to result. Iteration ends when we get to final index.
     */
    public static List<List<Integer>> subsetsNonUnique(int[] nums) {
        //   1      2       3
        //  2 3    3
        // 3
        List<List<Integer>> output = new ArrayList<>();
        for (int k = 0; k <= nums.length; k++) {
            backtrack(0, new ArrayList<>(), nums, k, output);
        }
        return output;
    }

    public static void backtrack(int first, List<Integer> currentList, int[] nums, int currentSize, List<List<Integer>> result) {
        if (currentList.size() == currentSize) {
            result.add(new ArrayList<>(currentList));
            return;
        }

        for (int i = first; i < nums.length; i++) {
            // add i into the current combination
            currentList.add(nums[i]);
            // use next integers to complete the combination
            backtrack(i + 1, currentList, nums, currentSize, result);
            // backtrack
            currentList.remove(currentList.size() - 1);
        }
    }

    /**
     * O(nlogn) Pick a pivot and put it in the right index. Then recursively call sort on the left and right
     * side of partition.
     */
    public static void quicksort(int[] nums, int start, int end) {
        if (start < end) {
            int pIndex = partition(nums, start, end);
            quicksort(nums, start, pIndex-1);
            quicksort(nums, pIndex+1, end);
        }
    }

    // 50, 90, 80, 10, 70 -> 50, 10, 70, 90, 80
    public static int partition(int[] nums, int start, int end) {
        int pValue = nums[end];
        int i = start-1; // Last index of the array where value is less than our pivot
        int j = start; // current iteration of the array.
        while (j < end) {
            if (nums[j] < pValue) {
                i++;
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
            }
            j++;
        }

        // Last index of the array where value is less than our pivot + 1 is the start of
        // values greater than our pivot. Replace that with our pivot.
        nums[end] = nums[i+1];
        nums[i+1] = pValue;
        return i+1;
    }

    /**
     * O(n) similar to pivoting in quicksort. We keep track of low-index that tracks all the 0 values
     * starting at index 0. We keep track of high-index that tracks all the 2 values ending at index nums.length-1.
     * mid-index is used to iterate through the array.
     */
    public static void sortColors(int[] nums) {
        int lowIndex = 0;
        int midIndex = 0;
        int highIndex = nums.length-1;

        while (midIndex < highIndex) {
            if (nums[midIndex] == 0) {
                int temp = nums[lowIndex];
                nums[lowIndex] = nums[midIndex];
                nums[midIndex] = temp;

                lowIndex++;
                midIndex++;
            } else if (nums[midIndex] == 1) {
                midIndex++;
            } else {
                int temp = nums[highIndex];
                nums[highIndex] = nums[midIndex];
                nums[midIndex] = temp;

                highIndex--;
            }
        }
    }

    /**
     * Time: O(M*N) but the best case is much better due to our cache.
     * Space: O(M+N) since we are keeping track of zero rows and columns.
     *  To improve the space complexity, we can just set the first row and column of the matrix
     *  to zero if any that row or column contains a zero. Time will be same but space will improve
     *  to O(1) since we are updating the matrix itself.
     */
    public static void setZeroes(int[][] matrix) {
        boolean[] isZeroRows = new boolean[matrix.length];
        boolean[] isZeroColumns = new boolean[matrix[0].length];

        for (int i = 0; i < isZeroRows.length; i++) {
            if (!isZeroRows[i]) {
                for (int j = 0; j < isZeroColumns.length; j++) {
                    if (!isZeroColumns[j]) {
                        if (matrix[i][j] == 0) {
                            isZeroRows[i] = true;
                            isZeroColumns[j] = true;
                            setRowToZero(matrix, i);
                            setColumnToZero(matrix, i);
                        }
                    }
                }
            }
        }
    }

    public static void setRowToZero(int[][] matrix, int rowIndex) {
        Arrays.fill(matrix[rowIndex], 0);
    }

    public static void setColumnToZero(int[][] matrix, int columnIndex) {
        for (int i = 0; i < matrix.length; i++) {
            matrix[i][columnIndex] = 0;
        }
    }

    /**
     * O(N*M), where N and M are used to represent the number of rows and columns in the grid.
     */
    public static int uniquePathsMemorized(int m, int n) {
        int[][] dp = new int[m+1][n+1];
        return uniquePathsUtil(m, n, dp);
    }

    public static int uniquePathsUtil(int m, int n, int[][] dp) {
        if(m == 1 || n == 1) {
            return 1;
        }
        if(dp[m][n] != 0) {
            return dp[m][n];
        }
        return dp[m][n] = uniquePathsUtil(m-1, n, dp) + uniquePathsUtil(m, n-1, dp);
    }

    public static int uniquePaths(int m, int n) {
        return dfs(m, n, 0, 0, 0);
    }

    public static int dfs(int m, int n, int i, int j, int result) {
        if (i == m-1 && j == n-1) {
            return result + 1;
        }

        if (i < m-1) {
            result = dfs(m, n, i+1, j, result);
        }

        if (j < n-1) {
            result = dfs(m, n, i, j+1, result);
        }

        return result;
    }

    /**
     * Time: O(n) & Space: O(1)
     * Using XOR (exclusive or), we can find the integer without a match. Since binary values
     * for each integer will be canceled with its matching counterpart. We will be left with
     * the unique value.
     * 1 ^ 3 = 2
     * 0001 ^ 0011 = 0010
     */
    public static int uniqueInt(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }

    /**
     * O(n) we only iterate through the array one time and use two pointers to find the left and right
     * bound of the interval.
     */
    public static int[][] merge(int[][] intervals) {
        ArrayList<int[]> mergedIntervals = new ArrayList<>();

        int i = 0;
        int j = 1;
        while (i < intervals.length-1 && j < intervals.length) {
            if (intervals[j][0] <= intervals[i][1]) {
                j++;
            } else if (i == j-1) {
                int[] newInterval = new int[2];
                newInterval[0] = intervals[i][0];
                newInterval[1] = intervals[i][1];

                mergedIntervals.add(newInterval);
                i=j;
                j++;
            } else if (j-1 > i) {
                int[] newInterval = new int[2];
                newInterval[0] = intervals[i][0];
                newInterval[1] = intervals[j-1][1];

                mergedIntervals.add(newInterval);
                i=j;
                j++;
            }
        }

        if (i == j-1) {
            int[] newInterval = new int[2];
            newInterval[0] = intervals[i][0];
            newInterval[1] = intervals[i][1];

            mergedIntervals.add(newInterval);
        } else if (j-1 > i) {
            int[] newInterval = new int[2];
            newInterval[0] = intervals[i][0];
            newInterval[1] = intervals[j-1][1];

            mergedIntervals.add(newInterval);
        }

        int[][] result = new int[mergedIntervals.size()][2];
        return mergedIntervals.toArray(result);
    }

    /**
     * O(n) We start at last index and go backwards to see if we can reach the starting index.
     */
    public static boolean canJumpGreedy(int[] nums) {
        int prevIndex = nums.length-1;
        for (int i = nums.length-1; i >= 0; i--) {
            if (nums[i] + i >= prevIndex) {
                prevIndex = i;
            }
        }
        return prevIndex == 0;
    }

    /**
     * O(n*m) where n is the length of nums and m is the largest integer within the input array.
     */
    public static boolean canJump(int[] nums) {
        // If we create a tree, where children nodes are the possible jumps
        // we can make from the given position. Then if the last node is the
        // last index, then we return true.
        //        2
        //   3        1
        // 1 1 4      1
        // 1 4        4
        // 4
        return canJump(nums, 0);
    }

    public static boolean canJump(int[] nums, int currentIndex) {
        if (currentIndex == nums.length-1) {
            return true;
        } else if (currentIndex < nums.length-1) {
            int possibleJumps = nums[currentIndex];
            List<Integer> visitedIndexes = new LinkedList<>();
            for (int i = 1; i <= possibleJumps; i++) {
                visitedIndexes.add(currentIndex + i);
            }

            for (int index : visitedIndexes) {
                boolean result = canJump(nums, index);
                if (result) {
                    return result;
                }
            }
        }
        return false;
    }

    /**
     * O(m*n) We iterate through each of the elements in the matrix.
     */
    public static List<Integer> spiralOrderIterative(int[][] matrix) {
        int leftBound = 0;
        int rightBound = matrix[0].length-1;
        int topBound = 0;
        int bottomBound = matrix.length-1;

        List<Integer> result = new LinkedList<>();
        while (leftBound <= rightBound && topBound <= bottomBound) {
            //Traverse right and increment topBound
            for(int i = leftBound; i <= rightBound; i++) {
                result.add(matrix[topBound][i]);
            }
            topBound++;

            //Traverse down and decrement rightBound
            for(int i = topBound; i <= bottomBound; i++) {
                result.add(matrix[i][rightBound]);
            }
            rightBound--;

            //Make sure that row exists
            //And traverse left and decrement rowEnd
            if(topBound <= bottomBound) {
                for(int i = rightBound; i >= leftBound; i--) {
                    result.add(matrix[bottomBound][i]);
                }
            }
            bottomBound--;

            //Make sure that column exists
            //And traverse up and increment colBegin
            if(leftBound <= rightBound) {
                for(int i = bottomBound; i >= topBound; i--) {
                    result.add(matrix[i][leftBound]);
                }
            }
            leftBound++;
        }
        return result;
    }
    /**
     * O(nklogk), where n is the length of strs, and k is the maximum length of a string in strs.
     * The outer loop has complexity O(n) as we iterate through each string.
     * Then, we sort each string in O(klogk) time.
     */
    public static List<List<String>> groupAnagrams(String[] strs) {
        // If we order each of the strings, we can find out if those strings are anagram
        Map<String, List<Integer>> orderedStringToIndexes = new HashMap<>();

        for (int i = 0; i < strs.length; i++) {
            char[] stringChars = strs[i].toCharArray();
            Arrays.sort(stringChars);

            List<Integer> indexes = orderedStringToIndexes.getOrDefault(Arrays.toString(stringChars), new LinkedList<>());
            indexes.add(i);
            orderedStringToIndexes.put(Arrays.toString(stringChars), indexes);
        }

        List<List<String>> result = new LinkedList<>();
        for (Map.Entry<String, List<Integer>> entry : orderedStringToIndexes.entrySet()) {
            List<Integer> indexes = entry.getValue();
            List<String> anagrams = new LinkedList<>();
            for (Integer i : indexes) {
                anagrams.add(strs[i]);
            }
            result.add(anagrams);
        }
        return result;
    }

    /**
     * O(n) For each consecutive increasing subset, there is count total of i-1 + i-2 + .. + 0.
     * Where i is the count of elements in the subset.
     */
    public static int consecutiveIncreasing(int[] input) {
        int count = 0;
        int length = 1;
        for (int i = 1; i < input.length; i++) {
            if (input[i-1] < input[i]) {
                count += length;
                length++;
            } else {
                length = 1;
            }
        }
        return count;
    }

    /**
     * O(n) sumOfNNaturalNumbers is the key given num of digits in that are in decreasing subset.
     */
    public static long consecutiveDecreasingByOne(List<Integer> ratings) {
        long load = 0;
        long ans = 0;
        for(int i=0; i < ratings.size() ; i++) {
            if(load == 0) {
                load++;
            } else {
                if(ratings.get(i-1) == (ratings.get(i) + 1)) {
                    load++;
                } else {
                    ans += sumOfNNaturalNumbers(load);
                    load = 1;
                }
            }
        }
        ans += sumOfNNaturalNumbers(load);

        return ans;
    }

    private static long sumOfNNaturalNumbers(long n) {
        if (n%2 == 0)
            return ((n / 2) * (n + 1));
        else return n * ( (n + 1) / 2);
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

    /**
     * O(n*4^n) since 4 is max number of letters at a given digit. So, we can have 4 different tracks
     * for each digit in digits with length n. Keeping an index and incrementing the chars in digits during
     * a DFS() saves us from creating another for-loop.
     */
    public static List<String> letterCombinations(String digits) {
        String[] phoneDigits = new String[]{null, null, "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        //   a       b       c
        // d e f   d e f   d e f

        List<String> results = new LinkedList<>();
        dfs(digits, phoneDigits, results, new StringBuilder(), 0);
        return results;
    }

    public static void dfs(String digits, String[] phoneDigits, List<String> results, StringBuilder currentResult, int index) {
        if (currentResult.length() == digits.length()) {
            results.add(currentResult.toString());
            return;
        }
        char digitChar = digits.charAt(index);
        int digit = Integer.parseInt(String.valueOf(digitChar));

        String possibleLetters = phoneDigits[digit];
        for (int j = 0; j < possibleLetters.length(); j++) { // a,b,c -> d,e,f
            currentResult.append(possibleLetters.charAt(j)); // a,d
            dfs(digits, phoneDigits, results, currentResult, index+1);
            currentResult.deleteCharAt(currentResult.length()-1);
        }
    }

    /**
     * O(n2^n) If the subset is a palindrome, we add the subset into the currentList.
     * If the index to find the subset is greater than size of input, then we return the currentList.
     * Make sure to backtrack and remove the subset from the currentList after DFS().
     */
    public static List<List<String>> palindromePartition(String s) {
        // Find all SubSets
        // Check if each element in SubSet is a palindrome

        // Subsets are found using backtracking
        // if (current path size == iteration value) { add currentPath}

        // Possible SubSets
        //  a    a    b  -> a, a, b
        // a b  a b  a a -> aa, ab, aa, ab, ba, ba
        // b a  b a  a a -> aab, aba, aab, aba, baa, baa

        List<List<String>> result = new LinkedList<>();
        dfs(s, result, 0, new LinkedList<>());
        return result;
    }

    public static void dfs(String s, List<List<String>> result, int start, List<String> currentList) {
        // if index reach the max size of string then we have reached the leaf of the tree.
        if (start >= s.length()) {
            result.add(new ArrayList<>(currentList));
            return;
        }

        for (int end = start; end < s.length(); end++) {
            if (isPalindrome(s.substring(start, end + 1))) {
                // add current substring in the currentList
                currentList.add(s.substring(start, end + 1));
                dfs(s, result, end + 1, currentList);
                // backtrack and remove the current substring from currentList
                currentList.remove(currentList.size() - 1);
            }
        }
    }

    /**
     * O(n2^n) Use a HashSet to remove duplicates and use DFS to add the elements on your current iteration.
     */
    public static List<List<Integer>> subsets(int[] nums) {
        List<Integer> numsList = IntStream.of(nums)
                .boxed()
                .collect(Collectors.toList());
        Set<List<Integer>> subSets = new HashSet<>();
        subSets.add(numsList);

        dfs(numsList, 0, subSets);
        return new ArrayList<>(subSets);
    }

    public static void dfs(List<Integer> nums, int index, Set<List<Integer>> subSets) {
        if (index >= nums.size()) {
            subSets.add(new LinkedList<>());
            return;
        }

        for (int i = index; i < nums.size(); i++) {
            List<Integer> removedList = new LinkedList<>(nums);
            removedList.remove(i);
            subSets.add(removedList);
            dfs(removedList, i, subSets);
        }
    }

    /**
     * O(n^2) Typical backtracking problem except we create a hashmap that solves the duplicate issue.
     * One thing to note is to create a new list that's passed into the DFS() since all objects are
     * references.
     */
    public static List<List<Integer>> permuteUnique(int[] nums) {
        // Take one digit at a time and make dfs to each of the remaining digits
        // For the case of duplicates, we create a HashMap for each digit and its occurences

        // <1, 2> < 2, 1>
        //         root
        //      1       2
        //     1 2     1
        //    2   1   1

        Map<Integer, Integer> digitOccurrences = new HashMap<>();
        for (int num : nums) {
            digitOccurrences.put(num, digitOccurrences.getOrDefault(num, 0)+1);
        }

        List<List<Integer>> result = new LinkedList<>();
        dfs(digitOccurrences, nums.length, result, new LinkedList<>());
        return result;
    }

    public static void dfs(Map<Integer, Integer> digitOccurrences, int size, List<List<Integer>> result, List<Integer> currentResult) {
        if (currentResult.size() == size) {
            result.add(currentResult);
            return;
        }

        for (Map.Entry<Integer, Integer> digitOccurrence : digitOccurrences.entrySet()) {
            int currentOccurrences = digitOccurrence.getValue();
            if (currentOccurrences > 0) {
                currentResult.add(digitOccurrence.getKey());
                digitOccurrences.put(digitOccurrence.getKey(), currentOccurrences-1);
                dfs(digitOccurrences, size, result, new LinkedList<>(currentResult));
                currentResult.remove(digitOccurrence.getKey());
                digitOccurrences.put(digitOccurrence.getKey(), currentOccurrences);
            }
        }
    }

    /**
     * O(n*m*4^k) n and m are row and column of board and k is the length of the word.
     *
     */
    public static boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                boolean[][] visited = new boolean[board.length][board[i].length];
                Stack<int[]> boardToCheck = new Stack<>();

                if (board[i][j] == word.charAt(0)) {
                    visited[i][j] = true;
                    boardToCheck.add(new int[]{i, j+1});
                    boardToCheck.add(new int[]{i, j-1});
                    boardToCheck.add(new int[]{i+1, j});
                    boardToCheck.add(new int[]{i-1, j});

                    if (exists(board, word.substring(1), visited, boardToCheck)) {
                        return true;
                    } else {
                        visited[i][j] = false;
                    }
                }
            }
        }
        return false;
    }

    public static boolean exists(char[][] board, String word, boolean[][] visited, Stack<int[]> boardToCheck) {
        if (word == null || word.equals("")) {
            return true;
        }

        while (!boardToCheck.isEmpty()) {
            int[] positionToCheck = boardToCheck.pop();
            int i = positionToCheck[0];
            int j = positionToCheck[1];
            if (i < 0 || j < 0 || i >= board.length || j >= board[i].length || visited[i][j]) {
                continue;
            } else {
                if (board[i][j] == word.charAt(0)) {
                    visited[i][j] = true;
                    boardToCheck.add(new int[]{i, j + 1});
                    boardToCheck.add(new int[]{i, j - 1});
                    boardToCheck.add(new int[]{i + 1, j});
                    boardToCheck.add(new int[]{i - 1, j});

                    if (exists(board, word.substring(1), visited, boardToCheck)) {
                        return true;
                    } else {
                        visited[i][j] = false;
                    }
                }
            }
        }
        return false;
    }

    /**
     * O(n^2) We take rotating the matrix one circular layer at a time.
     * So start with outermost layer and rotate each of the index to it's corresponding spot while returning
     * the value what was replaced as it will replace the next corresponding spot.
     * Then we iterate to the inner circular layer until we reach one or less layer.
     */
    public static void rotate(int[][] matrix) {
        int topBound = 0;
        int bottomBound = matrix.length-1;
        int leftBound = 0;
        int rightBound = matrix.length-1;
        while (topBound < bottomBound) {
            for (int i = leftBound; i < rightBound; i++) {
                // topBound leftBound++
                // topBound++ rightBound
                // bottomBound rightBound--
                // bottomBound-- leftBound
                int prevValue = matrix[topBound][leftBound+i];
                prevValue = rotate(matrix, topBound+i, rightBound, prevValue);
                prevValue = rotate(matrix, bottomBound, rightBound-i, prevValue);
                prevValue = rotate(matrix, bottomBound-i, leftBound, prevValue);
                rotate(matrix, topBound, leftBound+i, prevValue);
            }

            topBound++;
            bottomBound--;
            leftBound++;
            rightBound--;
        }
    }

    public static int rotate(int[][] matrix, int x, int y, int prevValue) {
        int temp = matrix[x][y];
        matrix[x][y] = prevValue;
        return temp;
    }

    /**
     * O(n!) Since we are recursively calling on n-1 after each step.
     * For DFS/Backtracking problems, we define the base result. In this case, when currentPath is same as
     * permutation size.
     * Otherwise, we recursively call and set the currentPath with a new node value and remove it after the call.
     *
     * Combinatorial search problems involve finding, grouping, and assignments of objects that satisfy certain conditions.
     * Finding all permutations/subsets, solving sudoku, and 8-queens are classic combinatorial problems. Combinatorial
     * search problems boil down to DFS/backtracking on the state-space tree.
     *
     * Three-step system to solve combinatorial search problems
     * 1. Identify the state(s).
     *      - Keep track of the letters we have already selected when we do DFS
     *      - Which letters are left that we can still use (since each letter can only be used once)
     * 2. Draw the state-space tree.
     *      -   1       2       3
     *      -  2 3     1 3     1 2
     *      -  3 2     3 1     2 1
     * 3. DFS/backtrack on the state-space tree.
     */
    public static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> permutations = new LinkedList<>();
        permute(nums, permutations, new Stack<>());
        return permutations;
    }

    public static void permute(int[] nums, List<List<Integer>> permutations, Stack<Integer> currentPath) {
        if (currentPath.size() == nums.length) {
            permutations.add(new LinkedList<>(currentPath));
            return;
        }
        for (int num : nums) {
            if (!currentPath.contains(num)) {
                currentPath.add(num);
                permute(nums, permutations, currentPath);
                currentPath.pop();
            }
        }
    }

    /**
     * O(n^2) We say the n-1 value. So, for countAndSay(3) = say the value of countAndSay(2) = say "11".
     * Youtube explanation of the problem: https://www.youtube.com/watch?v=-wB1xj-kOe0
     */
    public static String countAndSay(int n) {
        String result = "1";

        for (int i = 1; i < n; i++) {
            int matchingCount = 1;
            char prevChar = result.charAt(0);
            String tempResult = "";
            for (int j = 1; j < result.length(); j++) {
                if (prevChar == result.charAt(j)) {
                    matchingCount++;
                } else {
                    tempResult = tempResult + matchingCount + prevChar;
                    matchingCount = 1;
                }
                prevChar = result.charAt(j);
            }
            result = tempResult + matchingCount + prevChar;
        }
        return result;
    }

    /**
     * O(nlogk) Create a PriorityQueue that sorts by frequency of the words, since we want to
     * return the K most frequent words. Iterate k times and return from the queue into a list.
     */
    public static List<String> topKFrequent(String[] words, int k) {
        Arrays.sort(words);
        PriorityQueue<String[]> frequencyQueue = new PriorityQueue<>((a,b)
                -> Integer.compare(Integer.parseInt(b[1]), Integer.parseInt(a[1])));

        String prevWord = null;
        int currentCount = 0;
        for (String word : words) {
            if (prevWord == null || prevWord.equals(word)) {
                currentCount++;
            } else {
                frequencyQueue.add(new String[]{prevWord, String.valueOf(currentCount)});
                currentCount = 1;
            }
            prevWord = word;
        }
        frequencyQueue.add(new String[]{prevWord, String.valueOf(currentCount)});

        List<String> result = new LinkedList<>();
        while (k > 0) {
            k--;
            result.add(Objects.requireNonNull(frequencyQueue.poll())[0]);
        }
        return result;
    }

    /**
     * O(nlogn) We create a priority queue that takes in all the sorting logic required by the
     * problem. We add each of the logs into the queue and return it one by one.
     * Another solution is to create two separate arrays for letters vs digits. We can sort the
     * letters array by using a comparator with the same sorting logic for letter. We then merge
     * the two arrays together. Since digits array doesn't have any sorting logic, just the order
     * that it was originally in, we can just append the array to the sorted digits array.
     */
    public static String[] reorderLogFiles(String[] logs) {
        // Create priority queue with all the sorting logic
        PriorityQueue<String> reorderedLogs = new PriorityQueue<>((String a, String b) -> {
            if (a.startsWith("let") && b.startsWith("dig")) {
                return -1;
            } else if (a.startsWith("dig") && b.startsWith("let")) {
                return 1;
            } else if (a.startsWith("let") && b.startsWith("let")) {
                String aLog = a.substring(4);
                String bLog = b.substring(4);

                if (aLog.compareTo(bLog) < 0) {
                    return -1;
                } else if (aLog.compareTo(bLog) > 0) {
                    return 1;
                } else {
                    int aId = Integer.parseInt(a.substring(3, 4));
                    int bId = Integer.parseInt(b.substring(3,4));
                    if (aId <= bId) {
                        return -1;
                    } else {
                        return 1;
                    }
                }
            } else {
                int aLogId = Integer.parseInt(a.substring(3,4));
                int bLogId = Integer.parseInt(b.substring(3,4));

                if (aLogId <= bLogId) {
                    return -1;
                } else {
                    return 1;
                }
            }
        });

        reorderedLogs.addAll(Arrays.asList(logs));

        String[] result = new String[logs.length];
        int i = 0;
        while(!reorderedLogs.isEmpty()) {
            result[i] = reorderedLogs.poll();
            i++;
        }
        return result;
    }

    /**
     * O(n^2) This is brute force implementation.
     * Optimal solution consists of Binary Tree with all the nodes on the left side is less than the root.
     * We start from the right most nums value, since we know that won't have any smaller count to the right.
     * When we add a node, we get count of nodes on the left (height function) and keep iterating the counter
     * until we have reached the position in the tree for that node.
     */
    public static List<Integer> countSmaller(int[] nums) {
        List<Integer> countSmaller = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            int currentCount = 0;
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[j] < nums[i]) {
                    currentCount++;
                }
            }
            countSmaller.add(currentCount);
        }
        return countSmaller;
    }

    /**
     * O(nlogn) Since generating the priority queue is logn and we need to iterate through the points
     * to populate the priority queue.
     * Another solution would be to compute all the distance into an array. Sort the array and then
     * find the kth distance from the sorted distance array. Then calculate the distance again and
     * print if it's smaller than the kth distance. Same time complexity O(nlogn).
     */
    public static int[][] kClosest(int[][] points, int k) {
        PriorityQueue<List<Double>> minHeap = new PriorityQueue<>(Comparator.comparingDouble(a -> a.get(0)));

        for (int i = 0; i < points.length; i++) {
            double distance = Math.sqrt(Math.pow(points[i][0], 2) + Math.pow(points[i][1], 2));
            minHeap.add(Arrays.asList(distance, (double) i));
        }

        int[][] result = new int[k][2];
        while (k > 0) {
            k--;
            List<Double> minDistance = minHeap.poll();
            result[k] = points[minDistance.get(1).intValue()];
        }
        return result;
    }

    /**
     * O(n*k), where k is the number of steps at each position.
     * For this problem, Chess Knight can move to eight positions, so we add eight new positions to our
     * queue. We also add the steps it took to get to that position within the queue.
     */
    public static int getMinSteps(int x, int y, int tx, int ty) {
        Queue<int[]> queue = new LinkedList<>();
        queue.add(new int[]{x, y, 0});
        int minSteps = Integer.MAX_VALUE;
        while (!queue.isEmpty()) {
            int[] currentPosition = queue.poll();
            System.out.println(currentPosition[0] + " " + currentPosition[1] + " " + currentPosition[2]);
            if (currentPosition[0] == tx && currentPosition[1] == ty) {
                if (currentPosition[2] < minSteps) {
                    minSteps = currentPosition[2];
                }
                break;
            } else {
                queue.add(new int[]{currentPosition[0]+2, currentPosition[1]+1, currentPosition[2]+1});
                queue.add(new int[]{currentPosition[0]-2, currentPosition[1]+1, currentPosition[2]+1});
                queue.add(new int[]{currentPosition[0]+2, currentPosition[1]-1, currentPosition[2]+1});
                queue.add(new int[]{currentPosition[0]-2, currentPosition[1]-1, currentPosition[2]+1});
                queue.add(new int[]{currentPosition[0]+1, currentPosition[1]+2, currentPosition[2]+1});
                queue.add(new int[]{currentPosition[0]-1, currentPosition[1]+2, currentPosition[2]+1});
                queue.add(new int[]{currentPosition[0]+1, currentPosition[1]-2, currentPosition[2]+1});
                queue.add(new int[]{currentPosition[0]-1, currentPosition[1]-2, currentPosition[2]+1});
            }
        }

        return minSteps;
    }

    /**
     * O(n^2) Three main ideas here:
     * 1) Create a Map of all the nodes with their edges. Transform to <1, [2,3,4]> <2, [1,3,5]> <3,[1,2,6]>...
     * 2) Use three sum solution to find all three nodes that make up the trio.
     *      For a given edge, we know all the edges of each of the nodes. So for, edge [1,2] from input,
     *      we know <1, [2,3,4]> and <2, [1,3,5]>. If we can find matching integers within the edges of the map
     *      then we have our three nodes that make up a trio. Each of the nodes from our input edge, 1 and 2, and
     *      our iterated node (3) which is present in edges for node 1 and 2.
     * 3) Given a trio, we can find the degree by adding the edges of the three nodes and subtracting by 6.
     *      Subtract by 6 because for a no degree trio, there are 6 edges.
     */
    public static int minTrioDegree(int n, int[][] edges) {
        // Map of edges for a given node
        Map<Integer, List<Integer>> edgesForNode = new HashMap();
        for (int[] edge : edges) {
            edgesForNode.compute(edge[0], (k, val) -> val == null ? new LinkedList<>() : val).add(edge[1]);
            edgesForNode.compute(edge[1], (k, val) -> val == null ? new LinkedList<>() : val).add(edge[0]);
        }

        // Use three sum logic
        int minDegree = Integer.MAX_VALUE;
        for (int[] edge : edges) {
            List<Integer> edges1 = edgesForNode.get(edge[0]);
            List<Integer> edges2 = edgesForNode.get(edge[1]);

            for (Integer currentEdge : edges1) {
                if (edges2.contains(currentEdge)) {
                    // Find the min degree
                    int currentDegree = (edges1.size() + edges2.size() + edgesForNode.get(currentEdge).size()) - 6;
                    if (currentDegree < minDegree) {
                        minDegree = currentDegree;
                    }
                }
            }
        }
        return minDegree == Integer.MAX_VALUE ? -1 : minDegree;
    }

    /**
     * O(n) Make sure jobDifficulty is sorted. We can add up all the lowest value in the array for each
     * day-1. For the last day, we add the last job's difficulty since it's sorted and is the max difficulty job.
     */
    public static int minDifficulty(int[] jobDifficulty, int d) {
        if (d > jobDifficulty.length) {
            return -1;
        }

        Arrays.sort(jobDifficulty);
        int index = 0;
        int count = 0;
        while (index < d-1) {
            count += jobDifficulty[index];
            index++;
        }

        count += jobDifficulty[jobDifficulty.length-1];
        return count;
    }

    /**
     * O(n) Key here to understand that you can solve this the same way as two or three sum
     * problem. We just need to figure out the math to find the value in the Map.
     * (60 - (currentTime % 60)) % 60
     */
    public static int numPairsDivisibleBy60(int[] time) {
        int count = 0;
        Map<Integer, Integer> timeMap = new HashMap<>();
        for (int currentTime : time) {
            if (timeMap.containsKey((60 - (currentTime % 60)) % 60)) {
                count++;
            }
            timeMap.put(currentTime % 60, -1);
        }
        return count;
    }

    /**
     * O(logn) since we are using binary search to only check at half of the array. Then iterating left and right
     * once we find the target, which is minimal computation if we know target index.
     *
     * Another option is to use binary search twice. One to find the first occurrence, by binary search and looking
     * at left half, and then finding the last occurrence, by binary search and looking at right half.
     */
    public static int[] searchRange(int[] nums, int target) {
        int leftBound = 0;
        int rightBound = nums.length;
        int[] result = new int[] {-1, -1};
        while (leftBound <= rightBound) {
            int index = (rightBound - leftBound) / 2;
            if (nums[index] == target) {
                leftBound = index-1;
                rightBound = index+1;
                result[0] = index;
                result[1] = index;

                while (leftBound > 0 && nums[leftBound] == target) {
                    result[0] = leftBound;
                    leftBound--;
                }
                while (rightBound < nums.length && nums[rightBound] == target) {
                    result[1] = rightBound;
                    rightBound++;
                }
                break;
            } else if (nums[index] > target) {
                rightBound = index-1;
            } else {
                leftBound = index+1;
            }
        }

        return result;
    }

    /**
     * O(n) Iterating through the shorter list once and taking the smaller value from
     * each list as we iterate.
     */
    public static ListNode merge(ListNode l1, ListNode l2) {
        ListNode mergedList = new ListNode();
        ListNode current = mergedList;

        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                current.next = l1;
                l1 = l1.next;
            } else {
                current.next = l2;
                l2 = l2.next;
            }
            current = current.next;
        }
        current.next = (l1 != null) ? l1 : l2;
        return mergedList.next;
    }

    /**
     * O(N * logN) Idea here is to create a priority queue that dynamically returns the
     * class that will benefit the most from adding another passing student.
     * Then we populate the priority queue and add each of the extra student to class
     * that's returned from the priority queue.
     */
    public static double maxAverageRatio(int[][] classes, int extraStudents) {
        PriorityQueue<double[]> priorityQueue = new PriorityQueue<>((a, b) -> {
            double diffPassRatioA = (a[0] + 1) / (a[1] + 1 ) - (a[0] / a[1]);
            double diffPassRatioB = (b[0] + 1) / (b[1] + 1 ) - (b[0] / b[1]);

            if (diffPassRatioA == diffPassRatioB) return 0;
            return diffPassRatioA - diffPassRatioB > 0 ? -1 : 1;
        });

        for (int[] singleClass : classes) {
            double[] passRatioClass = new double[3];
            passRatioClass[0] = singleClass[0]; // passing students
            passRatioClass[1] = singleClass[1]; // total students
            passRatioClass[2] = singleClass[0] * 1.0D / singleClass[1]; // passing ratio
            priorityQueue.add(passRatioClass);
        }

        for (int i = 0; i < extraStudents; i++) {
            double[] maxPassRatioClass = priorityQueue.poll();
            maxPassRatioClass[0] += 1;
            maxPassRatioClass[1] += 1;
            maxPassRatioClass[2] = maxPassRatioClass[0] * 1.0D / maxPassRatioClass[1];

            priorityQueue.add(maxPassRatioClass);
        }

        double sumPassRatio = 0;
        while (!priorityQueue.isEmpty()) {
            sumPassRatio += priorityQueue.poll()[2];
        }
        return sumPassRatio / classes.length;
    }

    public static char slowestKey(int[] releaseTimes, String keysPressed) {
        int maxDuration = 0;
        int previousDuration = 0;
        List<Integer> maxDurationIndexes = new LinkedList();

        for (int i = 0; i < releaseTimes.length; i++) {
            int currentDuration = releaseTimes[i] - previousDuration;
            if (currentDuration > maxDuration) {
                maxDuration = currentDuration;
                maxDurationIndexes = new LinkedList();
                maxDurationIndexes.add(i);
            } else if (currentDuration == maxDuration) {
                maxDurationIndexes.add(i);
            }
            previousDuration = releaseTimes[i];
        }

        Character slowestKey = null;
        for (Integer index : maxDurationIndexes) {
            if (slowestKey == null || keysPressed.charAt(index) > slowestKey) {
                slowestKey = keysPressed.charAt(index);
            }
        }

        return slowestKey;
    }

    public static int maximumUnits(int[][] boxTypes, int truckSize) {
        // Sort the 2D array by the units per box
        // Take the max weighted boxes first and keep iterating down to list
        Arrays.sort(boxTypes, (a,b) -> Integer.compare(b[1], a[1]));

        int totalUnits = 0;
        for (int i = 0; i < boxTypes.length; i++) {
            int numberOfBoxes = boxTypes[i][0];
            for (int j = 0; j < numberOfBoxes && truckSize > 0; j++) {
                totalUnits += boxTypes[i][1];
                truckSize--;
            }
        }

        return totalUnits;
    }

    /**
     * O(n) iterate through once.
     * Hardest part is to understand that at the end of the instructions,
     * if we are back to {0,0} or if the direction changes, then it's bounded.
     * That's because if the direction changes, we can go back to center in
     * either 1, 2, 3, or 4 more tries.
     */
    public static boolean isRobotBounded(String instructions) {
        // Bounded if ending point is at (0,0) or direction changes.
        int[] currentPoint = new int[2];
        char direction = 'N';
        for (char i = 0; i < instructions.length(); i++) {
            if (instructions.charAt(i) == 'G') {
                if (direction == 'N') {
                    currentPoint[1] = currentPoint[1] + 1;
                } else if (direction == 'E') {
                    currentPoint[0] = currentPoint[0] + 1;
                } else if (direction == 'W') {
                    currentPoint[0] = currentPoint[0] - 1;
                } else {
                    currentPoint[1] = currentPoint[1] - 1;
                }
            } else if (instructions.charAt(i) == 'L') {
                if (direction == 'N') {
                    direction = 'W';
                } else if (direction == 'E') {
                    direction = 'N';
                } else if (direction == 'W') {
                    direction = 'S';
                } else {
                    direction = 'E';
                }
            } else {
                if (direction == 'N') {
                    direction = 'E';
                } else if (direction == 'E') {
                    direction = 'S';
                } else if (direction == 'W') {
                    direction = 'N';
                } else {
                    direction = 'W';
                }
            }
        }

        if ((currentPoint[0] == 0 && currentPoint[1] == 0) || (direction != 'N')) {
            return true;
        }
        return false;
    }

    /**
     * For each iteration of the y-axis, we recursively call to see if
     * any of the child (x-axis) is connected. If so, we mark as visited
     * so that next y-axis iteration, we don't increment to final count.
     */
    public static int findCircleNum(int[][] isConnected) {
//        int[][] input = new int[3][3];
//        input[0] = new int[]{1, 1, 0};
//        input[1] = new int[]{1, 1, 0};
//        input[2] = new int[]{0, 0, 1};

        boolean[] isVisited = new boolean[isConnected.length];
        int count = 0;
        for (int i = 0; i < isConnected.length; i++) {
            if (!isVisited[i]) {
                count++;
                dfs(isConnected, isVisited, i);
            }
        }
        return count;
    }

    public static int findCircleNum2(int[][] isConnected) {
        Queue<Integer> queue = new LinkedList<>();
        boolean[] isVisited = new boolean[isConnected.length];
        int circleNum = 0;

        for (int i = 0; i < isConnected.length; i++) {
            if (!isVisited[i]) {
                circleNum++;
                queue.add(i);
                while (!queue.isEmpty()) {
                    int currentIndex = queue.poll();
                    isVisited[currentIndex] = true;
                    for (int j = 0; j < isConnected.length; j++) {
                        if (!isVisited[j] && isConnected[currentIndex][j] == 1) {
                            queue.add(j);
                        }
                    }
                }
            }
        }
        return circleNum;
    }

    public static void dfs(int[][] isConnected, boolean[] isVisited, int depth) {
        for (int i = 0; i < isConnected.length; i++) {
            if (!isVisited[i] && isConnected[depth][i] == 1) {
                isVisited[i] = true;
                dfs(isConnected, isVisited, depth);
            }
        }
    }

    /**
     * O(logn) Find the pivot with binary search, which is logn. Using the pivot,
     * we can find which half of the array to search (since left and right side of the
     * pivot) will be sorted. We can now find the target in logn within the sorted
     * halves. So O(logn) + O(logn) = O(logn)
     */
    public static int search(int[] nums, int target) {
        // Find the pivot index
        int leftIndex = 0;
        int rightIndex = nums.length-1;
        while (leftIndex < rightIndex) {
            int middle = leftIndex + (rightIndex - leftIndex) / 2;
            if (nums[middle] > nums[rightIndex]) {
                leftIndex = middle + 1;
            } else {
                rightIndex = middle;
            }
        }

        int pivot = leftIndex;
        System.out.println("Pivot: " + pivot);
        leftIndex = 0;
        rightIndex = nums.length-1;
        if (target >= nums[pivot] && target <= nums[rightIndex]) {
            leftIndex = pivot;
        } else {
            rightIndex = pivot;
        }
        while(leftIndex <= rightIndex) {
            int middle = leftIndex + (rightIndex - leftIndex) / 2;
            if (nums[middle] == target) {
                return middle;
            } else if (target > nums[middle]) {
                leftIndex = middle + 1;
            } else {
                rightIndex = middle - 1;
            }
        }
        return -1;
    }

    /**
     * O(logn) optimized solution will check to see if dividend is exponentially
     * larger than the divisor. Instead of subtracting by divisor everytime, we shift
     * the divisor each time, multiply by 10, to see if the dividend is still larger
     * than the divisor.
     */
    public static int divide(int dividend, int divisor) {
        int quotient = 0;
        boolean isNegative = false;
        if ((dividend < 0 && divisor >= 0) || (divisor < 0 && dividend >= 0)) {
            isNegative = true;
        }
        dividend = Math.abs(dividend);
        divisor = Math.abs(divisor);
        while (dividend >= divisor) {
            int shifts = 0;
            while (dividend >= (divisor << shifts)) {
                shifts++;
            }

            dividend = dividend - (divisor << (shifts-1));
            quotient = quotient + (1 << (shifts-1));
        }
        return isNegative ? -quotient : quotient;
    }

    /**
     * O(4^n/sqrt(n)) Recursively create valid parens and save it in the results.
     * We want the open parens to go first bc that's the only way to make valid
     * parens.
     */
    public static List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        generateParenthesis(result, "", 0, 0, n);
        return result;
    }

    public static void generateParenthesis(List<String> result, String s, int open, int close, int n) {
        if (open == n && close == n) {
            result.add(s);
            return;
        }
        if (open < n) {
            generateParenthesis(result, s + "(", open + 1, close, n);
        }
        if (close < open) {
            generateParenthesis(result, s + ")", open, close + 1, n);
        }
    }

    /**
     * O(n) we use fast and slow pointers, with fast pointer n steps ahead.
     * We iterate until the fast pointer at the final node. Then we assign slow node's
     * next to slow node's second next.
     */
    public static ListNode removeNthFromEnd(ListNode head, int n) {
//        ListNode head = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5)))));

        ListNode slow = head;
        ListNode fast = head;
        for (int i = 0; i < n; i++) {
            if (fast.next == null) {
                // Edge case for when n equals to the number of nodes, delete the head node.
                if (i == n - 1) {
                    head = head.next;
                }
                return head;
            }
            fast = fast.next;
        }

        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        if (slow.next != null) {
            slow.next = slow.next.next;
        }
        return head;
    }

    /**
     * O(n^2) This implementation has duplicates.
     * Another approach is to sort the array, then iterate through the
     * array with two left and right index (left = i+1, right = array length-1).
     * Since array is sorted, we can move the indexes depending on if the sum is
     * less than or greater than the nums[i]. This also prevents duplicates.
     */
    public static List<List<Integer>> threeSum(int[] nums) {
        // Add all integers in a HashMap with key as num and value as index
        // Iterate through array twice to get two indexes
        // Add the two values and check if the sum equals to negation of a key in HashMap
        Map<Integer, Integer> numsMap = IntStream.range(0, nums.length).boxed()
                .collect(Collectors.toMap(i -> nums[i], i -> i, (a,b) -> a));

        List<List<Integer>> threeSumList = new LinkedList();
        for (int i = 0; i < nums.length; i++) {
            for (int j = i; j < nums.length; j++) {
                if (numsMap.containsKey((nums[i] + nums[j]) * -1)) {
                    int thirdIndex = numsMap.get((nums[i] + nums[j]) * -1);
                    if (i != j && i != thirdIndex && j != thirdIndex) {
                        threeSumList.add(Arrays.asList(nums[i], nums[j], nums[thirdIndex]));
                    }
                }
            }
        }
        return threeSumList;
    }

    /**
     * O(n) we only iterate through the array once and add to the HashMap as we go along.
     */
    public int[] twoSum(int[] nums, int target) {
        int[] result = new int[2];
        // This map will store the difference and the corresponding index
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            // If we have seen the current element before
            // It means we have already encountered the other number of the pair
            if (map.containsKey(nums[i])) {
                result[0] = i;
                result[1] = map.get(nums[i]);
            }
            else {
                // If we have not seen the current before
                // It means we have not yet encountered any number of the pair
                // Save the difference of the target and the current element
                // with the index of the current element
                map.put(target - nums[i], i);
            }
        }
        return result;
    }

    /**
     * O(n) is the optimal time complexity. Using sliding window algorithm, we can start with
     * indexes that give us the largest width (0, height.length-1). We calculate the area and
     * shift the index that has a shorter height.
     */
    public static int maxArea(int[] height) {
        int maxArea = -1;
        int leftIndex = 0;
        int rightIndex = height.length-1;
        while (leftIndex < rightIndex) {
            int currentHeight = Math.min(height[leftIndex], height[rightIndex]);
            int currentWidth = rightIndex - leftIndex;
            int currentArea = currentHeight * currentWidth;

            if (currentArea > maxArea) {
                maxArea = currentArea;
            }
            if (height[leftIndex] > height[rightIndex]) {
                rightIndex--;
            } else {
                leftIndex++;
            }
        }
        return maxArea;
    }

    /**
     * O(n) Iterate once backwards and add the chars in a string builder.
     */
    public static int reverseInt(int x) {
        StringBuilder reverseString = new StringBuilder();
        String xString = String.valueOf(x);
        for (int i = xString.length()-1; i >=0; i--) {
            reverseString.append(xString.charAt(i));
        }

        double reverseDouble = Double.parseDouble(reverseString.toString());
        if (reverseDouble > Integer.MAX_VALUE || reverseDouble < Integer.MIN_VALUE) {
            return 0;
        } else {
            return (int) reverseDouble;
        }
    }

    /**
     * O(n) Iterate once through the string. 32-bit signed integer means in the
     * range [-231, 231 - 1] or [Integer.MIN_VALUE, Integer.MAX_VALUE].
     */
    public static int myAtoi(String s) {
        boolean isNegative = false;
        StringBuilder intValue = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char currentChar = s.charAt(i);
            if (currentChar == '-') {
                isNegative = true;
                continue;
            }

            try {
                Integer currentDigit = Integer.valueOf(String.valueOf(currentChar));
                intValue.append(currentDigit);
            } catch (NumberFormatException e) {
                //ignore;
            }
        }

        double result = Double.parseDouble(isNegative ? ("-" + intValue) : (intValue.toString()));
        if (result > Integer.MAX_VALUE) {
            return Integer.MAX_VALUE;
        } else if (result < Integer.MIN_VALUE) {
            return Integer.MIN_VALUE;
        } else {
            return (int) result;
        }
    }

    /**
     * Time complexity is O(n^2) since we are using the HashMap to cache the isPalindrome value
     * for each string.
     */
    public static String longestPalindrome(String s) {
        Map<String, Boolean> isPalindromeMap = new HashMap<>();
        String longestPalindrome = "";
        for (int i = 0; i < s.length(); i++) {
            for (int j = i+1; j <= s.length(); j++) {
                String palindromeString = s.substring(i, j);
                if (palindromeString.length() > longestPalindrome.length()) {
                    boolean isPalindrome = false;
                    if (isPalindromeMap.containsKey(palindromeString)) {
                        isPalindrome = isPalindromeMap.get(palindromeString);
                    } else {
                        isPalindrome = isPalindrome(palindromeString);
                        isPalindromeMap.put(palindromeString, isPalindrome);
                    }
                    if (isPalindrome) {
                        longestPalindrome = palindromeString;
                    }
                }
            }
        }
        return longestPalindrome;
    }

    /**
     * O(n/2) since we iterate through the string from half out.
     */
    public static boolean isPalindrome(String s) {
        int leftIndex = -1;
        int rightIndex = -1;
        if (s.length() % 2 == 0) {
            leftIndex = (s.length() / 2) -1;
            rightIndex = s.length() / 2;
        } else {
            leftIndex = ((s.length()-1) / 2) - 1;
            rightIndex = ((s.length()-1) / 2) + 1;
        }

        while (leftIndex >= 0
                && rightIndex < s.length()
                && (s.charAt(leftIndex) == s.charAt(rightIndex))) {
            leftIndex--;
            rightIndex++;
        }

        return leftIndex == -1 && rightIndex == s.length();
    }

    /**
     * We can improve to O(n), if we realize that we can ignore all the previous chars once
     * we find a matching char. Since any substring after the start of the previous max-run will
     * always be the best case scenario for that substring. So we just iterate forward to find
     * the next max possible substring.
     */
    public static int lengthOfLongestSubstring(String s) {
        Set<Character> charset = new HashSet<>();
        int maxSoFar = 0;
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            Character charIndex = s.charAt(i);
            if (charset.add(charIndex)) {
                count++;
                if (count > maxSoFar) {
                    maxSoFar = count;
                }
            } else {
                count = 0;
                charset.clear();
                i--;
            }
        }
        return maxSoFar;
    }

    /**
     * O(n) where n is larger of the two ListNode
     * Probably a bug with carryover on the last of the smaller node
     */
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        // Iterate through the two ListNodes together
        // Add the nodes at current iteration and add the sum to a new node
        // (maybe to another node for improve space efficiency)
        // If sum is more than 9, add the carry over to the next node's sum
        ListNode resultNode = new ListNode();
        ListNode head = resultNode;
        int carryover = 0;
        while (l1.next != null && l2.next != null) {
            int sum = l1.val + l2.val + carryover;
            if (sum >= 10) {
                resultNode.val = sum % 10;
                carryover = 1;
            } else {
                resultNode.val = sum;
                carryover = 0;
            }
            l1 = l1.next;
            l2 = l2.next;
            resultNode.next = new ListNode();
            resultNode = resultNode.next;
        }

        if (l1.next == null) {
            int sum = l1.val + l2.val + carryover;
            if (sum >= 10) {
                resultNode.val = sum % 10;
            } else {
                resultNode.val = sum;
            }
            resultNode.next = l2.next;
        } else {
            int sum = l1.val + l2.val + carryover;
            if (sum >= 10) {
                resultNode.val = sum % 10;
            } else {
                resultNode.val = sum;
            }
            resultNode.next = l1.next;
        }
        return head;
    }

    /**
     * O(nlogn) - n for the initial loop to create the hashmap and logn for sorting the entrySet
     */
    public static int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> frequencyByNum = new HashMap<>();
        for (int num : nums) {
            frequencyByNum.put(num, frequencyByNum.getOrDefault(num, 0) + 1);
        }

        return frequencyByNum.entrySet().stream()
                .sorted((a, b) -> {
                    if (Objects.equals(a.getValue(), b.getValue())) {
                        return b.getKey().compareTo(a.getKey());
                    } else {
                        return b.getValue().compareTo(a.getValue());
                    }
                }).map(Map.Entry::getKey)
                .limit(k)
                .mapToInt(i->i)
                .toArray();
    }

    public static class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    public static class TreeNode {
//        TreeNode root = new TreeNode(3, new TreeNode(9), new TreeNode(20, new TreeNode(15), new TreeNode(7)));
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode next;

        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static class LRUCache {
//        LRUCache lRUCache = new LRUCache(2);
//        lRUCache.put(1, 1); // cache is {1=1}
//        lRUCache.put(2, 2); // cache is {1=1, 2=2}
//        System.out.println(lRUCache.get(1));    // return 1
//        lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
//        System.out.println(lRUCache.get(2));    // returns -1 (not found)
//        lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
//        System.out.println(lRUCache.get(1));    // return -1 (not found)
//        System.out.println(lRUCache.get(3));    // return 3
//        System.out.println(lRUCache.get(4));    // return 4


        int capacity = 0;
        Queue<Integer> queue;
        Map<Integer, Integer> valueMap = new HashMap();

        public LRUCache(int capacity) {
            this.capacity = capacity;
            this.queue = new LinkedList<>();
        }

        public int get(int key) {
            if (!this.valueMap.containsKey(key)) {
                return -1;
            }

            this.queue.remove(key);
            this.queue.add(key);
            return this.valueMap.get(key);
        }

        public void put(int key, int value) {
            if (this.queue.size() >= capacity) {
                Integer removedKey = this.queue.remove();
                this.valueMap.remove(removedKey);
            }

            this.queue.add(key);
            this.valueMap.put(key, value);
        }
    }
}
