import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

public class GeneralProblems {

    public static void main(String[] args) {
        int[][] map = new int[][]{{1,0,0,0,1}, {0,0,0,0,0}, {0,0,1,0,0}};
        System.out.println(minTravelDistance(map));
    }

    /**
     * If we see problem with brackets, we should use Stacks.
     * Give this encoding rule, k[some_string] -> some_string is repeated k times.
     *      3[a2[c]] -> accaccacc
     * If we create two Stacks, one to push the repeats and one to push the current string.
     * We can pop both off when we see the closing bracket and add to the current string.
     * Time O(n), since we need to loop through the String once.
     * Space O(n), since we need to create two Stacks. O(n+n) = O(n).
     */
    public static String decode(String s) {
        Stack<Integer> repeats = new Stack<>();
        Stack<String> results = new Stack<>();

        String result = "";
        int i = 0;
        while (i < s.length()) {
            if (Character.isDigit(s.charAt(i))) {
                int repeat = 0;
                while(Character.isDigit(s.charAt(i))) {
                    repeat = (repeat * 10) + Integer.parseInt(s.charAt(i) + "");
                    i++;
                }
                repeats.add(repeat);
            } else if (s.charAt(i) == '[') {
                results.add(result);
                result = "";
                i++;
            } else if (s.charAt(i) == ']') {
                StringBuilder repeatedVal = new StringBuilder(results.pop());
                int repeat = repeats.pop();
                while(repeat > 0) {
                    repeatedVal.append(result);
                    repeat--;
                }
                result = repeatedVal.toString();
                i++;
            } else {
                result += s.charAt(i);
                i++;
            }
        }
        return result;
    }

    /**
     * Given these two inputs, "alex" and "aaleex", if second value seems like it was caused
     * by a long press of keystroke. Then return true. We iterate through the strings once
     * keeping two indexes for each value.
     *
     * Time O(n), where n is the longer length of the two inputs.
     * Space O(1)
     */
    public static boolean isLongPressed(String name, String typed) {
        int i = 0;
        int j = 0;
        boolean failedCheck = false;
        while (i < name.length() && j < typed.length()) {
            if (name.charAt(i) == typed.charAt(j)) {
                failedCheck = false;
                j++;
            } else {
                if (failedCheck) {
                    return false;
                }
                failedCheck = true;
                i++;
            }
        }

        while (i < name.length()) {
            if (typed.charAt(j-1) != name.charAt(i)) {
                return false;
            }
            i++;
        }
        return true;
    }

    /**
     * Given two string, where t is the all the characters required in a substring of s, find the shortest
     * substring. We have two pointers and keep moving the right pointer until we can create a valid substring.
     * Then we move left pointer until we see one of the characters from t. We also keep track of the shortest
     * substring.
     * Time O(nlogn), since we only iterate parts of the string to check if it's a valid substring
     * Space O(m), where m is the size of the t to create the HashMap.
     */
    public static String minWindowSubstring(String s, String t) {
        Map<Character, Integer> tCharToCountMap = charToCountMap(t);
        Set<Character> tSet = tCharToCountMap.keySet();

        int left = 0;
        int right = t.length();
        String minString = s + "1"; // Make it larger than the entire string
        while (right <= s.length()) {
            String subString = s.substring(left, right);
            if (isSubstringInString(subString, new HashMap<>(tCharToCountMap))) {
                if (subString.length() < minString.length()) {
                    minString = subString;
                }
                left++;
                while (left < s.length() && !tSet.contains(s.charAt(left))) {
                    left++;
                }
            } else {
                right++;
            }
        }
        return minString.length() > s.length() ? "" : minString;
    }

    public static Map<Character, Integer> charToCountMap(String t) {
        Map<Character, Integer> charToCountMap = new HashMap<>();
        for (char tChar : t.toCharArray()) {
            int count = charToCountMap.getOrDefault(tChar, 0);
            charToCountMap.put(tChar, count+1);
        }
        return charToCountMap;
    }

    public static boolean isSubstringInString(String s, Map<Character, Integer> charToCountMap) {
        for (char sChar : s.toCharArray()) {
            int count = charToCountMap.getOrDefault(sChar, 0);
            charToCountMap.put(sChar, count-1);
        }
        for (Integer count : charToCountMap.values()) {
            if (count > 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Given a stretched word and possible actual words, we check if the actual word exists for the stretchy word.
     * A word is stretchy if at least one of the chars has more than 2 of those characters. If it doesn't, then
     * num of characters must match to the actual word.
     * "hello" -> heeellooo", but not, "hello" -> "heello" since e needs occur more than twice.
     * Time O(m*n) where m is the size of wordlist and n is the max size of all the strings inputted.
     * Space O(1)
     */
    public static int stretchy(String s, String[] words) {
        // iterate the words
        // keep two pointers s and words[i]
        // for each pointer, we loop to find number of repeats
        // we compare the two counts, making sure the s repeats count is 3 more.
        // if so, return i
        // else, keep iterating
        // if nothing is returned return -1;
        for (int i = 0; i < words.length; i++) {
            int left = 0;
            int right = 0;
            boolean extension = false;

            while (left < s.length() && right < words[i].length()) {
                char sChar = s.charAt(left);
                char wordChar = words[i].charAt(right);
                if (sChar != wordChar) {
                    left = 0;
                    right = 0;
                    break;
                }
                int leftCount = 0;
                while (left < s.length() && s.charAt(left) == sChar) {
                    leftCount++;
                    left++;
                }
                int rightCount = 0;
                while (right < words[i].length() &&  words[i].charAt(right) == wordChar) {
                    rightCount++;
                    right++;
                }
                if (leftCount == rightCount) {
                    continue;
                }
                if (leftCount < rightCount+2) {
                    left = 0;
                    right = 0;
                    break;
                } else {
                    extension = true;
                }
            }
            if (extension && left == s.length() && right == words[i].length()) {
                return i+1;
            }
        }
        return -1;
    }

    /**
     * Format list of words given we have max width of line size. Each line of words should be in the following format.
     * "This   is  an"
     * "example    of"
     * "formatting   "
     * Time O(nlogn), iterate through each word, but we only format part of the word list that would fit in maxWidth.
     * Space O(1)
     */
    public static List<String> textFormat(String[] words, int maxWidth) {
        // Iterate through words
        // count the size plus a space between each word until we reach to maxWidth
        // then we format() with the words and figure out the spacing
        // then we set the size count = 0

        List<String> result = new LinkedList<>();
        List<String> currentLineWords = new LinkedList<>();
        int size = 0;
        for (String word : words) {
            int currentSize;
            if (size == 0) {
                currentSize = word.length();
            } else {
                currentSize = size + 1 + word.length();
            }

            if (currentSize > maxWidth) {
                String formattedString = format(currentLineWords, maxWidth);
                result.add(formattedString);
                currentLineWords.clear();
                size = word.length();
                currentLineWords.add(word);
            } else {
                size = currentSize;
                currentLineWords.add(word);
            }
        }
        if (!currentLineWords.isEmpty()) {
            result.add(format(currentLineWords, maxWidth));
        }
        return result;
    }

    public static String format(List<String> words, int maxWidth) {
        if (words.size() == 1) {
            StringBuilder result = new StringBuilder(words.get(0));
            int newSpaceBetween = maxWidth-words.get(0).length();
            while (newSpaceBetween > 0) {
                result.append(" ");
                newSpaceBetween--;
            }
            return result.toString();
        }


        int spaceCount = maxWidth;
        for (String word : words) {
            spaceCount = spaceCount - word.length();
        }

        int remainder = spaceCount % (words.size()-1);
        int spaceBetween = spaceCount / (words.size()-1);
        StringBuilder result = new StringBuilder(words.get(0));
        for (int i = 1; i < words.size(); i++) {
            int newSpaceBetween = spaceBetween;
            while (newSpaceBetween > 0) {
                result.append(" ");
                newSpaceBetween--;
            }
            while (remainder > 0) {
                result.append(" ");
                remainder--;
            }
            result.append(words.get(i));
        }

        return result.toString();
    }

    /**
     * Given a map containing 1s and 0s, where 1s denote a person. We need to find the min distance to
     * travel for all the people to meetup. We need to find the median row and column index from the
     * people list. Then we compute the distance it takes to get to the median position from each of
     * the people and return the sum.
     *
     * Time O(n*m) where n and m are length and width of the input map.
     * Space O(p) where p is number of people on the map.
     */
    public static int minTravelDistance(int[][] map) {
        List<Integer> peopleRows = new LinkedList<>();
        List<Integer> peopleCols = new LinkedList<>();
        for (int i = 0; i < map.length; i++) {
            for (int j = 0; j < map[i].length; j++) {
                if (map[i][j] == 1) {
                    peopleRows.add(i);
                    peopleCols.add(j);
                }
            }
        }
        Collections.sort(peopleRows);
        Collections.sort(peopleCols);

        int medianIndex = peopleRows.size() / 2;
        int medianRow = peopleRows.get(medianIndex);
        int medianCol = peopleCols.get(medianIndex);

        int travelDistance = 0;
        for (int i = 0 ; i < peopleCols.size(); i++) {
            travelDistance += Math.abs(peopleCols.get(i) - medianCol) + Math.abs(peopleRows.get(i) - medianRow);
        }
        return travelDistance;
    }

    /**
     * Number of sub arrays in nums that adds up to target, which can be solved using two pointers.
     * Tricky part is if the running sum is greater than target, then we keep iterating until
     * the running sum is less than or equal to target.
     * Time O(n), iteration within only goes up to right pointer, hence it just n + n for worst case.
     * Space O(1)
     */
    public static int subArraySum(int[] nums, int target) {
        int leftPointer = 0;
        int rightPointer = 0;

        int result = 0;
        int runningSum = 0;
        while (rightPointer < nums.length) {
            runningSum += nums[rightPointer];
            if (runningSum == target) {
                result++;
                runningSum -= nums[leftPointer];
                leftPointer++;
            } else if (runningSum > target) {
                while (runningSum >= target) {
                    if (runningSum == target) {
                        result++;
                    }
                    runningSum -= nums[leftPointer];
                    leftPointer++;
                }
            }
            rightPointer++;
        }
        return result;
    }

    /**
     * Find a substring of s, where t would be the substring of the new substring.
     * We use left and right pointers to only iterate through the initial string once.
     * We can do the check for second substring check in linear time but with shorter substring t.
     * Time O(n*m) where n is size of s and m is size of t
     * Space O(1)
     */
    public static String minWindowSubsequence(String s, String t) {
        int leftPointer = 0;
        int rightPointer = t.length();
        String minSubstring = null;
        while (rightPointer < s.length()) {
            String subString = s.substring(leftPointer, rightPointer);
            if (containsCharInOrder(subString, t)) {
                if (minSubstring == null) {
                    minSubstring = subString;
                } else if (subString.length() < minSubstring.length()) {
                    minSubstring = subString;
                }
                leftPointer++;
            } else {
                rightPointer++;
            }
        }
        return minSubstring;
    }

    public static boolean containsCharInOrder(String val, String t) {
        int prevIndex = 0;
        for (char c : t.toCharArray()) {
            int index = val.indexOf(c, prevIndex);
            if (index < 0) {
                return false;
            }
            prevIndex += index;
        }
        return true;
    }

    /**
     * Given two ListNode, which represents an integer in reverse order. We need to sum the two integers.
     * We just iterate through the list together and sum the values including carryover from prev sum and
     * add it to val1 ListNode.
     * When we exit the while loop, we need to check if val1 is null or val2 is null.
     * For val1 being null, we need to append the val2 to val1 since that's our solution without any extra space.
     *      This can be achieved by keeping track of val1's previous and pointing it's next to val2.
     * For va2 being null, we don't need to do anything further since the result is already in val1.
     * Finally, we worry about the carryover and add it to val1's current node if it exists.
     *
     * Time O(n), where n is the shorter of the two lists.
     * Space O(1), since we are using val1 to store our result.
     */
    public static ListNode sumTwoLists(ListNode val1, ListNode val2) {
        ListNode current1 = val1;
        ListNode current2 = val2;
        ListNode current1Prev = null;

        int carryOver = 0;
        while (current1 != null && current2 != null) {
            int result = current1.val + current2.val + carryOver;
            if (result >= 10) {
                carryOver = result / 10;
                result = result % 10;
            } else {
                carryOver = 0;
            }
            current1.val = result;
            current1Prev = current1;
            current1 = current1.next;
            current2 = current2.next;
        }
        if (current2 != null) {
            current1Prev.next = current2;
            current1 = current1Prev.next;
        }
        if (carryOver > 0) {
            current1.val = current1.val + carryOver;
        }

        return val1;
    }

    public static class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

}
