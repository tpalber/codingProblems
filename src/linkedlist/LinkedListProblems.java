package linkedlist;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

public class LinkedListProblems {

    public static void main(String[] args) {
//        Node head = new Node(1, new Node(2, new Node(3, new Node(4, null))));
//        Node head2 = new Node(5, new Node(6, null));
//        iterateLinkedList(zipperList(head2, head));

        System.out.println(maxArea(new int[] {3,2,4,5,7,6,1,3,8,9,11,10,7,5,2,6}));
    }

    /**
     * Find the max area of histogram, where each index in histogram represents height of histogram.
     * Brute force solution, at each index, we check left and right and find index that has value less than current.
     *      We find the length from left to right index and multiply by current value to find area. Time O(n^2)
     * Optimal solution, create the lowest left and lowest right index array for each index. We can use a stack
     *      to find those values. Once we have those two array, we iterate and find the length from left to right
     *      and multiply by current value to find area. Time O(n+n+n) = O(n) & Space O(n+n+n) = O(n).
     */
    public static int maxArea(int[] histogram) {
        int[] leftMinIndexArray = new int[histogram.length];
        int[] rightMinIndexArray = new int[histogram.length];
        Stack<Integer[]> stack = new Stack<>();
        stack.add(new Integer[] {-1,-1});

        for (int i = 0; i < histogram.length; i++) {
            Integer[] previousVal = stack.peek();
            if (previousVal[1] <= histogram[i]) {
                stack.add(new Integer[]{i, histogram[i]});
            } else {
                while (previousVal[1] > histogram[i]) {
                    previousVal = stack.pop();
                }
                stack.add(new Integer[] {previousVal[0],previousVal[1]});
            }
            leftMinIndexArray[i] = previousVal[0];
        }

        stack = new Stack<>();
        stack.add(new Integer[] {histogram.length,-1});
        for (int i = histogram.length-1; i >= 0; i--) {
            Integer[] previousVal = stack.peek();
            if (previousVal[1] <= histogram[i]) {
                stack.add(new Integer[]{i, histogram[i]});
            } else {
                while (previousVal[1] > histogram[i]) {
                    previousVal = stack.pop();
                }
                stack.add(new Integer[] {previousVal[0],previousVal[1]});
            }
            rightMinIndexArray[i] = previousVal[0];
        }

        int maxArea = Integer.MIN_VALUE;
        for (int i = 0; i < histogram.length; i++) {
            int area = (rightMinIndexArray[i] - leftMinIndexArray[i]) * histogram[i];
            if (area > maxArea) {
                maxArea = area;
            }
        }
        return maxArea;
    }

    /**
     * Find the shortest substring of s, which contains all the chars in t. Algorithm is written below.
     * Time O(n^2) since we iterate through substring to check if it's valid and iterate through s once.
     * Space O(m) since we have a frequency map with t sizes.
     *
     * Optimal solution can have time complexity of O(n) if we can figure out how to do the validation of substring
     * in constant time. Maybe create another Map which contains all the characters visited and the frequency of it.
     */
    public static String shortestSubstring(String s, String t) {
        // ABCDAC, AC
        // ABC
        // BC -> BCD -> BCDA
        // CDA
        // DA -> DAC
        // AC

        // Put the t in a frequency map, this will help us determine if substring is valid
        // We keep track of left and right index that will iterate s
        // At each iteration if right index > size, return solution
        // At each iteration you check to see if the substring is valid. If so,
        //      - compare it with previous valid substring size and update solution
        //      - move the left index forward and check again
        // At each iteration if substring is not valid,
        //      - move the right index forward and check again

        Map<Character, Integer> frequencyMap = getFrequencyMap(t);
        String shortestSubstring = "";
        int subStringSize = Integer.MAX_VALUE;
        int leftIndex = 0;
        int rightIndex = 1;
        while (rightIndex <= s.length()) {
            String substring = s.substring(leftIndex, rightIndex);
            if (validSubstring(substring, new HashMap<>(frequencyMap))) {
                if (substring.length() < subStringSize) {
                    shortestSubstring = substring;
                    subStringSize = shortestSubstring.length();
                }
                leftIndex++;
            } else {
                rightIndex++;
            }
        }
        return shortestSubstring;
    }

    private static Map<Character, Integer> getFrequencyMap(String t) {
        Map<Character, Integer> frequencyMap = new HashMap<>();
        for (char stringChar : t.toCharArray()) {
            frequencyMap.put(stringChar, frequencyMap.getOrDefault(stringChar, 0) + 1);
        }
        return frequencyMap;
    }

    private static boolean validSubstring(String s, Map<Character, Integer> frequencyMap) {
        for (char sChar : s.toCharArray()) {
            if (frequencyMap.containsKey(sChar)) {
                frequencyMap.put(sChar, frequencyMap.get(sChar)-1);
            }
        }

        for (int frequency : frequencyMap.values()) {
            if (frequency > 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Find the staring gas station given gas prices and miles in tank. Has a unique answer.
     * Optimal solution requires only iterating once through the array, but we keep track of candidate
     * where the remaining sum is greater or equal to 0. Otherwise, we set the candidate to the next
     * index of when the remaining sum is less than 0. We can also keep track of all the remaining sums
     * so that we don't have to iterate again later.
     * Time O(n) & Space O(1)
     */
    public static int startingGasStation(int[] gas, int[] cost) {
        int remaining = 0;
        int candidate = 0;
        int previousRemaining = 0;
        for(int i = 0; i < gas.length; i++) {
            remaining += gas[i] - cost[i];
            if (remaining < 0) {
                candidate = i+1;
                previousRemaining += remaining;
                remaining = 0;
            }
        }

        if (candidate >= gas.length || previousRemaining + remaining < 0) {
            return -1;
        } else {
            return candidate;
        }
    }
    public static Node mergeSortedLists(Node head1, Node head2) {
        if (head1 == null) {
            return head2;
        }
        if (head2 == null) {
            return head1;
        }
        if (head1.valInt <= head2.valInt) {
            head1.next = mergeSortedLists(head1.next, head2);
            return head1;
        } else {
            head2.next = mergeSortedLists(head1, head2.next);
            return head2;
        }
    }

    /**
     * Zipper add two linked list. Time O(min(n,m)) and Space O(min(n,m)) where n and m are length of two linked lists.
     * Recursive implementation seems easier but has more space complexity since each call is kept in the call stack.
     *      Set head1.next = head2; Set head2.next = recursive call response; Return head1;
     * Iterative implementation requires keeping a tail pointer, so that you can add the remaining longer linked list
     * to the tail.next.
     */
    public static Node zipperListRecursive(Node head1, Node head2) {
        if (head1 == null && head2 == null) {
            return null;
        }
        if (head1 == null) {
            return head2;
        }
        if (head2 == null) {
            return head1;
        }
        Node head1Next = head1.next;
        Node head2Next = head2.next;
        head1.next = head2;
        head2.next = zipperListRecursive(head1Next, head2Next);
        return head1;
    }

    public static Node zipperList(Node head1, Node head2) {
        Node current1 = head1.next;
        Node current2 = head2;
        Node tail = head1;
        while (current1 != null && current2 != null) {
            tail.next = current2;
            current2 = current2.next;
            tail.next.next = current1;
            current1 = current1.next;
            tail = tail.next.next;
        }

        if (current1 != null) {
            tail.next = current1;
        } else if (current2 != null) {
            tail.next = current2;
        }
        return head1;
    }

    /**
     * Reverse a linked list
     * 1. Stack implementation. Time O(n), Space O(n)
     * 2. Pointer manipulation, reversing the next pointer to the previous node. Time O(n), Space O(1) for iterative.
     *      (optimal solution, since we don't need to create another data structure)
     */
    public static Node reverse(Node head) {
        Node current = head;
        Node previous = null;
        while (current != null) {
            Node nextTemp = current.next;
            current.next = previous;
            previous = current;
            current = nextTemp;
        }

        return previous;
    }

    public static Node reverseRecursive(Node head) {
        Node previous = null;
        return reverseRecursive(head, previous);
    }

    public static Node reverseRecursive(Node head, Node previous) {
        if (head == null) {
            return previous;
        }
        Node nextTemp = head.next;
        head.next = previous;
        return reverseRecursive(nextTemp, head);
    }

    public static int getValue(Node head, int index) {
        Node current = head;
        while (current != null) {
            if (0 == index) {
                return current.valInt;
            }
            index--;
            current = current.next;
        }
        return -1;
    }

    public static int getValueRecursive(Node head, int index) {
        if (head == null) {
            return -1;
        } else if (index == 0) {
            return head.valInt;
        }
        return getValueRecursive(head.next, index-1);
    }

    public static boolean findInList(Node head, int target) {
        Node current = head;
        while (current != null) {
            if (current.valInt == target) {
                return true;
            }
            current = current.next;
        }
        return false;
    }

    public static boolean findInListRecursive(Node head, int target) {
        if (head == null) {
            return false;
        } else if (head.valInt == target) {
            return true;
        }
        return findInListRecursive(head.next, target);
    }

    public static int sumList(Node head) {
        int sum = 0;
        Node current = head;
        while (current != null) {
            sum += current.valInt;
            current = current.next;
        }
        return sum;
    }

    public static int sumListRecursive(Node head) {
        if (head == null) {
            return 0;
        }
        return head.valInt + sumListRecursive(head.next);
    }

    /**
     * Find the intersecting node of two linked list.
     * - Get the size of each linked list
     * - Find the diff between the sizes and increment the larger linked list by diff amount
     * - With both pointer at each node with same length, find the intersecting node
     *
     * Time O(m+n) since we iterate through each of the lists but never within each other.
     * Space O(1)
     */
    public static Node intersectingNode(Node node1, Node node2) {
        int count1 = getCount(node1);
        int count2 = getCount(node2);

        if (count1 >= count2) {
            return intersectingNode(count1-count2, node1, node2);
        } else {
            return intersectingNode(count2-count1, node2, node1);
        }
    }

    private static int getCount(Node root) {
        Node current = root;
        int count = 0;
        while (current != null) {
            count++;
            current = current.next;
        }
        return count;
    }

    private static Node intersectingNode(int diff, Node node1, Node node2) {
        Node current1 = node1;
        int i = 0;
        while (i < diff) {
            current1 = current1.next;
            i++;
        }

        Node current2 = node2;
        while (current1 != null || current2 != null) {
            if (current1.valInt == current2.valInt) {
                return current1;
            }
            current1 = current1.next;
            current2 = current2.next;
        }
        return null;
    }

    public static void iterateLinkedList(Node head) {
        Node current = head;
        while(current != null) {
            System.out.println(current.valInt);
            current = current.next;
        }
    }

    public static void recursiveLinkedList(Node head) {
        if (head == null) {
            return;
        }
        System.out.println(head.valInt);
        recursiveLinkedList(head.next);
    }

    public static class Node {
        String valString;
        int valInt;
        Node next;

        public Node() {}

        public Node(String valString, Node next) {
            this.valString = valString;
            this.next = next;
        }

        public Node(int valInt, Node next) {
            this.valInt = valInt;
            this.next = next;
        }

    }
}
