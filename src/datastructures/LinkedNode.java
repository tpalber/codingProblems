package datastructures;

public class LinkedNode {
    public String valString;
    public int valInt;
    public LinkedNode next;

    public LinkedNode() {}

    public LinkedNode(String valString, LinkedNode next) {
        this.valString = valString;
        this.next = next;
    }

    public LinkedNode(int valInt, LinkedNode next) {
        this.valInt = valInt;
        this.next = next;
    }
}
