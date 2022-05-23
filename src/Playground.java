import java.util.HashMap;
import java.util.Map;

public class Playground {
    public static void main(String[] args) {
        Map<Character, Integer> testMap = new HashMap<>();
        testMap.put(null, null);

        System.out.println(testMap.containsKey(null));
    }
}
