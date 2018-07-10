//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package word2vec.util;

import java.util.HashMap;

public class MapCount<T> {
    private HashMap<T, Integer> hm = null;

    public MapCount() {
        this.hm = new HashMap();
    }

    public void add(T t, int n) {
        Integer integer;
        if((integer = this.hm.get(t)) != null) {
            this.hm.put(t, Integer.valueOf(integer.intValue() + n));
        } else {
            this.hm.put(t, Integer.valueOf(n));
        }

    }

    public void add(T t) {
        this.add(t, 1);
    }

    public int size() {
        return this.hm.size();
    }

    public HashMap<T, Integer> get() {
        return this.hm;
    }

}
