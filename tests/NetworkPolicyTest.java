import java.net.InetAddress;
import java.util.Set;

public final class NetworkPolicyTest {
    private static int count;
    private static void check(boolean actual, String label) {
        if (!actual) throw new AssertionError(label);
        count++;
    }
    public static void main(String[] args) throws Exception {
        var local = InetAddress.getByName("127.0.0.1");
        var privatePeer = InetAddress.getByName("192.168.1.20");
        var publicPeer = InetAddress.getByName("8.8.8.8");
        var hosts = Set.of("127.0.0.1:8080", "localhost:8080", "192.168.1.10:8080");
        var lan = new NetworkPolicy(true, hosts);
        check(lan.allows(privatePeer, "192.168.1.10:8080", "http://192.168.1.10:8080", "same-origin"), "LAN same-origin upload");
        check(lan.allows(privatePeer, "192.168.1.10:8080", null, null), "LAN media download");
        check(lan.allows(local, "localhost:8080", null, null), "loopback health");
        check(!lan.allows(publicPeer, "192.168.1.10:8080", null, null), "public peers denied");
        check(!lan.allows(privatePeer, "attacker.example:8080", null, null), "DNS rebinding host denied");
        check(!lan.allows(privatePeer, "192.168.1.10:8080", "http://attacker.example", null), "foreign origin denied");
        check(!lan.allows(privatePeer, "192.168.1.10:8080", "http://127.0.0.1:8080", null), "different trusted origin denied");
        check(!lan.allows(privatePeer, "192.168.1.10:8080", "null", null), "null origin denied");
        check(!lan.allows(privatePeer, "192.168.1.10:8080", null, "cross-site"), "cross-site request denied");
        check(!lan.allows(privatePeer, "192.168.1.10:9999", null, null), "wrong port denied");
        check(!lan.allows(privatePeer, null, null, null), "missing host denied");
        var onlyLocal = new NetworkPolicy(false, hosts);
        check(!onlyLocal.allows(privatePeer, "192.168.1.10:8080", null, null), "local mode excludes peers");
        check(onlyLocal.allows(local, "127.0.0.1:8080", null, null), "local mode retained");
        check(NetworkPolicy.publicAuthority("http://192.168.250.10:9173").equals("192.168.250.10:9173"), "Docker external port authority");
        var docker = new NetworkPolicy(true, 8080, "http://192.168.250.10:9173");
        check(docker.allows(privatePeer,"192.168.250.10:9173","http://192.168.250.10:9173","same-origin"), "Docker published origin");
        check(!docker.allows(privatePeer,"192.168.250.10:9174",null,null), "Docker other port rejected");
        for (String url : new String[]{"http://attacker.example:9173","http://8.8.8.8:9173","http://192.168.250.10:9173/path","http://user@192.168.250.10:9173","http://192.168.250.10:0"}) {
            boolean rejected=false;
            try { NetworkPolicy.publicAuthority(url); } catch (IllegalArgumentException expected) { rejected=true; }
            check(rejected,"invalid published origin rejected");
        }
        System.out.println("PASS " + count + " network policy cases");
    }
}
