import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.URI;
import java.net.SocketException;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/** Explicit local interface authorities, with same-origin checks for the shared LAN studio. */
final class NetworkPolicy {
    private final boolean lan;
    private final Set<String> authorities;

    NetworkPolicy(boolean lan, int port) throws SocketException {
        this(lan, localAuthorities(lan, port));
    }

    NetworkPolicy(boolean lan, int port, String publicUrl) throws SocketException {
        this(lan, withPublicUrl(lan, port, publicUrl));
    }

    private static Set<String> withPublicUrl(boolean lan, int port, String publicUrl) throws SocketException {
        Set<String> hosts = localAuthorities(lan, port);
        if (publicUrl != null && !publicUrl.isBlank()) {
            if (!lan) throw new IllegalArgumentException("Public URL requires LAN mode.");
            hosts.add(publicAuthority(publicUrl));
        }
        return hosts;
    }

    static String publicAuthority(String publicUrl) {
        try {
            URI uri = URI.create(publicUrl);
            String host = uri.getHost();
            if (!"http".equals(uri.getScheme()) || host == null || !host.matches("[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+")
                    || uri.getUserInfo() != null || uri.getQuery() != null || uri.getFragment() != null
                    || !(uri.getPath().isEmpty() || uri.getPath().equals("/")) || uri.getPort() < 1 || uri.getPort() > 65535)
                throw new IllegalArgumentException("Use http://PRIVATE-IP:PORT as the public URL.");
            InetAddress address = InetAddress.getByName(host);
            if (!address.getHostAddress().equals(host) || !(address.isSiteLocalAddress() || address.isLoopbackAddress()))
                throw new IllegalArgumentException("Public URL must use a private or loopback IPv4 address.");
            return host + ":" + uri.getPort();
        } catch (Exception error) {
            throw new IllegalArgumentException("Invalid STUDIO_PUBLIC_URL: expected http://PRIVATE-IP:PORT", error);
        }
    }

    NetworkPolicy(boolean lan, Set<String> authorities) {
        this.lan = lan;
        this.authorities = Set.copyOf(authorities);
    }

    static Set<String> localAuthorities(boolean lan, int port) throws SocketException {
        Set<String> hosts = new HashSet<>(Set.of("127.0.0.1:" + port, "localhost:" + port, "[::1]:" + port));
        if (lan) {
            for (var interfaces = NetworkInterface.getNetworkInterfaces(); interfaces.hasMoreElements();) {
                var network = interfaces.nextElement();
                if (!network.isUp()) continue;
                for (var addresses = network.getInetAddresses(); addresses.hasMoreElements();) {
                    var address = addresses.nextElement();
                    if (address instanceof Inet4Address && address.isSiteLocalAddress()) hosts.add(address.getHostAddress() + ":" + port);
                }
            }
        }
        return hosts;
    }

    boolean allows(InetAddress peer, String host, String origin, String fetchSite) {
        if (peer == null || !(peer.isLoopbackAddress() || lan && peer.isSiteLocalAddress())) return false;
        if (host == null || !authorities.contains(host.toLowerCase(Locale.ROOT))) return false;
        if ("cross-site".equals(fetchSite)) return false;
        return origin == null || origin.equalsIgnoreCase("http://" + host);
    }

    Set<String> lanAuthorities() {
        Set<String> hosts = new HashSet<>(authorities);
        hosts.removeIf(host -> host.startsWith("127.") || host.startsWith("localhost:") || host.startsWith("[::1]:"));
        return Set.copyOf(hosts);
    }
}
