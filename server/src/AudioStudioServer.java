import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.BindException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;

/** Local Vue host and same-origin gateway to the managed algorithm process. */
public final class AudioStudioServer {
    private static final Map<String, String> MIME = Map.ofEntries(
        Map.entry("html", "text/html; charset=utf-8"),
        Map.entry("js", "application/javascript; charset=utf-8"),
        Map.entry("css", "text/css; charset=utf-8"),
        Map.entry("json", "application/json; charset=utf-8"),
        Map.entry("svg", "image/svg+xml"), Map.entry("png", "image/png"),
        Map.entry("jpg", "image/jpeg"), Map.entry("webp", "image/webp"),
        Map.entry("ico", "image/x-icon"), Map.entry("woff2", "font/woff2"),
        Map.entry("wav", "audio/wav"), Map.entry("mp3", "audio/mpeg")
    );

    public static void main(String[] args) {
        try {
            int port = 8081;
            String host = "127.0.0.1";
            String publicUrl = System.getenv("STUDIO_PUBLIC_URL");
            Path root = Path.of("dist");
            for (int i = 0; i < args.length; i++) {
                if (args[i].equals("--help")) {
                    System.out.println("java -jar audio-studio-server.jar [--root dist] [--port 8081] [--host 127.0.0.1|0.0.0.0|LAN-IP]");
                    return;
                }
                if (i + 1 == args.length) throw new IllegalArgumentException("Missing option value: " + args[i]);
                String option = args[i], value = args[++i];
                switch (option) {
                    case "--root" -> root = Path.of(value);
                    case "--port" -> port = Integer.parseInt(value);
                    case "--host" -> host = value;
                    default -> throw new IllegalArgumentException("Unknown option: " + option);
                }
            }
            if (port < 1 || port > 65535) throw new IllegalArgumentException("Port must be between 1 and 65535.");
            boolean loopback = Set.of("127.0.0.1", "localhost", "::1").contains(host);
            NetworkPolicy network = new NetworkPolicy(!loopback, port, publicUrl);
            if (!loopback && !host.equals("0.0.0.0") && !network.lanAuthorities().contains(host + ":" + port))
                throw new IllegalArgumentException("Bind to loopback, 0.0.0.0, or an active private IPv4 interface.");
            Path documentRoot = root.toRealPath();
            if (!Files.isRegularFile(documentRoot.resolve("index.html"))) {
                throw new IllegalArgumentException("dist/index.html is missing. Extract the complete deployment package.");
            }
            HttpServer server = HttpServer.create(new InetSocketAddress(host, port), 64);
            AlgorithmService algorithms = new AlgorithmService(documentRoot.getParent());
            BusinessService business = new BusinessService(documentRoot.getParent(),algorithms);
            var executor = Executors.newFixedThreadPool(Math.max(4, Math.min(16, Runtime.getRuntime().availableProcessors())));
            server.setExecutor(executor);
            server.createContext("/", exchange -> handle(exchange, documentRoot, algorithms, business, network));
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                server.stop(1);
                business.close();
                algorithms.close();
                executor.shutdownNow();
            }));
            server.start();
            System.out.println("\nAudio Toolkit Short - Classic Studio");
            System.out.println("Open: http://" + (host.equals("0.0.0.0") ? "localhost" : host) + ":" + port + "/");
            if (!loopback) {
                if (publicUrl != null && !publicUrl.isBlank()) System.out.println("Deployment URL: " + publicUrl);
                network.lanAuthorities().stream().sorted().forEach(address -> System.out.println("LAN: http://" + address + "/"));
                System.out.println("LAN access only. Task and file ownership is enforced for each browser session.");
            }
            System.out.println(business.auth.publicLan()?"Mode: LAN public tools with private visitor sessions. Administrator login remains protected.":"Mode: account business service with private task ownership.");
            System.out.println("Press Ctrl+C to stop.\n");
        } catch (BindException error) {
            System.err.println("Port is already in use. Try: Start-Local.cmd --port 8081");
            System.exit(1);
        } catch (Exception error) {
            System.err.println("Could not start Audio Toolkit: " + error.getMessage());
            System.exit(1);
        }
    }

    private static void handle(HttpExchange exchange, Path root, AlgorithmService algorithms, BusinessService business, NetworkPolicy network) throws IOException {
        try (exchange) {
            var headers = exchange.getResponseHeaders();
            headers.set("X-Content-Type-Options", "nosniff");
            headers.set("X-Frame-Options", "DENY");
            headers.set("Referrer-Policy", "same-origin");
            headers.set("Cache-Control", "no-cache");
            String requestPath = exchange.getRequestURI().getPath();
            boolean apiRequest = requestPath.startsWith("/api/") || requestPath.startsWith("/auth/");
            // OAuth's cross-site GET is protected by one-use state + PKCE + state cookie.
            boolean oauthReturn=requestPath.equals("/auth/google/callback") && exchange.getRequestMethod().equals("GET");
            if (!network.allows(exchange.getRemoteAddress().getAddress(), exchange.getRequestHeaders().getFirst("Host"),
                    apiRequest && !oauthReturn ? exchange.getRequestHeaders().getFirst("Origin") : null,
                    apiRequest && !oauthReturn ? exchange.getRequestHeaders().getFirst("Sec-Fetch-Site") : null)) {
                respond(exchange, 403, "application/json; charset=utf-8", "{\"success\":false,\"message\":\"请从本机或局域网地址打开工作室，禁止跨站调用。\"}");
                return;
            }
            if (apiRequest) {
                headers.set("Cache-Control","no-store");
                try {
                    if(!business.handle(exchange)) BusinessAuth.reply(exchange,404,Map.of("success",false,"message","接口不存在"));
                } catch(BusinessAuth.Failure error) {
                    BusinessAuth.reply(exchange,error.status,Map.of("success",false,"message",error.getMessage()));
                } catch(Exception error) {
                    System.err.println("Business request failed: "+error.getClass().getSimpleName());
                    BusinessAuth.reply(exchange,500,Map.of("success",false,"message","业务服务暂不可用，请查看服务器日志"));
                }
                return;
            }
            String method = exchange.getRequestMethod();
            if (!method.equals("GET") && !method.equals("HEAD")) {
                headers.set("Allow", "GET, HEAD");
                respond(exchange, 405, "application/json", "{\"error\":\"method_not_allowed\"}");
                return;
            }
            // URI.getPath decodes percent escapes. Refuse Windows alternate path forms.
            if (requestPath.indexOf('\0') >= 0 || requestPath.contains("\\") || requestPath.contains(":")) {
                respond(exchange, 403, "text/plain", "Forbidden");
                return;
            }
            Path candidate = root.resolve(requestPath.substring(1)).normalize();
            if (!candidate.startsWith(root)) {
                respond(exchange, 403, "text/plain", "Forbidden");
                return;
            }
            if (Files.isDirectory(candidate)) candidate = candidate.resolve("index.html");
            if (!Files.exists(candidate)) {
                String leaf = candidate.getFileName().toString();
                if (!leaf.contains(".")) candidate = root.resolve("index.html");
                else {
                    respond(exchange, 404, "text/plain", "Not found");
                    return;
                }
            }
            Path file = candidate.toRealPath();
            if (!file.startsWith(root) || !Files.isRegularFile(file)) {
                respond(exchange, 403, "text/plain", "Forbidden");
                return;
            }
            String name = file.getFileName().toString();
            String extension = name.contains(".") ? name.substring(name.lastIndexOf('.') + 1).toLowerCase() : "";
            headers.set("Content-Type", MIME.getOrDefault(extension, "application/octet-stream"));
            long size = Files.size(file);
            headers.set("Content-Length", Long.toString(size));
            if (method.equals("HEAD")) exchange.sendResponseHeaders(200, -1);
            else {
                exchange.sendResponseHeaders(200, size);
                Files.copy(file, exchange.getResponseBody());
            }
        } catch (IOException error) {
            // A disconnected client must not terminate the host.
            System.err.println("Request could not be completed: " + error.getClass().getSimpleName());
        }
    }

    private static void respond(HttpExchange exchange, int status, String type, String value) throws IOException {
        byte[] body = value.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", type);
        if (exchange.getRequestMethod().equals("HEAD")) {
            exchange.getResponseHeaders().set("Content-Length", Integer.toString(body.length));
            exchange.sendResponseHeaders(status, -1);
        } else {
            exchange.sendResponseHeaders(status, body.length);
            exchange.getResponseBody().write(body);
        }
    }
}
