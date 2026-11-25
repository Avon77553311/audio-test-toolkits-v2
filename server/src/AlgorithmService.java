import com.sun.net.httpserver.HttpExchange;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.SecureRandom;
import java.time.Duration;
import java.util.Base64;
import java.util.Set;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** Owns a private Python child. Only Java's same-origin API is browser-facing. */
final class AlgorithmService implements AutoCloseable {
    private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
    private final String token;
    private Process process;
    private int port;
    private static final long MAX_BODY = 102L * 1024 * 1024;

    AlgorithmService(Path project) throws IOException {
        byte[] secret = new byte[32];
        new SecureRandom().nextBytes(secret);
        token = Base64.getUrlEncoder().withoutPadding().encodeToString(secret);
        if (!Files.isRegularFile(project.resolve("worker/app.py"))) return;
        try (ServerSocket socket = new ServerSocket(0, 0, InetAddress.getLoopbackAddress())) {
            port = socket.getLocalPort();
        }
        Path localPython = project.resolve(System.getProperty("os.name").startsWith("Windows") ? ".venv/Scripts/python.exe" : ".venv/bin/python");
        String python = System.getenv("STUDIO_PYTHON");
        if (python == null || python.isBlank()) python = Files.isRegularFile(localPython) ? localPython.toString() : "python";
        Path runtime = project.resolve("runtime");
        Files.createDirectories(runtime);
        ProcessBuilder builder = new ProcessBuilder(python, "-u", project.resolve("worker/app.py").toString(), "--port", Integer.toString(port));
        builder.directory(project.toFile());
        builder.environment().put("STUDIO_WORKER_TOKEN", token);
        builder.environment().put("PYTHONIOENCODING", "utf-8");
        builder.environment().put("STUDIO_WORKER_CONCURRENCY", "32");
        builder.redirectErrorStream(true).redirectOutput(ProcessBuilder.Redirect.appendTo(runtime.resolve("worker.log").toFile()));
        process = builder.start();
        long deadline = System.nanoTime() + Duration.ofSeconds(20).toNanos();
        while (process.isAlive() && System.nanoTime() < deadline) {
            try {
                var response = client.send(HttpRequest.newBuilder(uri("/api/health")).header("X-Studio-Token", token).timeout(Duration.ofSeconds(1)).GET().build(), HttpResponse.BodyHandlers.discarding());
                if (response.statusCode() == 200) {
                    System.out.println("Algorithm service connected (private loopback worker).");
                    return;
                }
            } catch (Exception ignored) {}
            try { Thread.sleep(150); } catch (InterruptedException error) { Thread.currentThread().interrupt(); break; }
        }
        System.err.println("Audio worker unavailable. Run Setup-Tools.cmd; see runtime/worker.log.");
    }

    private URI uri(String path) { return URI.create("http://127.0.0.1:" + port + path); }

    HttpResponse<byte[]> call(String method,String path,String body) throws Exception {
        var request=HttpRequest.newBuilder(uri(path)).header("X-Studio-Token",token).header("Content-Type","application/json").timeout(Duration.ofSeconds(20));
        request.method(method,body==null?HttpRequest.BodyPublishers.noBody():HttpRequest.BodyPublishers.ofString(body));
        return client.send(request.build(),HttpResponse.BodyHandlers.ofByteArray());
    }

    HttpResponse<byte[]> capture(HttpExchange exchange,Map<String,String> trustedHeaders) throws Exception {
        String length=exchange.getRequestHeaders().getFirst("Content-Length");
        if(length!=null&&Long.parseLong(length)>MAX_BODY)throw new BusinessAuth.Failure(413,"上传内容过大");
        var request=HttpRequest.newBuilder(uri(exchange.getRequestURI().toASCIIString())).header("X-Studio-Token",token).timeout(Duration.ofSeconds(90));
        String type=exchange.getRequestHeaders().getFirst("Content-Type");if(type!=null)request.header("Content-Type",type);
        trustedHeaders.forEach(request::header);
        request.method(exchange.getRequestMethod(),HttpRequest.BodyPublishers.ofInputStream(()->new LimitedStream(exchange.getRequestBody())));
        return client.send(request.build(),HttpResponse.BodyHandlers.ofByteArray());
    }

    void proxy(HttpExchange exchange) throws IOException {
        // AudioStudioServer applies the same network/origin policy to static files and every API.
        if (process == null || !process.isAlive()) {
            json(exchange, 503, "算法服务未启动，请运行 Setup-Tools.cmd 后重新启动网站");
            return;
        }
        String length = exchange.getRequestHeaders().getFirst("Content-Length");
        if (length != null && Long.parseLong(length) > MAX_BODY) { json(exchange, 413, "上传内容过大"); return; }
        String method = exchange.getRequestMethod();
        if (!Set.of("GET", "HEAD", "POST", "DELETE").contains(method)) { json(exchange, 405, "不支持的请求方法"); return; }
        HttpRequest.Builder request = HttpRequest.newBuilder(uri(exchange.getRequestURI().toASCIIString()))
            .header("X-Studio-Token", token).timeout(Duration.ofSeconds(90));
        for (String name : Set.of("Content-Type", "Range", "If-Range")) {
            String value = exchange.getRequestHeaders().getFirst(name);
            if (value != null) request.header(name, value);
        }
        request.method(method, method.equals("GET") || method.equals("HEAD") ? HttpRequest.BodyPublishers.noBody()
            : HttpRequest.BodyPublishers.ofInputStream(() -> new LimitedStream(exchange.getRequestBody())));
        try {
            var response = client.send(request.build(), HttpResponse.BodyHandlers.ofInputStream());
            try (InputStream input = response.body()) {
                for (String name : Set.of("Content-Type", "Content-Disposition", "Content-Range", "Accept-Ranges", "Last-Modified")) {
                    response.headers().firstValue(name).ifPresent(value -> exchange.getResponseHeaders().set(name, value));
                }
                if (method.equals("HEAD")) {
                    response.headers().firstValue("Content-Length").ifPresent(value -> exchange.getResponseHeaders().set("Content-Length", value));
                    exchange.sendResponseHeaders(response.statusCode(), -1);
                } else {
                    long size = response.headers().firstValueAsLong("Content-Length").orElse(0);
                    exchange.sendResponseHeaders(response.statusCode(), size == 0 ? 0 : size);
                    input.transferTo(exchange.getResponseBody());
                }
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            json(exchange, 503, "本地任务连接已中断");
        } catch (IOException error) {
            json(exchange, 503, "音频服务暂不可用，请检查 runtime/worker.log");
        }
    }

    private static void json(HttpExchange exchange, int status, String message) throws IOException {
        byte[] body = ("{\"success\":false,\"message\":\"" + message + "\"}").getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, body.length);
        exchange.getResponseBody().write(body);
    }

    private static final class LimitedStream extends FilterInputStream {
        private long remaining = MAX_BODY;
        LimitedStream(InputStream source) { super(source); }
        @Override public int read() throws IOException { int value = super.read(); if (value != -1 && --remaining < 0) throw new IOException("Upload limit exceeded"); return value; }
        @Override public int read(byte[] data, int offset, int length) throws IOException { int count = super.read(data, offset, length); if (count > 0 && (remaining -= count) < 0) throw new IOException("Upload limit exceeded"); return count; }
    }

    @Override public void close() {
        if (process == null || !process.isAlive()) return;
        try { client.send(HttpRequest.newBuilder(uri("/internal/shutdown")).header("X-Studio-Token", token).timeout(Duration.ofSeconds(10)).POST(HttpRequest.BodyPublishers.noBody()).build(), HttpResponse.BodyHandlers.discarding()); } catch (Exception ignored) {}
        process.destroy();
        try { if (!process.waitFor(5, TimeUnit.SECONDS)) process.destroyForcibly(); }
        catch (InterruptedException error) { Thread.currentThread().interrupt(); process.destroyForcibly(); }
    }
}
