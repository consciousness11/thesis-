/**
 * TCP client used by {@link PPOSchedulingAlgorithm} to talk to the Python
 * sidecar that hosts the PPO + GAT policy.
 *
 * Wire format: 4-byte big-endian length prefix followed by a UTF-8 JSON body
 * shaped like {"type": "OBSERVE", ...payload fields}.  See the plan file for
 * the full frame catalogue.
 *
 * JSON is hand-rolled because WorkflowSim ships with no JSON dependency in
 * {@code lib/}.  The format we exchange is small enough (flat objects with
 * primitives and arrays of primitives or arrays-of-pairs) that a full parser
 * is overkill.
 */
package org.workflowsim.scheduling;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.cloudbus.cloudsim.Log;

public class RLBridge {

    private static RLBridge instance;

    private final String host;
    private final int port;
    private Socket socket;
    private DataInputStream in;
    private DataOutputStream out;
    private boolean healthy;

    private RLBridge(String host, int port) {
        this.host = host;
        this.port = port;
    }

    public static synchronized RLBridge get() {
        if (instance == null) {
            String host = System.getProperty("ppo.host", "127.0.0.1");
            int port = Integer.parseInt(System.getProperty("ppo.port", "7777"));
            instance = new RLBridge(host, port);
        }
        return instance;
    }

    /** Lazily connects on first use; idempotent after success. */
    public synchronized boolean connect() {
        if (healthy) {
            return true;
        }
        try {
            socket = new Socket(host, port);
            socket.setTcpNoDelay(true);
            in = new DataInputStream(socket.getInputStream());
            out = new DataOutputStream(socket.getOutputStream());
            healthy = true;
            Map<String, Object> hello = new LinkedHashMap<>();
            hello.put("type", "HELLO");
            hello.put("schema", 1);
            send(hello);
            Log.printLine("[RLBridge] connected to " + host + ":" + port);
            return true;
        } catch (IOException e) {
            Log.printLine("[RLBridge] connect failed: " + e.getMessage());
            healthy = false;
            return false;
        }
    }

    public synchronized boolean isHealthy() {
        return healthy;
    }

    public synchronized void close() {
        try {
            if (socket != null) {
                socket.close();
            }
        } catch (IOException ignored) {
        }
        healthy = false;
    }

    public synchronized void send(Map<String, Object> frame) throws IOException {
        if (!healthy) {
            throw new IOException("bridge not connected");
        }
        String body = JsonWriter.write(frame);
        byte[] bytes = body.getBytes("UTF-8");
        out.writeInt(bytes.length);
        out.write(bytes);
        out.flush();
    }

    public synchronized Map<String, Object> recv() throws IOException {
        if (!healthy) {
            throw new IOException("bridge not connected");
        }
        int len = in.readInt();
        if (len < 0 || len > 16 * 1024 * 1024) {
            throw new IOException("invalid frame length " + len);
        }
        byte[] buf = new byte[len];
        in.readFully(buf);
        return JsonParser.parseObject(new String(buf, "UTF-8"));
    }

    /**
     * Minimal JSON writer for the subset of values we emit: String, Number,
     * Boolean, List, Map, int[], and array-of-int-pairs (encoded as
     * {@code List<int[]>}).
     */
    static final class JsonWriter {

        static String write(Object v) {
            StringBuilder sb = new StringBuilder();
            writeValue(sb, v);
            return sb.toString();
        }

        @SuppressWarnings("unchecked")
        private static void writeValue(StringBuilder sb, Object v) {
            if (v == null) {
                sb.append("null");
            } else if (v instanceof String) {
                writeString(sb, (String) v);
            } else if (v instanceof Boolean) {
                sb.append(((Boolean) v) ? "true" : "false");
            } else if (v instanceof Number) {
                Number n = (Number) v;
                double d = n.doubleValue();
                if (Double.isNaN(d) || Double.isInfinite(d)) {
                    sb.append("0");
                } else if (n instanceof Integer || n instanceof Long) {
                    sb.append(n.toString());
                } else {
                    sb.append(Double.toString(d));
                }
            } else if (v instanceof int[]) {
                sb.append('[');
                int[] arr = (int[]) v;
                for (int i = 0; i < arr.length; i++) {
                    if (i > 0) sb.append(',');
                    sb.append(arr[i]);
                }
                sb.append(']');
            } else if (v instanceof double[]) {
                sb.append('[');
                double[] arr = (double[]) v;
                for (int i = 0; i < arr.length; i++) {
                    if (i > 0) sb.append(',');
                    sb.append(arr[i]);
                }
                sb.append(']');
            } else if (v instanceof List) {
                sb.append('[');
                List<Object> list = (List<Object>) v;
                for (int i = 0; i < list.size(); i++) {
                    if (i > 0) sb.append(',');
                    writeValue(sb, list.get(i));
                }
                sb.append(']');
            } else if (v instanceof Map) {
                sb.append('{');
                Map<String, Object> map = (Map<String, Object>) v;
                boolean first = true;
                for (Map.Entry<String, Object> e : map.entrySet()) {
                    if (!first) sb.append(',');
                    first = false;
                    writeString(sb, e.getKey());
                    sb.append(':');
                    writeValue(sb, e.getValue());
                }
                sb.append('}');
            } else {
                writeString(sb, v.toString());
            }
        }

        private static void writeString(StringBuilder sb, String s) {
            sb.append('"');
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '"': sb.append("\\\""); break;
                    case '\\': sb.append("\\\\"); break;
                    case '\n': sb.append("\\n"); break;
                    case '\r': sb.append("\\r"); break;
                    case '\t': sb.append("\\t"); break;
                    default:
                        if (c < 0x20) {
                            sb.append(String.format("\\u%04x", (int) c));
                        } else {
                            sb.append(c);
                        }
                }
            }
            sb.append('"');
        }
    }

    /**
     * Minimal recursive-descent JSON parser for the subset Python sends back:
     * flat objects with String keys and values that are String, Number,
     * Boolean, null, or arrays of those.
     */
    static final class JsonParser {

        private final String src;
        private int pos;

        private JsonParser(String src) {
            this.src = src;
        }

        static Map<String, Object> parseObject(String s) {
            JsonParser p = new JsonParser(s);
            p.skipWs();
            Object o = p.parseValue();
            if (!(o instanceof Map)) {
                throw new IllegalArgumentException("expected JSON object at top level");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> m = (Map<String, Object>) o;
            return m;
        }

        private Object parseValue() {
            skipWs();
            if (pos >= src.length()) {
                throw new IllegalArgumentException("unexpected EOF");
            }
            char c = src.charAt(pos);
            if (c == '{') return parseObj();
            if (c == '[') return parseArr();
            if (c == '"') return parseString();
            if (c == 't' || c == 'f') return parseBool();
            if (c == 'n') return parseNull();
            return parseNumber();
        }

        private Map<String, Object> parseObj() {
            Map<String, Object> m = new LinkedHashMap<>();
            pos++;
            skipWs();
            if (peek() == '}') { pos++; return m; }
            while (true) {
                skipWs();
                String key = parseString();
                skipWs();
                expect(':');
                Object v = parseValue();
                m.put(key, v);
                skipWs();
                if (peek() == ',') { pos++; continue; }
                expect('}');
                return m;
            }
        }

        private List<Object> parseArr() {
            List<Object> list = new ArrayList<>();
            pos++;
            skipWs();
            if (peek() == ']') { pos++; return list; }
            while (true) {
                list.add(parseValue());
                skipWs();
                if (peek() == ',') { pos++; continue; }
                expect(']');
                return list;
            }
        }

        private String parseString() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (pos < src.length()) {
                char c = src.charAt(pos++);
                if (c == '"') return sb.toString();
                if (c == '\\') {
                    char esc = src.charAt(pos++);
                    switch (esc) {
                        case '"': sb.append('"'); break;
                        case '\\': sb.append('\\'); break;
                        case '/': sb.append('/'); break;
                        case 'n': sb.append('\n'); break;
                        case 'r': sb.append('\r'); break;
                        case 't': sb.append('\t'); break;
                        case 'u':
                            sb.append((char) Integer.parseInt(src.substring(pos, pos + 4), 16));
                            pos += 4;
                            break;
                        default: sb.append(esc);
                    }
                } else {
                    sb.append(c);
                }
            }
            throw new IllegalArgumentException("unterminated string");
        }

        private Boolean parseBool() {
            if (src.startsWith("true", pos)) { pos += 4; return Boolean.TRUE; }
            if (src.startsWith("false", pos)) { pos += 5; return Boolean.FALSE; }
            throw new IllegalArgumentException("bad bool at " + pos);
        }

        private Object parseNull() {
            if (src.startsWith("null", pos)) { pos += 4; return null; }
            throw new IllegalArgumentException("bad null at " + pos);
        }

        private Object parseNumber() {
            int start = pos;
            if (peek() == '-') pos++;
            while (pos < src.length() && "0123456789.eE+-".indexOf(src.charAt(pos)) >= 0) {
                pos++;
            }
            String tok = src.substring(start, pos);
            if (tok.contains(".") || tok.contains("e") || tok.contains("E")) {
                return Double.parseDouble(tok);
            }
            try {
                return Integer.parseInt(tok);
            } catch (NumberFormatException ex) {
                return Long.parseLong(tok);
            }
        }

        private void expect(char c) {
            skipWs();
            if (pos >= src.length() || src.charAt(pos) != c) {
                throw new IllegalArgumentException("expected '" + c + "' at " + pos);
            }
            pos++;
        }

        private char peek() {
            return pos < src.length() ? src.charAt(pos) : '\0';
        }

        private void skipWs() {
            while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) pos++;
        }
    }
}
