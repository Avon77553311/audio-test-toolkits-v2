import com.sun.net.httpserver.HttpExchange;
import com.google.gson.*;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.PBEKeySpec;
import java.net.*;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.*;
import java.sql.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import static java.nio.charset.StandardCharsets.UTF_8;

final class BusinessAuth {
    record Identity(String id,String login,String name,String role,String csrf,String session) {boolean admin(){return role.equals("admin");}boolean guest(){return role.equals("guest");}}
    static final class Failure extends RuntimeException {final int status;Failure(int status,String message){super(message);this.status=status;}}
    final BusinessDb db;
    final Map<String,String> config=new HashMap<>();
    final Map<String,List<Long>> attempts=new ConcurrentHashMap<>();
    final HttpClient http=HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(8)).build();
    final String setupToken;
    BusinessAuth(BusinessDb db,Path project) throws Exception {
        this.db=db;
        Path env=project.resolve(".env.local");
        if(Files.isRegularFile(env))for(String line:Files.readAllLines(env,UTF_8)) {
            String[] pair=line.strip().split("=",2);
            if(pair.length==2&&!pair[0].startsWith("#"))config.put(pair[0],pair[1].strip().replaceAll("^['\"]|['\"]$",""));
        }
        if(!Set.of("accounts","lan-public").contains(value("STUDIO_ACCESS_MODE","lan-public")))throw new IllegalArgumentException("STUDIO_ACCESS_MODE must be lan-public or accounts");
        setupToken=value("STUDIO_SETUP_TOKEN","");
        if(needsSetup() && setupToken.isBlank()) {
            Path file=project.resolve("runtime/setup-token.txt");
            if(!Files.exists(file)){Files.writeString(file,random(),UTF_8);try{Files.setPosixFilePermissions(file,java.nio.file.attribute.PosixFilePermissions.fromString("rw-------"));}catch(UnsupportedOperationException ignored){}}
            config.put("STUDIO_SETUP_TOKEN",Files.readString(file,UTF_8).strip());
            System.out.println("First administrator setup key: runtime/setup-token.txt (keep this file private).");
        }
    }
    String value(String key,String fallback){return System.getenv().getOrDefault(key,config.getOrDefault(key,fallback));}
    boolean enabled(String key){return Set.of("1","true","yes","on").contains(value(key,"false").toLowerCase(Locale.ROOT));}
    boolean publicLan(){return value("STUDIO_ACCESS_MODE","lan-public").equals("lan-public");}
    boolean google(){return !publicLan()&&!value("GOOGLE_CLIENT_ID","").isBlank()&&!value("GOOGLE_CLIENT_SECRET","").isBlank()&&!value("GOOGLE_REDIRECT_URI","").isBlank();}
    boolean needsSetup() throws Exception{return db.read(c->BusinessDb.one(c,"SELECT id FROM users WHERE role='admin' AND status='active' LIMIT 1"))==null;}
    static String random(){byte[] b=new byte[32];new SecureRandom().nextBytes(b);return Base64.getUrlEncoder().withoutPadding().encodeToString(b);}
    static String digest(String text){try{return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(text.getBytes(UTF_8)));}catch(Exception e){throw new IllegalStateException(e);}}
    static boolean equal(String a,String b){return MessageDigest.isEqual(a.getBytes(UTF_8),b.getBytes(UTF_8));}
    static String password(String plain) throws Exception {
        if(plain.length()<12||plain.length()>256)throw new Failure(400,"密码长度需为 12–256 个字符");
        String salt=random();return salt+":"+derive(plain,salt);
    }
    static String derive(String plain,String salt) throws Exception {
        var spec=new PBEKeySpec(plain.toCharArray(),salt.getBytes(UTF_8),600000,256);
        try{return Base64.getEncoder().encodeToString(SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(spec).getEncoded());}finally{spec.clearPassword();}
    }
    static String cookie(HttpExchange x,String key) {
        for(String line:x.getRequestHeaders().getOrDefault("Cookie",List.of()))for(String part:line.split(";")){String[] pair=part.strip().split("=",2);if(pair.length==2&&pair[0].equals(key))return pair[1];}
        return "";
    }
    void setCookie(HttpExchange x,String name,String token,int seconds,String path) {
        x.getResponseHeaders().add("Set-Cookie",name+"="+token+"; Path="+path+"; HttpOnly; SameSite=Lax; Max-Age="+seconds+(enabled("STUDIO_COOKIE_SECURE")?"; Secure":""));
    }
    Identity identity(HttpExchange x) throws Exception {
        String token=cookie(x,"studio_session");if(token.isEmpty())return null;
        var u=db.read(c->BusinessDb.one(c,"SELECT u.id,u.login,u.name,u.role,s.csrf FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=? AND s.expires>? AND u.status='active'",digest(token),System.currentTimeMillis()));
        return u==null?null:new Identity(BusinessDb.text(u,"id"),BusinessDb.text(u,"login"),BusinessDb.text(u,"name"),BusinessDb.text(u,"role"),BusinessDb.text(u,"csrf"),digest(token));
    }
    Identity require(HttpExchange x) throws Exception {
        Identity user=identity(x);if(user==null||user.guest())throw new Failure(401,"请先登录管理员后台");
        checkCsrf(x,user);return user;
    }
    void checkCsrf(HttpExchange x,Identity user) {
        if(!Set.of("GET","HEAD").contains(x.getRequestMethod())&&!equal(user.csrf(),Objects.toString(x.getRequestHeaders().getFirst("X-CSRF-Token"),"")))throw new Failure(403,"会话校验失败，请刷新页面后重试");
    }
    Identity visitor(HttpExchange x,boolean create)throws Exception {
        String token=cookie(x,"studio_visitor");
        var row=token.isEmpty()?null:db.read(c->BusinessDb.one(c,"SELECT u.id,u.login,u.name,u.role,s.csrf FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=? AND s.expires>? AND u.role='guest' AND u.status='active'",digest(token),System.currentTimeMillis()));
        if(row!=null)return new Identity(BusinessDb.text(row,"id"),BusinessDb.text(row,"login"),BusinessDb.text(row,"name"),"guest",BusinessDb.text(row,"csrf"),digest(token));
        if(!create)return null;
        String key="visitor:"+x.getRemoteAddress().getAddress().getHostAddress();long time=System.currentTimeMillis();
        synchronized(attempts){attempts.entrySet().removeIf(e->e.getValue().stream().allMatch(t->t<time-600000));var a=attempts.computeIfAbsent(key,k->new ArrayList<>());a.removeIf(t->t<time-600000);if(a.size()>=60)throw new Failure(429,"新建访客会话过于频繁，请稍后再试");a.add(time);}
        String id=UUID.randomUUID().toString(),secret=random(),csrf=random();
        db.tx(c->{BusinessDb.update(c,"DELETE FROM sessions WHERE expires<?",time);BusinessDb.update(c,"INSERT INTO users(id,login,name,role,quota,created) VALUES(?,?,?,'guest',0,?)",id,"~visitor-"+id,"局域网访客",BusinessDb.now());BusinessDb.update(c,"INSERT INTO sessions VALUES(?,?,?,?)",digest(secret),id,csrf,time+Duration.ofDays(30).toMillis());return null;});
        setCookie(x,"studio_visitor",secret,2592000,"/");
        return new Identity(id,"~visitor-"+id,"局域网访客","guest",csrf,digest(secret));
    }
    Identity requireTool(HttpExchange x)throws Exception {
        if(!publicLan())return require(x);
        Identity user=identity(x);if(user==null||!user.admin())user=visitor(x,false);
        if(user==null)throw new Failure(401,"访客会话未建立或已过期，请刷新页面");
        checkCsrf(x,user);return user;
    }
    Identity session(HttpExchange x,Map<String,Object> row) throws Exception {
        String token=random(),csrf=random();
        db.tx(c->{BusinessDb.update(c,"DELETE FROM sessions WHERE expires<?",System.currentTimeMillis());BusinessDb.update(c,"INSERT INTO sessions VALUES(?,?,?,?)",digest(token),row.get("id"),csrf,System.currentTimeMillis()+Duration.ofHours(12).toMillis());return null;});
        setCookie(x,"studio_session",token,43200,"/");
        return new Identity(BusinessDb.text(row,"id"),BusinessDb.text(row,"login"),BusinessDb.text(row,"name"),BusinessDb.text(row,"role"),csrf,digest(token));
    }
    static Map<String,Object> publicUser(Identity u){return Map.of("id",u.id(),"login",u.login(),"name",u.name(),"role",u.role());}
    static JsonObject body(HttpExchange x) throws Exception {
        byte[] bytes=x.getRequestBody().readNBytes(65537);if(bytes.length>65536)throw new Failure(413,"请求内容过大");
        try {JsonElement v=JsonParser.parseString(new String(bytes,UTF_8));if(!v.isJsonObject())throw new IllegalArgumentException();return v.getAsJsonObject();}catch(Exception e){throw new Failure(400,"JSON 格式无效");}
    }
    static String text(JsonObject b,String key){return b.has(key)&&b.get(key).isJsonPrimitive()?b.get(key).getAsString():"";}
    static String login(String name){String n=name.strip().toLowerCase(Locale.ROOT);if(!n.matches("[a-z0-9][a-z0-9_.@+-]{2,119}"))throw new Failure(400,"账号需为 3–120 位字母、数字或邮箱字符");return n;}
    void limit(HttpExchange x,String login) {
        String key=x.getRemoteAddress().getAddress().getHostAddress()+":"+login;long now=System.currentTimeMillis();
        synchronized(attempts){if(attempts.size()>10000)attempts.entrySet().removeIf(e->e.getValue().stream().allMatch(t->t<now-300000));var a=attempts.computeIfAbsent(key,k->new ArrayList<>());a.removeIf(t->t<now-300000);if(a.size()>=10)throw new Failure(429,"尝试过于频繁，请五分钟后重试");a.add(now);}
    }
    boolean handle(HttpExchange x,String path) throws Exception {
        String method=x.getRequestMethod();
        if(path.equals("/api/me")&&method.equals("GET")) {
            Identity u=identity(x);if(publicLan()&&u!=null&&!u.admin())u=null;
            Identity v=publicLan()&&u==null?visitor(x,true):null;
            Map<String,Object> result=new LinkedHashMap<>();result.put("success",true);result.put("user",u==null?null:publicUser(u));result.put("visitor",v==null?null:publicUser(v));result.put("accessMode",publicLan()?"lan-public":"accounts");result.put("csrf",u!=null?u.csrf():v!=null?v.csrf():"");result.put("setupRequired",needsSetup());result.put("googleAvailable",google());reply(x,200,result);return true;
        }
        if(path.equals("/api/auth/bootstrap")&&method.equals("POST")) {
            var b=body(x);limit(x,"bootstrap");
            if(!equal(text(b,"setupToken"),value("STUDIO_SETUP_TOKEN",setupToken)))throw new Failure(403,"初始化密钥无效，请查看服务器 runtime/setup-token.txt");
            String name=login(text(b,"login")),hash=password(text(b,"password")),id=UUID.randomUUID().toString();
            var row=db.tx(c->{if(BusinessDb.one(c,"SELECT id FROM users WHERE role='admin' AND status='active'")!=null)throw new Failure(409,"管理员已初始化");BusinessDb.update(c,"INSERT INTO users(id,login,name,password,role,created) VALUES(?,?,?,?,?,?)",id,name,name,hash,"admin",BusinessDb.now());BusinessDb.audit(c,id,"bootstrap",id,Map.of());return BusinessDb.one(c,"SELECT * FROM users WHERE id=?",id);});
            Identity u=session(x,row);reply(x,201,Map.of("success",true,"user",publicUser(u),"csrf",u.csrf()));return true;
        }
        if(path.equals("/api/auth/login")&&method.equals("POST")) {
            var b=body(x);String name=login(text(b,"login")),plain=text(b,"password");limit(x,name);
            if(plain.length()>256)throw new Failure(400,"密码长度无效");
            var row=db.read(c->BusinessDb.one(c,"SELECT * FROM users WHERE login=? AND status='active'",name));
            String hash=row==null?"":BusinessDb.text(row,"password");String[] split=hash.split(":",2);
            String actual=derive(plain,split.length==2?split[0]:"dummy-login-salt");
            if(row==null||split.length!=2||!equal(actual,split[1]))throw new Failure(401,"账号或密码不正确");
            if(publicLan()&&!BusinessDb.text(row,"role").equals("admin"))throw new Failure(403,"工具已开放免登录，此入口仅允许管理员登录");
            Identity u=session(x,row);reply(x,200,Map.of("success",true,"user",publicUser(u),"csrf",u.csrf()));return true;
        }
        if(Set.of("/auth/logout","/api/auth/logout").contains(path)&&method.equals("POST")) {
            Identity u=require(x);db.tx(c->{BusinessDb.update(c,"DELETE FROM sessions WHERE token=?",u.session());return null;});setCookie(x,"studio_session","",0,"/");reply(x,200,Map.of("success",true));return true;
        }
        if(path.equals("/auth/google")&&method.equals("GET")) {
            if(!google())throw new Failure(503,"Google 登录尚未配置");
            String state=random(),verifier=random();
            db.tx(c->{BusinessDb.update(c,"DELETE FROM oauth_states WHERE expires<?",System.currentTimeMillis());BusinessDb.update(c,"INSERT INTO oauth_states VALUES(?,?,?)",digest(state),verifier,System.currentTimeMillis()+600000);return null;});
            setCookie(x,"studio_oauth",state,600,"/auth/google");
            String challenge=Base64.getUrlEncoder().withoutPadding().encodeToString(MessageDigest.getInstance("SHA-256").digest(verifier.getBytes(UTF_8)));
            redirect(x,"https://accounts.google.com/o/oauth2/v2/auth?"+form(Map.of("client_id",value("GOOGLE_CLIENT_ID",""),"redirect_uri",value("GOOGLE_REDIRECT_URI",""),"response_type","code","scope","openid email profile","state",state,"code_challenge",challenge,"code_challenge_method","S256")));return true;
        }
        if(path.equals("/auth/google/callback")&&method.equals("GET")) {
            if(!google())throw new Failure(503,"Google 登录尚未配置");
            Map<String,String> q=query(x);String state=q.getOrDefault("state","");
            if(state.isBlank()||!equal(state,cookie(x,"studio_oauth")))throw new Failure(403,"登录状态校验失败");
            var stored=db.tx(c->{var v=BusinessDb.one(c,"SELECT verifier FROM oauth_states WHERE state=? AND expires>?",digest(state),System.currentTimeMillis());BusinessDb.update(c,"DELETE FROM oauth_states WHERE state=?",digest(state));return v;});
            if(stored==null)throw new Failure(403,"登录请求已过期或已经使用");
            var request=HttpRequest.newBuilder(URI.create("https://oauth2.googleapis.com/token")).timeout(Duration.ofSeconds(15)).header("Content-Type","application/x-www-form-urlencoded").POST(HttpRequest.BodyPublishers.ofString(form(Map.of("client_id",value("GOOGLE_CLIENT_ID",""),"client_secret",value("GOOGLE_CLIENT_SECRET",""),"code",q.getOrDefault("code",""),"redirect_uri",value("GOOGLE_REDIRECT_URI",""),"grant_type","authorization_code","code_verifier",BusinessDb.text(stored,"verifier"))))).build();
            var token=http.send(request,HttpResponse.BodyHandlers.ofString());if(token.statusCode()!=200)throw new Failure(502,"Google 授权失败，请重新登录");
            String access=JsonParser.parseString(token.body()).getAsJsonObject().get("access_token").getAsString();
            var response=http.send(HttpRequest.newBuilder(URI.create("https://openidconnect.googleapis.com/v1/userinfo")).timeout(Duration.ofSeconds(15)).header("Authorization","Bearer "+access).GET().build(),HttpResponse.BodyHandlers.ofString());
            if(response.statusCode()!=200)throw new Failure(502,"无法验证 Google 身份");
            var profile=JsonParser.parseString(response.body()).getAsJsonObject();
            if(!profile.has("email_verified")||!profile.get("email_verified").getAsBoolean())throw new Failure(403,"Google 邮箱尚未验证");
            String email=login(text(profile,"email")),sub=text(profile,"sub");if(sub.isBlank())throw new Failure(403,"Google 身份无效");
            Set<String> domains=new HashSet<>(Arrays.asList(value("GOOGLE_ALLOWED_DOMAINS","").toLowerCase().split(",")));domains.remove("");
            if(!domains.isEmpty()&&domains.stream().noneMatch(d->email.endsWith("@"+d.strip())))throw new Failure(403,"此邮箱域名不允许登录");
            var row=db.tx(c->{var u=BusinessDb.one(c,"SELECT * FROM users WHERE google_sub=?",sub);if(u==null){if(BusinessDb.one(c,"SELECT id FROM users WHERE login=?",email)!=null)throw new Failure(409,"此邮箱已有本地账号，请使用原登录方式");String id=UUID.randomUUID().toString();BusinessDb.update(c,"INSERT INTO users(id,login,name,google_sub,created) VALUES(?,?,?,?,?)",id,email,text(profile,"name").isBlank()?email:text(profile,"name"),sub,BusinessDb.now());u=BusinessDb.one(c,"SELECT * FROM users WHERE id=?",id);}if(!BusinessDb.text(u,"status").equals("active"))throw new Failure(403,"账户不可用");return u;});
            session(x,row);setCookie(x,"studio_oauth","",0,"/auth/google");redirect(x,"/");return true;
        }
        return false;
    }
    static Map<String,String> query(HttpExchange x){Map<String,String> values=new HashMap<>();String q=x.getRequestURI().getRawQuery();if(q!=null)for(String s:q.split("&")){String[] a=s.split("=",2);values.put(URLDecoder.decode(a[0],UTF_8),a.length==2?URLDecoder.decode(a[1],UTF_8):"");}return values;}
    static String form(Map<String,String> values){return values.entrySet().stream().map(e->URLEncoder.encode(e.getKey(),UTF_8)+"="+URLEncoder.encode(e.getValue(),UTF_8)).reduce((a,b)->a+"&"+b).orElse("");}
    static void redirect(HttpExchange x,String to)throws Exception{x.getResponseHeaders().set("Location",to);x.sendResponseHeaders(302,-1);}
    static void reply(HttpExchange x,int status,Object data)throws java.io.IOException{byte[] body=BusinessDb.JSON.toJson(data).getBytes(UTF_8);x.getResponseHeaders().set("Content-Type","application/json; charset=utf-8");x.sendResponseHeaders(status,body.length);x.getResponseBody().write(body);}
}
